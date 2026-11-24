"""
infer_groq.py — Llama 3.3 70B inference via Groq API with multi-account key rotation.

Implements KeyRotator for round-robin key management (1-4 accounts) with per-key
rate limit and token budget tracking. All predictions are cached atomically to JSON.
"""

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from groq import Groq, RateLimitError, InternalServerError

from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.evaluate import (
    normalize_label,
    compute_metrics,
    save_results,
    reset_parse_failures,
    get_parse_failures,
    save_parse_failures,
)

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"




# --------------------------------------------------------------------------- #
#  Exceptions                                                                   #
# --------------------------------------------------------------------------- #

class AllKeysExhausted(Exception):
    """Raised when all API keys are rate-limited or token-exhausted."""
    pass


# --------------------------------------------------------------------------- #
#  Key Rotation                                                                 #
# --------------------------------------------------------------------------- #

class KeyRotator:
    """Round-robin API key manager with per-key rate limit and token tracking.

    Manages 1-4 Groq API keys across separate accounts. Each key has independent
    rate limits (30 RPM, 100K TPD). The rotator tracks per-key usage and
    cooldown state.

    Args:
        api_keys: List of 1-4 API keys (each from a different Groq account).
        daily_token_budget: Per-key daily token cap. Default 95,000 (safe margin).
        cooldown_seconds: How long to wait after a 429. Default 60.
    """

    def __init__(
        self,
        api_keys: list[str],
        daily_token_budget: int = 95_000,
        cooldown_seconds: int = 60,
    ):
        assert 1 <= len(api_keys) <= 4, (
            f"KeyRotator requires between 1 and 4 API keys, got {len(api_keys)}"
        )

        self.keys = api_keys
        self.daily_token_budget = daily_token_budget
        self.cooldown_seconds = cooldown_seconds
        self.current_index = 0

        # Pre-initialize Groq clients
        self.clients: dict[str, Groq] = {}
        for key in api_keys:
            self.clients[key] = Groq(api_key=key)

        # Per-key state
        self.state: dict[str, dict] = {}
        for key in api_keys:
            self.state[key] = {
                "request_count": 0,
                "token_count": 0,
                "first_request_time": None,
                "last_429_time": None,
                "exhausted_for_day": False,
            }

    def get_active_key(self) -> tuple[str, Groq]:
        """Return (key, client) for the next available key.

        Round-robin starting from current_index. Skips keys in cooldown or
        exhausted. Resets exhausted keys after 24h from first_request_time.

        Raises:
            AllKeysExhausted: If all keys are in cooldown or exhausted.
        """
        now = datetime.now(timezone.utc)
        attempts = 0

        while attempts < len(self.keys):
            key = self.keys[self.current_index % len(self.keys)]
            self.current_index = (self.current_index + 1) % len(self.keys)
            state = self.state[key]

            # Check if 24h rolling window has passed → reset
            if (
                state["exhausted_for_day"]
                and state["first_request_time"] is not None
                and (now - state["first_request_time"]).total_seconds() >= 86400
            ):
                logger.info(f"Key {key[:8]}...: 24h window passed, resetting counters")
                state["request_count"] = 0
                state["token_count"] = 0
                state["first_request_time"] = None
                state["exhausted_for_day"] = False

            # Skip if exhausted for the day
            if state["exhausted_for_day"]:
                attempts += 1
                continue

            # Skip if in 429 cooldown
            if state["last_429_time"] is not None:
                elapsed = (now - state["last_429_time"]).total_seconds()
                if elapsed < self.cooldown_seconds:
                    attempts += 1
                    continue
                else:
                    state["last_429_time"] = None  # Cooldown expired

            return key, self.clients[key]

        raise AllKeysExhausted(
            "All API keys are currently rate-limited or token-exhausted. "
            f"Usage: {self.get_usage_summary()}"
        )

    def mark_rate_limited(self, key: str) -> None:
        """Record a 429 event. Sets last_429_time = now."""
        self.state[key]["last_429_time"] = datetime.now(timezone.utc)
        logger.warning(f"Key {key[:8]}...: rate limited (429)")

    def mark_token_budget_used(self, key: str) -> None:
        """Mark key as exhausted for the day."""
        self.state[key]["exhausted_for_day"] = True
        logger.warning(
            f"Key {key[:8]}...: daily token budget exhausted "
            f"({self.state[key]['token_count']}/{self.daily_token_budget})"
        )

    def record_usage(self, key: str, tokens_used: int) -> None:
        """Update request_count and token_count for this key."""
        state = self.state[key]
        state["request_count"] += 1

        if state["first_request_time"] is None:
            state["first_request_time"] = datetime.now(timezone.utc)

        state["token_count"] += tokens_used

        if state["token_count"] >= self.daily_token_budget:
            self.mark_token_budget_used(key)

    def get_usage_summary(self) -> dict:
        """Return per-key usage stats for logging."""
        summary = {}
        for key in self.keys:
            s = self.state[key]
            summary[key[:8] + "..."] = {
                "requests": s["request_count"],
                "tokens": s["token_count"],
                "exhausted": s["exhausted_for_day"],
                "in_cooldown": s["last_429_time"] is not None,
            }
        return summary


# --------------------------------------------------------------------------- #
#  Single Prediction                                                            #
# --------------------------------------------------------------------------- #

def predict_single(
    rotator: KeyRotator,
    text: str,
    max_retries: int = 5,
) -> tuple[str, dict, int]:
    """Predict sentiment for a single text using Groq API.

    Handles 429 (rotate to next key) and 5xx (exponential backoff on same key).

    Args:
        rotator: KeyRotator instance.
        text: Tweet text to classify.
        max_retries: Max total retry attempts across all error types.

    Returns:
        (raw_output_text, full_response_dict, tokens_used)

    Raises:
        AllKeysExhausted: If all retries exhausted with all keys unavailable.
    """
    for attempt in range(max_retries):
        try:
            key, client = rotator.get_active_key()
        except AllKeysExhausted:
            if attempt < max_retries - 1:
                wait = 60
                logger.info(f"All keys exhausted, sleeping {wait}s before retry...")
                time.sleep(wait)
                continue
            raise

        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(text)},
                ],
                temperature=0,
                max_tokens=10,
            )
            latency = time.perf_counter() - start

            raw_output = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            rotator.record_usage(key, tokens_used)

            # Build serializable response dict
            response_dict = {
                "model": GROQ_MODEL,
                "raw_output": raw_output,
                "tokens_used": tokens_used,
                "latency_seconds": round(latency, 4),
                "key_prefix": key[:8],
            }

            return raw_output, response_dict, tokens_used

        except RateLimitError:
            rotator.mark_rate_limited(key)
            logger.info(f"429 on key {key[:8]}..., rotating to next key...")
            continue  # Retry immediately with next key

        except InternalServerError:
            wait = min(2 ** attempt + random.random(), 60)
            logger.warning(f"5xx on key {key[:8]}..., backoff {wait:.1f}s")
            time.sleep(wait)
            continue

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    raise AllKeysExhausted(f"Max retries ({max_retries}) exceeded")


# --------------------------------------------------------------------------- #
#  Cache Management                                                             #
# --------------------------------------------------------------------------- #

def _load_cache(cache_path: Path) -> dict:
    """Load existing cache file or return empty structure."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Corrupted cache file at {cache_path}, backing up and starting fresh."
            )
            backup_path = cache_path.with_suffix(".json.bak")
            cache_path.replace(backup_path)
    return {"metadata": {}, "predictions": {}}


def _save_cache(cache_path: Path, cache: dict) -> None:
    """Atomic write: write to .tmp, then rename.

    On Windows, antivirus/indexers or sync clients can briefly lock the target
    file. Retry replace a few times to tolerate transient lock contention.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            tmp_path.replace(cache_path)
            return
        except PermissionError:
            if attempt == max_attempts - 1:
                raise
            # Exponential backoff: 0.2s, 0.4s, ... up to ~6.2s total.
            time.sleep(0.2 * (2 ** attempt))


# --------------------------------------------------------------------------- #
#  Full Evaluation                                                              #
# --------------------------------------------------------------------------- #

def run_groq_evaluation(
    rotator: KeyRotator,
    test_dataset,
    original_indices: list[int],
    language: str,
    cache_dir: str = "results/api_cache/",
    seed: int | None = None,
) -> dict:
    """Run full Groq evaluation for one language with caching and resume.

    Cache path: {cache_dir}/llama-3.3-70b-versatile/{language}/predictions.json

    Args:
        rotator: KeyRotator with 1-4 keys.
        test_dataset: 200-sample subsampled HuggingFace Dataset.
        original_indices: From subsample_indices.json.
        language: Language code.
        cache_dir: Base cache directory.

    Returns:
        metrics_dict from compute_metrics().

    Raises:
        AllKeysExhausted: If daily limits hit mid-evaluation (cache is safe).
    """
    cache_path = Path(cache_dir) / GROQ_MODEL / language / "predictions.json"
    cache = _load_cache(cache_path)

    # Initialize metadata
    cache["metadata"] = {
        "model": GROQ_MODEL,
        "language": language,
        "total_samples": len(test_dataset),
        "completed": len(cache.get("predictions", {})),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    if "predictions" not in cache:
        cache["predictions"] = {}

    reset_parse_failures()

    # Identify uncompleted samples
    completed_keys = set(cache["predictions"].keys())
    texts = test_dataset["text"]
    labels = test_dataset["label"]

    total = len(original_indices)
    skipped = 0
    processed = 0

    logger.info(
        f"Groq eval {language}: {total} total, "
        f"{len(completed_keys)} already cached"
    )

    for i, (orig_idx, text, true_label) in enumerate(
        zip(original_indices, texts, labels)
    ):
        idx_str = str(orig_idx)

        # Skip already completed
        if idx_str in completed_keys:
            skipped += 1
            continue

        # Predict (may raise AllKeysExhausted)
        raw_output, response_dict, tokens_used = predict_single(rotator, text)

        # Normalize label
        pred_label = normalize_label(raw_output, "llama70b", language, idx_str)

        # Store in cache
        cache["predictions"][idx_str] = {
            "text": text,
            "true_label": int(true_label),
            "pred_label": pred_label,
            "raw_output": raw_output,
            "tokens_used": tokens_used,
            "latency_seconds": response_dict["latency_seconds"],
        }
        cache["metadata"]["completed"] = len(cache["predictions"])
        cache["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Atomic save after each prediction
        _save_cache(cache_path, cache)
        processed += 1

        if (processed) % 20 == 0:
            logger.info(
                f"  Progress: {processed + skipped}/{total} "
                f"({processed} new, {skipped} cached)"
            )

    logger.info(
        f"Groq eval {language} complete: {processed} new, {skipped} cached"
    )

    # Capture parse failure count BEFORE saving (save resets the list)
    parse_failures = get_parse_failures()
    save_parse_failures("llama70b", language)

    # Compute metrics from all predictions (cached + new)
    y_true = []
    y_pred = []
    total_latency = 0.0
    for idx_str in sorted(cache["predictions"].keys(), key=int):
        entry = cache["predictions"][idx_str]
        y_true.append(entry["true_label"])
        y_pred.append(entry["pred_label"])
        total_latency += entry.get("latency_seconds", 0)

    metrics = compute_metrics(
        y_true,
        y_pred,
        latency_seconds=total_latency,
        parse_failure_count=len(parse_failures),
        total_samples=len(y_true),
    )

    save_results(metrics, "llama70b", language, seed=seed)
    return metrics

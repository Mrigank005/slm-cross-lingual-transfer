"""
infer_gemini.py — Gemini 2.5 Pro inference via Google AI Studio (Tier 1, billing enabled).

Uses the new `google-genai` SDK (replaces deprecated `google-generativeai`).
Single API key with 150 RPM, 1000 RPD. All predictions are cached atomically
to JSON for resume support.

The GeminiClient wraps a google.genai.Client instance. Evaluation is
single-threaded — no concurrent requests.
"""

import json
import logging
import random
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

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

GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_MAX_OUTPUT_TOKENS = 512
_RETRYABLE_EMPTY_FINISH_REASONS = {"MAX_TOKENS", "UNKNOWN"}
_EMPTY_SENTINEL_OUTPUTS = {"[TRUNCATED_EMPTY]", "[EMPTY_RESPONSE]", "[BLOCKED]"}


def _finish_reason_name(reason) -> str:
    """Return a stable finish-reason label for logging/fallback logic."""
    if reason is None:
        return "UNKNOWN"
    # New SDK returns enum-like objects with a .name attribute
    if hasattr(reason, "name"):
        return reason.name
    if isinstance(reason, str):
        return reason
    return str(reason)


def _extract_response_text(response) -> tuple[str, str]:
    """Extract model text from a GenerateContentResponse.

    Returns:
        (text, finish_reason_name)
    """
    candidates = getattr(response, "candidates", None) or []
    candidate = candidates[0] if candidates else None
    finish_reason = _finish_reason_name(
        getattr(candidate, "finish_reason", None) if candidate else None
    )

    # Quick accessor — response.text
    text = ""
    try:
        quick_text = getattr(response, "text", None)
        if quick_text:
            text = quick_text.strip()
    except Exception:
        text = ""

    if text:
        return text, finish_reason

    # Fallback: parse candidate content parts directly.
    if candidate is not None:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
        text = "\n".join(chunks).strip()

    return text, finish_reason


# --------------------------------------------------------------------------- #
#  Client                                                                       #
# --------------------------------------------------------------------------- #

class GeminiClient:
    """Single-key Gemini API client with usage tracking.

    Tier 1 (billing enabled): 150 RPM, 1000 RPD — single key is sufficient
    for 800 total samples (200 per language × 4 languages).

    Args:
        api_key: Google AI Studio API key.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key must not be empty")
        self.api_key = api_key
        self.request_count = 0
        # Create the new-style client
        self._client = genai.Client(api_key=api_key)

    @property
    def client(self) -> genai.Client:
        """Return the underlying google.genai.Client."""
        return self._client

    def record_usage(self) -> None:
        """Increment request count after a successful request."""
        self.request_count += 1

    def get_usage_summary(self) -> dict:
        """Return usage stats for logging."""
        return {"request_count": self.request_count}


class GeminiKeyRotator:
    """Round-robin Gemini key manager with per-key RPM/RPD limits and cooldown.

    Designed for multi-key local evaluation flows where several API keys are
    available (for example GOOGLE_API_KEY_1..3).
    """

    def __init__(
        self,
        api_keys: list[str],
        rpm_limit: int = 150,
        rpd_limit: int = 1000,
        cooldown_seconds: int = 30,
    ):
        if not api_keys:
            raise ValueError("At least one Gemini API key is required")

        self.keys = api_keys
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.cooldown_seconds = cooldown_seconds
        self.clients = {k: genai.Client(api_key=k) for k in self.keys}
        self.state = {
            k: {
                "minute_window": deque(),
                "day_count": 0,
                "day_start": time.time(),
                "cooldown_until": 0.0,
            }
            for k in self.keys
        }
        self.idx = 0

    def _refresh_state(self, key: str, now: float) -> None:
        state = self.state[key]
        while state["minute_window"] and now - state["minute_window"][0] >= 60:
            state["minute_window"].popleft()
        if now - state["day_start"] >= 86400:
            state["day_start"] = now
            state["day_count"] = 0

    def _next_wait_seconds(self, key: str, now: float) -> float:
        state = self.state[key]
        waits = [0.0]

        if state["cooldown_until"] > now:
            waits.append(state["cooldown_until"] - now)
        if len(state["minute_window"]) >= self.rpm_limit:
            waits.append(60 - (now - state["minute_window"][0]))
        if state["day_count"] >= self.rpd_limit:
            waits.append(86400 - (now - state["day_start"]))

        return max(waits)

    def acquire_client(self) -> tuple[str, genai.Client]:
        """Get next available key+client, waiting if all keys are saturated."""
        while True:
            now = time.time()
            best_wait = None

            for _ in range(len(self.keys)):
                key = self.keys[self.idx % len(self.keys)]
                self.idx = (self.idx + 1) % len(self.keys)
                self._refresh_state(key, now)

                wait = self._next_wait_seconds(key, now)
                if wait <= 0:
                    return key, self.clients[key]
                best_wait = wait if best_wait is None else min(best_wait, wait)

            time.sleep(max(best_wait or 1.0, 0.5))

    def mark_success(self, key: str) -> None:
        now = time.time()
        state = self.state[key]
        self._refresh_state(key, now)
        state["minute_window"].append(now)
        state["day_count"] += 1

    def mark_rate_limited(self, key: str, retry_seconds: int | None = None) -> None:
        now = time.time()
        retry = retry_seconds if retry_seconds is not None else self.cooldown_seconds
        self.state[key]["cooldown_until"] = max(
            self.state[key]["cooldown_until"], now + retry
        )

    def usage_summary(self) -> dict:
        summary = {}
        now = time.time()
        for key in self.keys:
            self._refresh_state(key, now)
            state = self.state[key]
            summary[key[:8] + "..."] = {
                "rpm_used": len(state["minute_window"]),
                "rpd_used": state["day_count"],
                "in_cooldown": state["cooldown_until"] > now,
            }
        return summary


class _RotatingModelsProxy:
    """Proxy that injects key-rotation into the models.generate_content() path."""

    def __init__(self, owner: "RotatingGeminiClient"):
        self.owner = owner

    def generate_content(self, *args, **kwargs):
        key, client = self.owner.rotator.acquire_client()
        self.owner._last_key = key
        try:
            return client.models.generate_content(*args, **kwargs)
        except genai_errors.ClientError as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                self.owner.rotator.mark_rate_limited(key, retry_seconds=30)
            raise


class RotatingGeminiClient:
    """Gemini client adapter compatible with run_gemini_evaluation().

    This class keeps the same external interface as GeminiClient:
    - .client.models.generate_content(...)
    - .record_usage()
    - .get_usage_summary()
    """

    def __init__(self, api_keys: list[str]):
        self.rotator = GeminiKeyRotator(
            api_keys=api_keys,
            rpm_limit=150,
            rpd_limit=1000,
            cooldown_seconds=30,
        )
        self._last_key: str | None = None
        self.request_count = 0
        self._models_proxy = _RotatingModelsProxy(self)

    @property
    def client(self):
        class _ClientShim:
            pass

        shim = _ClientShim()
        shim.models = self._models_proxy
        return shim

    def record_usage(self) -> None:
        if self._last_key is not None:
            self.rotator.mark_success(self._last_key)
        self.request_count += 1

    def get_usage_summary(self) -> dict:
        return {
            "request_count": self.request_count,
            "per_key": self.rotator.usage_summary(),
        }


# --------------------------------------------------------------------------- #
#  Single Prediction                                                            #
# --------------------------------------------------------------------------- #

def predict_single(
    client: GeminiClient | RotatingGeminiClient,
    text: str,
    max_retries: int = 5,
) -> tuple[str, dict, int]:
    """Predict sentiment for a single text using Gemini API.

    Handles 429 (backoff), 5xx (backoff), and safety filter blocks.

    Args:
        client: GeminiClient instance.
        text: Tweet text to classify.
        max_retries: Max total retry attempts.

    Returns:
        (raw_output_text, full_response_dict, tokens_used)

    Raises:
        RuntimeError: If max retries exceeded.
    """
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            response = client.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=build_user_prompt(text),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=128,  # Minimal thinking — single-word classification
                    ),
                ),
            )
            latency = time.perf_counter() - start

            client.record_usage()

            raw_output, finish_reason = _extract_response_text(response)

            if not raw_output:
                if finish_reason in {
                    "SAFETY",
                    "RECITATION",
                    "BLOCKLIST",
                    "PROHIBITED_CONTENT",
                    "SPII",
                }:
                    raw_output = "[BLOCKED]"
                elif (
                    finish_reason in _RETRYABLE_EMPTY_FINISH_REASONS
                    and attempt < max_retries - 1
                ):
                    wait = min(2 ** attempt + random.random(), 8)
                    logger.debug(
                        "Gemini returned empty text (finish_reason=%s), retrying in %.1fs (attempt %s/%s)",
                        finish_reason,
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)
                    continue
                elif finish_reason == "MAX_TOKENS":
                    raw_output = "[TRUNCATED_EMPTY]"
                else:
                    raw_output = "[EMPTY_RESPONSE]"

                logger.debug(
                    "Gemini returned no text content (finish_reason=%s) for text: %s...",
                    finish_reason,
                    text[:50],
                )

            # Extract token count
            usage = getattr(response, "usage_metadata", None)
            if usage:
                tokens_used = getattr(usage, "total_token_count", 0) or 0
            else:
                # Fallback: BPE approximation (better than whitespace for non-Latin)
                tokens_used = len(text) // 4 + 20

            response_dict = {
                "model": GEMINI_MODEL,
                "raw_output": raw_output,
                "tokens_used": tokens_used,
                "latency_seconds": round(latency, 4),
            }

            return raw_output, response_dict, tokens_used

        except genai_errors.ClientError as e:
            # 429 rate limit
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = min(60 * (attempt + 1), 300)
                logger.warning(f"429 rate limited, backoff {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            logger.error(f"Gemini client error: {e}")
            raise

        except genai_errors.ServerError:
            wait = min(2 ** attempt + random.random(), 60)
            logger.warning(f"5xx error, backoff {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue

        except Exception as e:
            logger.error(f"Unexpected Gemini error: {e}")
            raise

    raise RuntimeError(f"Max retries ({max_retries}) exceeded for Gemini API")


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

# Polite inter-request delay: 150 RPM = 2.5 req/s, use 0.5s to stay well under
_INTER_REQUEST_DELAY = 0.5  # seconds


def run_gemini_evaluation(
    client: GeminiClient | RotatingGeminiClient,
    test_dataset,
    original_indices: list[int],
    language: str,
    cache_dir: str = "results/api_cache/",
    seed: int | None = None,
    retry_original_indices: list[int] | None = None,
) -> dict:
    """Run full Gemini evaluation for one language with caching and resume.

    Cache path: {cache_dir}/gemini-2.5-pro/{language}/predictions.json

    Tier 1 (billing enabled): 150 RPM, 1000 RPD. Single key handles all
    800 samples (200/language × 4 languages) in well under 1 day.

    Args:
        client: GeminiClient (single key) or RotatingGeminiClient (multi-key).
        test_dataset: 200-sample subsampled Dataset.
        original_indices: From subsample_indices.json.
        language: Language code.
        cache_dir: Base cache directory.
        retry_original_indices: Optional original sample IDs to force re-evaluate.

    Returns:
        metrics_dict from compute_metrics().
    """
    cache_path = Path(cache_dir) / "gemini-2.5-pro" / language / "predictions.json"
    cache = _load_cache(cache_path)

    cache["metadata"] = {
        "model": "gemini-2.5-pro",
        "language": language,
        "total_samples": len(test_dataset),
        "completed": len(cache.get("predictions", {})),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    if "predictions" not in cache:
        cache["predictions"] = {}

    reset_parse_failures()

    # Evict cached entries with failed outputs so they get re-evaluated
    _BAD_OUTPUTS = {"[TRUNCATED_EMPTY]", "[EMPTY_RESPONSE]", "[BLOCKED]"}
    if cache.get("predictions"):
        evicted = [
            k for k, v in cache["predictions"].items()
            if v.get("raw_output") in _BAD_OUTPUTS
        ]
        for k in evicted:
            del cache["predictions"][k]
        if evicted:
            logger.info(f"Evicted {len(evicted)} cached entries with failed outputs for re-evaluation")
            _save_cache(cache_path, cache)

    if retry_original_indices:
        retry_ids = {str(i) for i in retry_original_indices}
        force_evicted = [k for k in list(cache["predictions"].keys()) if k in retry_ids]
        for k in force_evicted:
            del cache["predictions"][k]
        if force_evicted:
            logger.info(
                "Force-evicted %s entries for targeted Gemini re-evaluation: %s",
                len(force_evicted),
                ",".join(sorted(force_evicted, key=int)),
            )
            _save_cache(cache_path, cache)

    completed_keys = set(cache["predictions"].keys())
    texts = test_dataset["text"]
    labels = test_dataset["label"]

    total = len(original_indices)
    skipped = 0
    processed = 0

    logger.info(
        f"Gemini eval {language}: {total} total, "
        f"{len(completed_keys)} already cached"
    )

    last_request_time = 0.0

    for i, (orig_idx, text, true_label) in enumerate(
        zip(original_indices, texts, labels)
    ):
        idx_str = str(orig_idx)

        if idx_str in completed_keys:
            skipped += 1
            continue

        # Polite inter-request throttle
        elapsed = time.time() - last_request_time
        if elapsed < _INTER_REQUEST_DELAY and last_request_time > 0:
            sleep_time = _INTER_REQUEST_DELAY - elapsed
            time.sleep(sleep_time)

        # Predict
        raw_output, response_dict, tokens_used = predict_single(client, text)
        last_request_time = time.time()

        # Avoid noisy parse-failure logs for known sentinel placeholders.
        if raw_output in _EMPTY_SENTINEL_OUTPUTS:
            pred_label = 1
        else:
            pred_label = normalize_label(raw_output, "gemini25pro", language, idx_str)

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

        _save_cache(cache_path, cache)
        processed += 1

        if (processed) % 10 == 0:
            logger.info(
                f"  Progress: {processed + skipped}/{total} "
                f"({processed} new, {skipped} cached)"
            )

    logger.info(
        f"Gemini eval {language} complete: {processed} new, {skipped} cached"
    )

    # Capture parse failure count BEFORE saving (save resets the list)
    parse_failures = get_parse_failures()
    save_parse_failures("gemini25pro", language)

    # Compute metrics from all predictions
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

    save_results(metrics, "gemini25pro", language, seed=seed)
    return metrics

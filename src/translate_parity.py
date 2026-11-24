"""
translate_parity.py — Machine-translate the non-English test sets into English (Groq).

Purpose: reviewer request (BHASHA 2026, R1) to measure tokenizer fragmentation as
*parity* — Qwen2.5 tokens of a tweet divided by Qwen2.5 tokens of its English
translation — instead of tokens per whitespace word, which penalises Arabic for
attaching clitics to words. The translations are used only to count tokens; no
model is re-scored on them.

Input:  results/api_cache/qwen25_base/<lang>/predictions.json  (the 870 test texts)
Output: results/translations/<lang>_en.json  ({"metadata": {...}, "translations": {id: text}})

The cache is written atomically after every batch, so the run is resumable: re-run
the same command and only missing ids are sent.

Usage (from the repo root):
    python -m src.translate_parity --langs spanish hindi arabic
    python -m src.translate_parity --langs arabic --limit 20      # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from src.infer_groq import AllKeysExhausted, KeyRotator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("translate_parity")

# llama-3.3-70b-versatile (used for the paper's Llama baseline) is no longer served
# by Groq as of Nov 2026; gpt-oss-120b is the strongest multilingual model available.
DEFAULT_MODEL = "openai/gpt-oss-120b"
CACHE_IN = Path("results/api_cache/qwen25_base")
OUT_DIR = Path("results/translations")

SYSTEM_PROMPT = (
    "You are a professional translator. Translate each tweet into natural, faithful "
    "English. Preserve the meaning, tone and sentiment exactly; do not soften, explain "
    "or add anything. Keep @user mentions, URLs, hashtags and emoji unchanged. Tweets "
    "in Romanized Hindi often mix Hindi and English: translate the Hindi parts and keep "
    "the English parts as they are. Reply with a JSON object whose keys are the given "
    "ids and whose values are the English translations, and nothing else."
)

LANG_NAMES = {"spanish": "Spanish", "hindi": "Romanized Hindi (often code-mixed with English)", "arabic": "Arabic"}


def _load_texts(lang: str) -> dict[str, str]:
    data = json.loads((CACHE_IN / lang / "predictions.json").read_text(encoding="utf-8"))
    preds = data["predictions"]
    assert len(preds) == 870, f"{lang}: expected 870 test items, found {len(preds)}"
    return {k: v["text"] for k, v in preds.items()}


def _load_out(path: Path, model: str) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "metadata": {
            "model": model,
            "system_prompt": SYSTEM_PROMPT,
            "temperature": 0,
            "created": datetime.now(timezone.utc).isoformat(),
        },
        "translations": {},
    }


def _save_out(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    data["metadata"]["completed"] = len(data["translations"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(5):  # Windows can hold a lock briefly
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.2 * (2**attempt))
    tmp.replace(path)


def _translate_batch(rotator: KeyRotator, model: str, lang: str, items: dict[str, str]) -> dict[str, str]:
    """Send one batch; return {id: translation} for the ids the model answered."""
    user = (
        f"Source language: {LANG_NAMES[lang]}.\n"
        "Translate every item. Input JSON:\n"
        + json.dumps({"items": [{"id": k, "text": t} for k, t in items.items()]}, ensure_ascii=False)
    )
    for attempt in range(8):
        try:
            key, client = rotator.get_active_key()
        except AllKeysExhausted:
            wait = 65
            logger.info(f"All keys cooling down or exhausted; sleeping {wait}s")
            time.sleep(wait)
            continue
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=4096,
                response_format={"type": "json_object"},
                **({"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}),
            )
        except RateLimitError:
            rotator.mark_rate_limited(key)
            continue
        except (InternalServerError, APITimeoutError, APIConnectionError) as e:
            logger.warning(f"Groq transient error ({type(e).__name__}); retrying")
            time.sleep(2 * (attempt + 1))
            continue
        used = getattr(resp.usage, "total_tokens", 0) or 0
        rotator.record_usage(key, used)
        try:
            out = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Response was not valid JSON; retrying batch")
            continue
        if isinstance(out, dict) and "items" in out and isinstance(out["items"], list):
            out = {str(x.get("id")): x.get("translation") or x.get("text") for x in out["items"]}
        return {k: str(v).strip() for k, v in out.items() if k in items and isinstance(v, str) and v.strip()}
    raise RuntimeError("Batch failed after 8 attempts")


def run(langs: list[str], model: str, batch_size: int, limit: int | None) -> None:
    load_dotenv(".env")
    keys = [os.environ[f"GROQ_API_KEY_{i}"] for i in range(1, 5) if os.environ.get(f"GROQ_API_KEY_{i}")]
    if not keys:
        raise SystemExit("No GROQ_API_KEY_n found in .env")
    rotator = KeyRotator(keys)
    logger.info(f"Using {len(keys)} Groq key(s), model {model}")

    for lang in langs:
        texts = _load_texts(lang)
        path = OUT_DIR / f"{lang}_en.json"
        data = _load_out(path, model)
        if data["metadata"]["model"] != model:
            raise SystemExit(f"{path} was made with {data['metadata']['model']}; refusing to mix models")
        todo = [k for k in texts if k not in data["translations"]]
        if limit is not None:
            todo = todo[:limit]
        logger.info(f"{lang}: {len(data['translations'])} done, {len(todo)} to translate")

        i = 0
        while i < len(todo):
            batch_ids = todo[i : i + batch_size]
            got = _translate_batch(rotator, model, lang, {k: texts[k] for k in batch_ids})
            missing = [k for k in batch_ids if k not in got]
            for k in missing:  # retry stragglers one at a time
                one = _translate_batch(rotator, model, lang, {k: texts[k]})
                got.update(one)
            data["translations"].update(got)
            _save_out(path, data)
            still = [k for k in batch_ids if k not in got]
            if still:
                logger.warning(f"{lang}: no translation returned for ids {still}")
            i += batch_size
            logger.info(f"{lang}: {len(data['translations'])}/870")
        logger.info(f"{lang}: finished with {len(data['translations'])}/870. Usage: {rotator.get_usage_summary()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", default=["spanish", "hindi", "arabic"], choices=list(LANG_NAMES))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="translate at most N new items per language")
    a = ap.parse_args()
    run(a.langs, a.model, a.batch_size, a.limit)

"""
parity.py — Tokenizer parity against English translations (BHASHA 2026, reviewer R1).

    parity(language) = sum of Qwen2.5 tokens over the tweets
                       / sum of Qwen2.5 tokens over their English translations

Both sides are cleaned the same way first: hashtags, @mentions, URLs, punctuation and
emoji are removed, because translations copy them unchanged and they would pull every
ratio towards 1. For comparison the script also reports the paper's original measure,
tokens per whitespace word, on the same cleaned text, plus the ratio of word counts,
which shows how much of the tokens-per-word figure comes from word segmentation alone.

Validation: on the 30 hand-annotated items (results/error_analysis/annotation_sample.csv)
the ratio is computed twice, once with the human translation and once with the machine
translation, so the machine-translation-based figure can be checked against people.

Usage (repo root):  python -m src.parity
Writes:             results/translations/parity_summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

TOKENIZER = "unsloth/Qwen2.5-3B-Instruct"  # same tokenizer as the evaluated model
LANGS = ["spanish", "hindi", "arabic"]
TR_DIR = Path("results/translations")
CACHE = Path("results/api_cache/qwen25_base")
ANNOT = Path("results/error_analysis")
N_BOOT, SEED = 1000, 42

_STRIP = re.compile(r"(#\S+|@\S+|https?\S*|\bhttp\b)", flags=re.IGNORECASE)


def clean(s: str) -> str:
    s = _STRIP.sub(" ", str(s))
    s = re.sub(r"[^\w\s]", " ", s)  # punctuation and emoji; \w keeps Arabic/Devanagari letters
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def _texts(lang: str) -> dict[str, str]:
    d = json.loads((CACHE / lang / "predictions.json").read_text(encoding="utf-8"))["predictions"]
    return {k: v["text"] for k, v in d.items()}


def _boot_ratio(num: np.ndarray, den: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(num)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> None:
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False)) if s else 0
    nword = lambda s: len(s.split()) if s else 0
    rng = np.random.default_rng(SEED)
    summary: dict = {"tokenizer": TOKENIZER, "cleaning": "hashtags, @mentions, URLs, punctuation, emoji removed", "languages": {}}

    # English reference: tokens per word on the cleaned English test set (parity is 1 by definition)
    en = [clean(t) for t in _texts("english").values()]
    en_tpw = sum(map(ntok, en)) / sum(map(nword, en))
    summary["languages"]["english"] = {"tokens_per_word": round(en_tpw, 3), "parity": 1.0}

    rows = []
    for lang in LANGS:
        tr = json.loads((TR_DIR / f"{lang}_en.json").read_text(encoding="utf-8"))
        trans = tr["translations"]
        src = _texts(lang)
        recs = []
        for k, text in src.items():
            if k not in trans:
                continue
            o, e = clean(text), clean(trans[k])
            to, te = ntok(o), ntok(e)
            if to == 0 or te == 0:  # tweet was only hashtags/mentions/links
                continue
            recs.append((k, to, te, nword(o), nword(e)))
        df = pd.DataFrame(recs, columns=["id", "tok_src", "tok_en", "words_src", "words_en"])
        lo, hi = _boot_ratio(df.tok_src.values, df.tok_en.values, rng)
        res = {
            "n_items": int(len(df)),
            "n_translated": len(trans),
            "translation_model": tr["metadata"]["model"],
            "parity": round(df.tok_src.sum() / df.tok_en.sum(), 3),
            "parity_ci95": [round(lo, 3), round(hi, 3)],
            "parity_median_item": round(float((df.tok_src / df.tok_en).median()), 3),
            "tokens_per_word": round(df.tok_src.sum() / df.words_src.sum(), 3),
            "tokens_per_word_of_translation": round(df.tok_en.sum() / df.words_en.sum(), 3),
            "word_ratio_src_over_en": round(df.words_src.sum() / df.words_en.sum(), 3),
        }
        summary["languages"][lang] = res
        rows.append((lang, res))

    # Validation on the 30 hand-translated items
    ann = pd.read_csv(ANNOT / "annotation_sample.csv", encoding="utf-8-sig")
    lang_of = dict(zip(ann.text, ann.language))
    val = {}
    for lang in LANGS:
        trans = json.loads((TR_DIR / f"{lang}_en.json").read_text(encoding="utf-8"))["translations"]
        by_text = {v: k for k, v in _texts(lang).items()}
        h_src = h_en = m_en = 0
        n = 0
        for _, r in ann[ann.text.map(lang_of) == lang].iterrows():
            k = by_text.get(r.text)
            if k is None or k not in trans:
                continue
            o = ntok(clean(r.text))
            h_src += o
            h_en += ntok(clean(r.translation))
            m_en += ntok(clean(trans[k]))
            n += 1
        if n:
            val[lang] = {
                "n": n,
                "parity_human_translation": round(h_src / h_en, 3),
                "parity_machine_translation": round(h_src / m_en, 3),
            }
    summary["validation_on_annotated_items"] = val

    TR_DIR.mkdir(parents=True, exist_ok=True)
    (TR_DIR / "parity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"English tokens/word (cleaned test set): {en_tpw:.3f}")
    print(f"{'lang':8s} {'n':>4s} {'parity':>7s} {'95% CI':>15s} {'median':>7s} {'tok/word':>8s} {'words/EN':>8s}")
    for lang, r in rows:
        print(f"{lang:8s} {r['n_items']:4d} {r['parity']:7.3f} {str(r['parity_ci95']):>15s} "
              f"{r['parity_median_item']:7.3f} {r['tokens_per_word']:8.3f} {r['word_ratio_src_over_en']:8.3f}")
    print("Validation (30 annotated items):", json.dumps(val))


if __name__ == "__main__":
    main()

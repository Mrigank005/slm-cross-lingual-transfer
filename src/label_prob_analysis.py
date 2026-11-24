"""
label_prob_analysis.py — Analyse the outputs of notebooks/06_label_probs_target_prompts.ipynb.

Label probabilities (is the neutral collapse a bias, or a loss of class separation?):
  * per-class one-vs-rest AUC from the first-token label probabilities (threshold-free);
  * balanced decision rule: add one offset per class to the log-probabilities so that the
    predicted classes are balanced (1/3 each). This uses only the known class balance of the
    test sets, never item labels. Then re-score macro-F1 and neutral share.
  If AUC stays high and the balanced rule recovers much of the macro-F1, the collapse is mainly a
  bias towards neutral; if AUC is low, the model is not separating the classes.

Target-language instruction: macro-F1 and neutral share when the instruction is given in the
  language of the input (label words kept in English), compared with the English instruction.

Usage (repo root):
    python -m src.label_prob_analysis
Reads results/label_probs/ and results/target_language_prompts/ and writes summary.json in each.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

LP_DIR = Path("results/label_probs")
TP_DIR = Path("results/target_language_prompts")
LANGS = ["english", "spanish", "hindi", "arabic"]
FT = ["qwen25_finetuned_s42", "qwen25_finetuned_s123", "qwen25_finetuned_s456"]


def softmax3(logp: np.ndarray) -> np.ndarray:
    z = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def balance_offsets(logp: np.ndarray, iters: int = 20000) -> np.ndarray:
    """Per-class offsets b so that argmax(logp + b) is ~balanced (within 3 items per class).

    Uses only the known uniform class prior of the test sets, never item labels (prior matching).
    The step shrinks when the update starts oscillating, so heavily skewed models also converge.
    """
    b = np.zeros(3)
    target = len(logp) / 3
    step, prev = 0.05, None
    for _ in range(iters):
        counts = np.bincount(np.argmax(logp + b, axis=1), minlength=3)
        dev = counts - target
        if np.abs(dev).max() <= 3:
            return b
        if prev is not None and np.sign(dev).tolist() != np.sign(prev).tolist():
            step *= 0.7
        prev = dev
        b -= step * dev / target  # nudge over-used classes down
    raise RuntimeError(f"balance_offsets did not converge: counts={counts.tolist()}")


def label_prob_cell(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    y = np.array(d["true_label"])
    logp = np.array(d["logp_neg_neu_pos"])
    p = softmax3(logp)
    pred_gen = np.array(d["new_pred"])
    pred_arg = p.argmax(axis=1)
    b = balance_offsets(logp)
    pred_bal = np.argmax(logp + b, axis=1)
    auc = [roc_auc_score((y == c).astype(int), p[:, c]) for c in range(3)]
    return {
        "match_rate_vs_cache": d["match_rate_vs_cache"],
        "argmax_vs_generated_agreement": float(np.mean(pred_arg == pred_gen)),
        "top_token_is_label": float(np.mean(d["top_token_is_label"])),
        "macro_f1_generated": f1_score(y, pred_gen, average="macro"),
        "neutral_share_generated": float(np.mean(pred_gen == 1)),
        "auc_neg": auc[0], "auc_neu": auc[1], "auc_pos": auc[2], "auc_macro": float(np.mean(auc)),
        "macro_f1_balanced": f1_score(y, pred_bal, average="macro"),
        "neutral_share_balanced": float(np.mean(pred_bal == 1)),
        "offsets_neg_neu_pos": b.round(3).tolist(),
    }


def agg(cells: list[dict]) -> dict:
    out = {}
    for k in cells[0]:
        if k == "offsets_neg_neu_pos":
            continue
        v = np.array([c[k] for c in cells], dtype=float)
        out[k] = {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0}
    return out


def main() -> None:
    summary = {}
    if LP_DIR.exists():
        rows = defaultdict(dict)
        for lang in LANGS:
            base = LP_DIR / f"qwen25_base_{lang}.json"
            if base.exists():
                rows["qwen25_base"][lang] = agg([label_prob_cell(base)])
            ft = [LP_DIR / f"{m}_{lang}.json" for m in FT]
            if all(f.exists() for f in ft):
                rows["qwen25_finetuned"][lang] = agg([label_prob_cell(f) for f in ft])
        summary = dict(rows)
        (LP_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Label probabilities (mean over seeds for the fine-tuned model)")
        print(f"{'model':18s} {'lang':8s} {'match':>6s} {'F1':>6s} {'neu%':>6s} {'AUCneg':>6s} {'AUCneu':>6s} {'AUCpos':>6s} {'F1bal':>6s} {'neu%bal':>7s}")
        for m, langs in summary.items():
            for lang, r in langs.items():
                g = lambda k: r[k]["mean"]
                print(f"{m:18s} {lang:8s} {g('match_rate_vs_cache'):6.3f} {g('macro_f1_generated'):6.3f} "
                      f"{100*g('neutral_share_generated'):6.1f} {g('auc_neg'):6.3f} {g('auc_neu'):6.3f} "
                      f"{g('auc_pos'):6.3f} {g('macro_f1_balanced'):6.3f} {100*g('neutral_share_balanced'):7.1f}")

    if TP_DIR.exists():
        tp = defaultdict(dict)
        for lang in ["spanish", "hindi", "arabic"]:
            for m, files in {"qwen25_base": ["qwen25_base"], "qwen25_finetuned": FT}.items():
                paths = [TP_DIR / f"{f}_{lang}.json" for f in files]
                if not all(p.exists() for p in paths):
                    continue
                f1s, neus, f1e, neue = [], [], [], []
                for p, f in zip(paths, files):
                    d = json.loads(p.read_text(encoding="utf-8"))
                    y, pr = np.array(d["true_label"]), np.array(d["new_pred"])
                    f1s.append(f1_score(y, pr, average="macro")); neus.append(np.mean(pr == 1))
                    e = json.loads((LP_DIR / f"{f}_{lang}.json").read_text(encoding="utf-8")) if (LP_DIR / f"{f}_{lang}.json").exists() else None
                    if e:
                        ye, pe = np.array(e["true_label"]), np.array(e["new_pred"])
                        f1e.append(f1_score(ye, pe, average="macro")); neue.append(np.mean(pe == 1))
                sd = lambda v: round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0
                tp[m][lang] = {
                    "target_prompt_macro_f1": [round(float(np.mean(f1s)), 4), sd(f1s)],
                    "target_prompt_neutral_share": [round(float(np.mean(neus)), 4), sd(neus)],
                    "english_prompt_macro_f1": [round(float(np.mean(f1e)), 4), sd(f1e)] if f1e else None,
                    "english_prompt_neutral_share": [round(float(np.mean(neue)), 4), sd(neue)] if neue else None,
                }
        (TP_DIR / "summary.json").write_text(json.dumps(tp, indent=2), encoding="utf-8")
        print("\nTarget-language instruction [mean, sd]")
        print(json.dumps(tp, indent=1))


if __name__ == "__main__":
    main()

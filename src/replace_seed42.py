"""
replace_seed42.py — Write the QLoRA seed-42 predictions from a single full inference pass.

Takes the seed-42 labels produced by notebooks/06_label_probs_target_prompts.ipynb (cell 4: the released Hub
checkpoint with the same loader, prompt, batching and greedy decoding as src/infer_local.py) for all 870 test
items per language, writes them to results/api_cache/qwen25_finetuned/seed_42/, and recomputes the seed-42
metrics and the 3-seed aggregates.

Item texts, gold labels and per-item latencies are kept from the cache (latency is a T4 batch timing of the
same model and is not re-measured by notebook 06).

Usage (repo root):
    python -m src.replace_seed42 --src results/label_probs [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluate import aggregate_seed_results, compute_metrics, save_results

LANGS = ["english", "spanish", "hindi", "arabic"]
CACHE = Path("results/api_cache/qwen25_finetuned/seed_42")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/label_probs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for lang in LANGS:
        run = json.loads((Path(args.src) / f"qwen25_finetuned_s42_{lang}.json").read_text(encoding="utf-8"))
        assert run["repo"] == "Mrigank005/SLM-sentiment-crosslingual-seed-42" and run["prompt"] == "english"
        cpath = CACHE / lang / "predictions.json"
        cache = json.loads(cpath.read_text(encoding="utf-8"))
        preds = cache["predictions"]
        assert list(preds) == run["ids"], f"{lang}: item ids/order differ from the cache"

        changed = 0
        for k, new, raw, gold in zip(run["ids"], run["new_pred"], run["raw_output"], run["true_label"]):
            item = preds[k]
            assert item["true_label"] == gold, (lang, k)
            changed += item["pred_label"] != new
            item["pred_label"], item["raw_output"] = int(new), raw

        y_true = [preds[k]["true_label"] for k in run["ids"]]
        y_pred = [preds[k]["pred_label"] for k in run["ids"]]
        latency = sum(preds[k].get("latency_seconds", 0.0) for k in run["ids"])
        fails = sum(p not in (0, 1, 2) for p in y_pred)
        metrics = compute_metrics(y_true, y_pred, latency_seconds=latency,
                                  parse_failure_count=fails, total_samples=len(y_true))
        print(f"{lang:8s} labels changed: {changed:3d}   macro-F1 {metrics['f1_macro']:.4f}   "
              f"neutral share {100 * sum(p == 1 for p in y_pred) / len(y_pred):.1f}%")
        if args.dry_run:
            continue
        cache["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        cache["metadata"]["source"] = "single full pass, notebooks/06_label_probs_target_prompts.ipynb cell 4"
        tmp = cpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cpath)
        save_results(metrics, "qwen25_finetuned", lang, seed=42)

    if not args.dry_run:
        for lang in LANGS:
            aggregate_seed_results("qwen25_finetuned", lang, seeds=[42, 123, 456])
        print("seed-42 cache, seed-42 metrics and 3-seed aggregates rewritten")


if __name__ == "__main__":
    main()

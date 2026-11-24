"""
infer_xlmr.py — XLM-RoBERTa-base inference for sequence classification.

This is a classifier (encoder + linear head), NOT a generative model.
Does NOT use the prompt-based pipeline (SYSTEM_PROMPT, build_user_prompt).
Direct: tokenize -> forward pass -> argmax -> label index.

NOTE: XLM-R-base has multilingual pretraining (100 languages).
We fine-tune on English-only labels as a matched-condition encoder baseline.
Cross-lingual transfer here benefits from multilingual pretraining,
unlike the Qwen models which lack explicit multilingual training.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluate import compute_metrics, save_results, reset_parse_failures, save_parse_failures

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_xlmr_model(repo_id_or_path: str) -> tuple:
    """Load XLM-R sequence classification model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(repo_id_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id_or_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    logger.info("Loaded XLM-R model from %s", repo_id_or_path)
    return model, tokenizer


def predict_batch_xlmr(model, tokenizer, texts: list[str], batch_size: int = 32) -> tuple[list[int], float]:
    """Batched classification inference. Returns (pred_labels, latency_seconds)."""
    pred_labels: list[int] = []
    total_latency = 0.0

    device = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).tolist()
        total_latency += time.perf_counter() - start

        pred_labels.extend(int(p) for p in batch_preds)

    return pred_labels, total_latency


def run_xlmr_evaluation(
    model,
    tokenizer,
    test_dataset,
    original_indices: list[int],
    language: str,
    seed: int | None = None,
) -> dict:
    """Run full XLM-R evaluation with cache-resume support."""
    if seed is None:
        raise ValueError("seed must be provided for xlm_roberta_base evaluation")

    logger.info("Starting XLM-R evaluation: language=%s, seed=%s", language, seed)
    reset_parse_failures()

    texts = test_dataset["text"]
    y_true_all = list(test_dataset["label"])

    pred_path = (
        Path("results/api_cache")
        / "xlm_roberta_base"
        / f"seed_{seed}"
        / language
        / "predictions.json"
    )

    predictions_data = {
        "metadata": {
            "model": "xlm_roberta_base",
            "language": language,
            "seed": seed,
            "total_samples": len(test_dataset),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "predictions": {},
    }

    if pred_path.exists():
        with open(pred_path, "r", encoding="utf-8") as f:
            predictions_data = json.load(f)
        predictions_data.setdefault("metadata", {})
        predictions_data.setdefault("predictions", {})

    completed_keys = set(predictions_data["predictions"].keys())

    pending_texts = []
    pending_true = []
    pending_indices = []
    skipped = 0
    for orig_idx, text, true_label in zip(original_indices, texts, y_true_all):
        idx_str = str(orig_idx)
        if idx_str in completed_keys:
            skipped += 1
            continue
        pending_indices.append(orig_idx)
        pending_texts.append(text)
        pending_true.append(int(true_label))

    total_latency_new = 0.0
    pred_labels_new: list[int] = []
    if pending_texts:
        pred_labels_new, total_latency_new = predict_batch_xlmr(
            model,
            tokenizer,
            pending_texts,
            batch_size=32,
        )

    per_sample_latency = (
        total_latency_new / len(pending_texts) if len(pending_texts) > 0 else 0.0
    )

    for orig_idx, true_label, pred_label, text in zip(
        pending_indices,
        pending_true,
        pred_labels_new,
        pending_texts,
    ):
        predictions_data["predictions"][str(orig_idx)] = {
            "text": text,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "raw_output": LABEL_MAP.get(int(pred_label), str(pred_label)),
            "tokens_used": 0,
            "latency_seconds": round(per_sample_latency, 4),
        }

    predictions_data["metadata"] = {
        "model": "xlm_roberta_base",
        "language": language,
        "seed": seed,
        "total_samples": len(test_dataset),
        "completed": len(predictions_data["predictions"]),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pred_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    os.replace(str(tmp_path), str(pred_path))

    logger.info(
        "XLM-R predictions saved to %s (%s new, %s cached)",
        pred_path,
        len(pending_texts),
        skipped,
    )

    y_true = []
    y_pred = []
    total_latency = 0.0
    for idx_str in sorted(predictions_data["predictions"].keys(), key=int):
        entry = predictions_data["predictions"][idx_str]
        y_true.append(int(entry["true_label"]))
        y_pred.append(int(entry["pred_label"]))
        total_latency += float(entry.get("latency_seconds", 0.0))

    metrics = compute_metrics(
        y_true,
        y_pred,
        latency_seconds=total_latency,
        parse_failure_count=0,
        total_samples=len(y_true),
    )

    save_parse_failures("xlm_roberta_base", language, seed=seed)
    save_results(metrics, "xlm_roberta_base", language, seed=seed)
    return metrics

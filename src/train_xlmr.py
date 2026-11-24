"""
train_xlmr.py — Fine-tuning XLM-RoBERTa-base for 3-class sentiment classification.

XLM-R's cross-lingual transfer benefits from its multilingual pretraining (100 languages), unlike the Qwen models which have no explicit multilingual training objective. We fine-tune on English-only labels as a matched-condition encoder baseline.
"""

import logging
import inspect
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_MAP_INV = {"negative": 0, "neutral": 1, "positive": 2}


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    """Tokenize dataset examples for sequence classification."""

    def _preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    keep_cols = {"text", "label"}
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]

    tokenized = dataset.map(_preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized


def _compute_warmup_steps(
    num_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    warmup_ratio: float,
) -> int:
    """Convert warmup ratio into deterministic warmup steps."""
    effective_batch = max(1, per_device_train_batch_size * gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(num_examples / effective_batch))
    total_steps = max(1, steps_per_epoch * num_train_epochs)
    return max(0, int(total_steps * warmup_ratio))


def train_xlmr(config: dict, train_dataset, val_dataset, seed: int = 42):
    """Fine-tune XLM-R-base on English sentiment data.

    Uses full-parameter fine-tuning via standard HuggingFace Trainer.
    """
    xlmr_cfg = config["xlmr"]

    model_name = xlmr_cfg["model_name"]
    max_seq_length = int(xlmr_cfg["max_seq_length"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(xlmr_cfg["num_labels"]),
        id2label=LABEL_MAP,
        label2id=LABEL_MAP_INV,
    )

    train_tok = _tokenize_dataset(train_dataset, tokenizer, max_seq_length)
    val_tok = _tokenize_dataset(val_dataset, tokenizer, max_seq_length)

    per_device_train_batch_size = int(xlmr_cfg["per_device_train_batch_size"])
    gradient_accumulation_steps = int(xlmr_cfg["gradient_accumulation_steps"])
    num_train_epochs = int(xlmr_cfg["num_train_epochs"])

    if "warmup_steps" in xlmr_cfg and xlmr_cfg["warmup_steps"] is not None:
        warmup_steps = int(xlmr_cfg["warmup_steps"])
    else:
        warmup_ratio = float(xlmr_cfg.get("warmup_ratio", 0.0))
        warmup_steps = _compute_warmup_steps(
            num_examples=len(train_tok),
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
        )
        logger.info(
            "Converted warmup_ratio=%s to warmup_steps=%s",
            warmup_ratio,
            warmup_steps,
        )

    drive_template = config.get("checkpointing", {}).get("drive_path_template")
    base_drive_path = config.get("checkpointing", {}).get("drive_path", "./checkpoints")
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if drive_template:
        output_dir = Path(drive_template.format(seed=seed)) / "xlm_roberta_base" / f"run_{run_stamp}"
    else:
        output_dir = Path(base_drive_path) / f"seed_{seed}" / "xlm_roberta_base" / f"run_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using checkpoint directory: %s", output_dir)

    load_best_model_at_end = bool(xlmr_cfg["load_best_model_at_end"])

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(xlmr_cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        weight_decay=float(xlmr_cfg["weight_decay"]),
        eval_strategy=xlmr_cfg["evaluation_strategy"],
        save_strategy=xlmr_cfg["save_strategy"],
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=xlmr_cfg["metric_for_best_model"],
        seed=seed,
        report_to=config.get("training", {}).get("report_to", "wandb"),
    )

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_tok,
        "eval_dataset": val_tok,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": _compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=int(xlmr_cfg["early_stopping_patience"]),
            )
        ],
    }

    # Transformers/Trainer signatures vary by version; try both argument names safely.
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    try:
        trainer = Trainer(**trainer_kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument 'tokenizer'" in msg:
            trainer_kwargs.pop("tokenizer", None)
            trainer = Trainer(**trainer_kwargs)
        elif "unexpected keyword argument 'processing_class'" in msg:
            trainer_kwargs.pop("processing_class", None)
            trainer_kwargs["tokenizer"] = tokenizer
            trainer = Trainer(**trainer_kwargs)
        else:
            raise

    logger.info(
        "Starting XLM-R fine-tuning: seed=%s, train=%s, val=%s",
        seed,
        len(train_tok),
        len(val_tok),
    )
    trainer.train()

    logger.info(
        "XLM-R training complete: best_checkpoint=%s, best_metric=%s",
        trainer.state.best_model_checkpoint,
        trainer.state.best_metric,
    )

    return model, tokenizer, trainer

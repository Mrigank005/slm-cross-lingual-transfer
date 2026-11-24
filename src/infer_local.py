"""
infer_local.py — Local inference for Qwen2.5 3B (base and fine-tuned).

Handles batched inference with latency tracking for both zero-shot baseline
and QLoRA fine-tuned model evaluation.
"""

import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Suppress harmless warnings from transformers internals
warnings.filterwarnings('ignore', message='.*Unsloth should be imported.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.modeling_attn_mask_utils')

import torch
from transformers import AutoTokenizer

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


# --------------------------------------------------------------------------- #
#  Model Loading                                                                #
# --------------------------------------------------------------------------- #

def load_base_model(
    model_name: str = "unsloth/Qwen2.5-3B-Instruct",
    load_in_4bit: bool = True,
) -> tuple:
    """Load base Qwen2.5-3B-Instruct in 4-bit for zero-shot evaluation.

    Args:
        model_name: HuggingFace model ID.
        load_in_4bit: Whether to load in 4-bit quantization.

    Returns:
        (model, tokenizer)
    """
    try:
        # Try unsloth first (faster on supported hardware)
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            load_in_4bit=load_in_4bit,
            dtype=None,  # auto-detect
        )
        FastLanguageModel.for_inference(model)
        model.generation_config.max_length = None  # avoid max_new_tokens conflict
        logger.info(f"Loaded base model via Unsloth: {model_name}")

    except ImportError:
        # Fallback to standard transformers + bitsandbytes
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()
        logger.info(f"Loaded base model via transformers: {model_name}")

    return model, tokenizer


def load_finetuned_model(
    repo_id_or_path: str,
    load_in_4bit: bool = True,
) -> tuple:
    """Load fine-tuned model from HuggingFace Hub or local checkpoint.

    Args:
        repo_id_or_path: HF repo ID or local path to merged model/adapter.
        load_in_4bit: Whether to load in 4-bit.

    Returns:
        (model, tokenizer)
    """
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=repo_id_or_path,
            max_seq_length=512,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
        model.generation_config.max_length = None  # avoid max_new_tokens conflict
        logger.info(f"Loaded fine-tuned model via Unsloth: {repo_id_or_path}")

    except ImportError:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(repo_id_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id_or_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()
        logger.info(f"Loaded fine-tuned model via transformers: {repo_id_or_path}")

    return model, tokenizer


# --------------------------------------------------------------------------- #
#  Batched Inference                                                            #
# --------------------------------------------------------------------------- #

def predict_batch(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
) -> tuple[list[str], float]:
    """Run batched inference with latency tracking.

    Uses tokenizer.apply_chat_template() for proper ChatML formatting.
    Times ONLY generation (excludes model loading).

    Args:
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        texts: List of tweet texts to classify.
        batch_size: Inference batch size.

    Returns:
        (raw_outputs, total_latency_seconds):
            raw_outputs: List of raw model output strings.
            total_latency_seconds: Total generation time.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_outputs = []
    total_latency = 0.0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Format each text into ChatML messages
        batch_inputs = []
        for text in batch_texts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(text)},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_inputs.append(formatted)

        # Tokenize batch
        encodings = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        input_lengths = [len(encodings.input_ids[j]) for j in range(len(batch_texts))]

        # Generate — time this part only
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start
        total_latency += elapsed

        # Decode only the generated tokens (exclude input)
        for j, output in enumerate(outputs):
            generated_tokens = output[input_lengths[j] :]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            raw_outputs.append(decoded.strip())

        logger.debug(
            f"Batch {i // batch_size + 1}: "
            f"{len(batch_texts)} samples, {elapsed:.2f}s"
        )

    return raw_outputs, total_latency


# --------------------------------------------------------------------------- #
#  Full Evaluation Pipeline                                                     #
# --------------------------------------------------------------------------- #

def run_local_evaluation(
    model,
    tokenizer,
    test_dataset,
    original_indices: list[int],
    model_name: str,
    language: str,
    seed: int | None = None,
) -> dict:
    """Full local evaluation pipeline.

    Args:
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        test_dataset: 200-sample subsampled dataset.
        original_indices: Indices from subsample_indices.json.
        model_name: "qwen25_base" or "qwen25_finetuned".
        language: Language code.

    Returns:
        metrics_dict from compute_metrics().
    """
    logger.info(f"Starting local evaluation: {model_name} / {language}")
    reset_parse_failures()

    texts = test_dataset["text"]
    y_true_all = list(test_dataset["label"])

    if model_name == "qwen25_finetuned" and seed is None:
        raise ValueError(
            "seed parameter is required for qwen25_finetuned — "
            "cache paths are seed-namespaced. Pass seed=42, 123, or 456."
        )

    if model_name == "qwen25_finetuned" and seed is not None:
        pred_path = (
            Path("results/api_cache")
            / model_name
            / f"seed_{seed}"
            / language
            / "predictions.json"
        )
    else:
        pred_path = Path("results/api_cache") / model_name / language / "predictions.json"

    predictions_data = {
        "metadata": {
            "model": model_name,
            "language": language,
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
    raw_outputs_new: list[str] = []
    if pending_texts:
        raw_outputs_new, total_latency_new = predict_batch(model, tokenizer, pending_texts)

    per_sample_latency = (
        total_latency_new / len(pending_texts) if len(pending_texts) > 0 else 0.0
    )

    for orig_idx, true_label, raw, text in zip(
        pending_indices,
        pending_true,
        raw_outputs_new,
        pending_texts,
    ):
        pred_label = normalize_label(raw, model_name, language, str(orig_idx))
        predictions_data["predictions"][str(orig_idx)] = {
            "text": text,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "raw_output": raw,
            "tokens_used": 0,
            "latency_seconds": round(per_sample_latency, 4),
        }

    # Save parse failures (to eval_scores/, not api_cache/)
    parse_failures = get_parse_failures()
    save_parse_failures(model_name, language, seed=seed)

    predictions_data["metadata"] = {
        "model": model_name,
        "language": language,
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
        f"Predictions saved to {pred_path} "
        f"({len(pending_texts)} new, {skipped} cached)"
    )

    # Compute metrics from all cached + newly inferred predictions
    y_true = []
    y_pred = []
    total_latency = 0.0
    for idx_str in sorted(predictions_data["predictions"].keys(), key=int):
        entry = predictions_data["predictions"][idx_str]
        y_true.append(int(entry["true_label"]))
        y_pred.append(int(entry["pred_label"]))
        total_latency += float(entry.get("latency_seconds", 0.0))

    # Compute metrics
    metrics = compute_metrics(
        y_true,
        y_pred,
        latency_seconds=total_latency,
        parse_failure_count=len(parse_failures),
        total_samples=len(y_true),
    )

    # Save metrics
    save_results(metrics, model_name, language, seed=seed)

    return metrics

"""
evaluate.py — Metrics computation, label normalization, bootstrap CIs,
confusion matrices, majority-class baseline, and parse failure tracking.

All evaluation functions for the 5×4 cross-lingual sentiment comparison.
"""

import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Label Normalization with Parse Failure Tracking                              #
# --------------------------------------------------------------------------- #

_parse_failures: list[dict] = []

LABEL_ALIASES: dict[str, int] = {
    "positive": 2, "pos": 2, "2": 2,
    "neutral": 1, "neu": 1, "1": 1,
    "negative": 0, "neg": 0, "0": 0,
}

# Models whose parse failures go to eval_scores/ (not api_cache/)
_LOCAL_MODELS = {"qwen25_base", "qwen25_finetuned", "majority_baseline", "xlm_roberta_base"}

# Mapping from short model names to API cache directory names
_API_MODEL_DIR_MAP = {
    "llama70b": "llama-3.3-70b-versatile",
    "gemini25pro": "gemini-2.5-pro",
}


def normalize_label(
    raw_output: str,
    model_name: str = "",
    language: str = "",
    sample_id: str = "",
) -> int:
    """Parse raw LLM output into integer label {0, 1, 2}.

    Steps:
        1. Strip whitespace, lowercase, take first line only.
        2. Check exact match against LABEL_ALIASES keys.
        3. If no exact match, check if any alias key is a substring.
           (handles "The sentiment is positive" → "positive" → 2)
        4. If still no match, log failure and fall back to 1 (neutral).

    Parse failures are appended to the module-level _parse_failures list.

    Returns:
        int: 0 (negative), 1 (neutral), or 2 (positive).
    """
    cleaned = raw_output.strip().lower().split("\n")[0].strip()

    # Step 2: exact match
    if cleaned in LABEL_ALIASES:
        return LABEL_ALIASES[cleaned]

    # Step 3: substring match (first match wins, ordered by priority)
    for alias, label_int in LABEL_ALIASES.items():
        if alias in cleaned:
            return label_int

    # Step 4: parse failure → fallback to neutral
    failure = {
        "raw_output": raw_output,
        "cleaned": cleaned,
        "fallback_to": "neutral",
        "model": model_name,
        "language": language,
        "sample_id": sample_id,
    }
    _parse_failures.append(failure)
    logger.warning(
        f"Parse failure: '{raw_output}' → fallback to neutral "
        f"(model={model_name}, lang={language}, id={sample_id})"
    )
    return 1


def get_parse_failures() -> list[dict]:
    """Return current parse failures list."""
    return _parse_failures.copy()


def reset_parse_failures() -> None:
    """Clear parse failures list. Call before each model×language evaluation."""
    global _parse_failures
    _parse_failures = []


def save_parse_failures(
    model_name: str,
    language: str,
    output_dir: str | None = None,
    seed: int | None = None,
) -> None:
    """Save parse failures to disk. Creates directories if needed.

    Path logic:
        - API models (llama70b, gemini25pro):
          results/api_cache/{model_api_name}/{language}/parse_failures.json
        - Local models (qwen25_base, qwen25_finetuned, majority_baseline):
          results/eval_scores/{model_name}_{language}_parse_failures.json

    This separation keeps api_cache/ strictly for API model responses.
    Resets the global _parse_failures list after saving.
    """
    failures = get_parse_failures()

    if model_name in _LOCAL_MODELS:
        base_dir = Path(output_dir or "results/eval_scores")
        if seed is not None:
            base_dir = base_dir / f"seed_{seed}"
        path = base_dir / f"{model_name}_{language}_parse_failures.json"
    else:
        base_dir = output_dir or "results/api_cache"
        api_dir_name = _API_MODEL_DIR_MAP.get(model_name, model_name)
        path = Path(base_dir) / api_dir_name / language / "parse_failures.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model": model_name,
        "language": language,
        "total_failures": len(failures),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
    }

    _atomic_write_json(data, path)

    logger.info(f"Saved {len(failures)} parse failures to {path}")
    reset_parse_failures()


# --------------------------------------------------------------------------- #
#  Bootstrap Confidence Intervals                                               #
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    y_true: list[int],
    y_pred: list[int],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for F1 Macro.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        n_resamples: Number of bootstrap iterations. Default 1000.
        ci: Confidence level. Default 0.95 → 2.5th and 97.5th percentiles.
        seed: Seed for numpy.random.Generator.

    Returns:
        (lower_bound, upper_bound): Rounded to 4 decimal places.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    f1_scores = []

    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        try:
            f1 = f1_score(y_true[indices], y_pred[indices], average="macro", zero_division=0)
            f1_scores.append(f1)
        except Exception:
            continue

    alpha = 1 - ci
    lower = round(float(np.percentile(f1_scores, 100 * alpha / 2)), 4)
    upper = round(float(np.percentile(f1_scores, 100 * (1 - alpha / 2))), 4)
    return lower, upper


# --------------------------------------------------------------------------- #
#  Metrics Computation                                                          #
# --------------------------------------------------------------------------- #

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
    latency_seconds: float | None = None,
    parse_failure_count: int = 0,
    total_samples: int = 0,
) -> dict:
    """Compute full evaluation metrics suite.

    Returns dict with: accuracy, f1_macro, f1_macro_ci_lower/upper,
    f1_weighted, per_class P/R/F1, latency, parse failures.

    Raises logging.warning if parse_failure_rate > 0.05 (5%).
    """
    if label_names is None:
        label_names = ["negative", "neutral", "positive"]

    if total_samples == 0:
        total_samples = len(y_true)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    # Bootstrap CI for F1 Macro
    ci_lower, ci_upper = bootstrap_ci(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1_per, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=[0, 1, 2], zero_division=0
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(support[i]),
        }

    # Parse failure rate
    parse_failure_rate = parse_failure_count / total_samples if total_samples > 0 else 0
    if parse_failure_rate > 0.05:
        logger.warning(
            f"Parse failure rate {parse_failure_rate:.1%} exceeds 5% threshold! "
            f"({parse_failure_count}/{total_samples} samples)"
        )

    # Latency
    avg_latency: float | None = None
    if latency_seconds is not None and total_samples > 0:
        avg_latency = round(float(latency_seconds / total_samples), 4)

    return {
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_macro_ci_lower": ci_lower,
        "f1_macro_ci_upper": ci_upper,
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": per_class,
        "avg_latency_seconds": avg_latency,
        "total_latency_seconds": round(float(latency_seconds), 4) if latency_seconds is not None else None,
        "parse_failure_count": parse_failure_count,
        "parse_failure_rate": round(float(parse_failure_rate), 4),
        "parse_failure_flagged": parse_failure_rate > 0.05,
        "n_samples": len(y_true),
    }


# --------------------------------------------------------------------------- #
#  Confusion Matrix                                                             #
# --------------------------------------------------------------------------- #

def compute_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
    save_path: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (7, 6),
    dpi: int = 300,
) -> np.ndarray:
    """Plot and optionally save confusion matrix as high-res figure.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Class names for axes. Default: negative/neutral/positive.
        save_path: If given, save figure to this path (PDF or PNG).
        title: Figure title.
        figsize: Figure size in inches.
        dpi: Resolution for saved figure.

    Returns:
        Confusion matrix as numpy array.
    """
    if label_names is None:
        label_names = ["negative", "neutral", "positive"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close(fig)
    return cm


# --------------------------------------------------------------------------- #
#  Majority-Class Baseline                                                      #
# --------------------------------------------------------------------------- #

def majority_class_baseline(
    y_true: list[int],
    label_names: list[str] | None = None,
) -> tuple[list[int], dict]:
    """Generate majority-class predictions and compute metrics.

    Finds the most frequent label in y_true, predicts that label for all samples.

    Args:
        y_true: Ground truth labels.
        label_names: Label names for reporting.

    Returns:
        (y_pred, metrics_dict): y_pred is the majority class repeated.
        metrics_dict is from compute_metrics() with latency=None.
    """
    if label_names is None:
        label_names = ["negative", "neutral", "positive"]

    counter: Counter[int] = Counter(y_true)
    majority_class: int = counter.most_common(1)[0][0]
    y_pred: list[int] = [majority_class] * len(y_true)

    logger.info(
        f"Majority baseline: class={label_names[majority_class]} "
        f"(count={counter[majority_class]}/{len(y_true)})"
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
        label_names=label_names,
        latency_seconds=None,
        parse_failure_count=0,
        total_samples=len(y_true),
    )
    return y_pred, metrics


# --------------------------------------------------------------------------- #
#  Results Persistence                                                          #
# --------------------------------------------------------------------------- #

VALID_MODEL_NAMES = {
    "qwen25_base", "qwen25_finetuned", "llama70b", "gemini25pro", "majority_baseline",
    "xlm_roberta_base",
}


def _atomic_write_json(data: dict, final_path: Path) -> None:
    """Write JSON atomically using temp file + rename."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp_path), str(final_path))  # atomic on POSIX and Windows


def save_results(
    metrics_dict: dict,
    model_name: str,
    language: str,
    output_dir: str = "results/eval_scores/",
    seed: int | None = None,
) -> str:
    """Save metrics to results/eval_scores/{model_name}_{language}.json.

    Args:
        metrics_dict: Output of compute_metrics().
        model_name: One of VALID_MODEL_NAMES.
        language: Language code.
        output_dir: Base output directory.

    Returns:
        Path to saved file.
    """
    if model_name not in VALID_MODEL_NAMES:
        logger.warning(
            f"Unexpected model_name '{model_name}'. "
            f"Expected one of: {VALID_MODEL_NAMES}"
        )

    base_path = Path(output_dir)
    if seed is not None:
        base_path = base_path / f"seed_{seed}"

    path = base_path / f"{model_name}_{language}.json"

    data = {
        "metadata": {
            "model": model_name,
            "language": language,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        **metrics_dict,
    }

    _atomic_write_json(data, path)

    logger.info(f"Results saved to {path}")
    return str(path)


def aggregate_seed_results(
    model_name: str,
    language: str,
    seeds: list[int],
    results_dir: str = "results/eval_scores/",
    cache_dir: str = "results/api_cache/",
) -> str:
    """Aggregate per-seed metrics into mean/std summary JSON.

    Reads:
        {results_dir}/seed_{seed}/{model}_{language}.json for each seed.
    Writes:
        {results_dir}/aggregated/{model}_{language}.json
    """
    if not seeds:
        raise ValueError("seeds must be a non-empty list")

    per_seed: dict[int, dict] = {}
    for seed in seeds:
        seed_path = Path(results_dir) / f"seed_{seed}" / f"{model_name}_{language}.json"
        if not seed_path.exists():
            raise FileNotFoundError(f"Missing per-seed result file: {seed_path}")
        with open(seed_path, "r", encoding="utf-8") as f:
            per_seed[seed] = json.load(f)

    n_samples_values = {seed: per_seed[seed]["n_samples"] for seed in seeds}
    if len(set(n_samples_values.values())) != 1:
        raise ValueError(
            f"n_samples mismatch across seeds — aggregation aborted. "
            f"Ensure all seeds are evaluated at the same sample count before aggregating. "
            f"Found: {n_samples_values}"
        )

    expected_n = 870
    actual_n = list(n_samples_values.values())[0]
    if actual_n != expected_n:
        raise ValueError(
            f"Expected n_samples={expected_n} but found {actual_n}. "
            f"Ensure n=870 evaluation is complete for all seeds before aggregating."
        )

    f1_scores_per_seed = {s: float(per_seed[s]["f1_macro"]) for s in seeds}
    f1_values = [f1_scores_per_seed[s] for s in seeds]
    acc_values = [float(per_seed[s]["accuracy"]) for s in seeds]

    seed_dependent_models = {"qwen25_finetuned", "xlm_roberta_base"}
    if model_name in seed_dependent_models:
        # Bootstrap CI on the mean of per-seed F1 values (cross-seed variability).
        rng = np.random.default_rng(42)
        n_bootstrap = 1000
        ci = 0.95
        alpha = 1 - ci
        per_seed_f1s = np.array([f1_scores_per_seed[s] for s in seeds], dtype=float)
        bootstrap_means = np.array([
            np.mean(rng.choice(per_seed_f1s, size=len(per_seed_f1s), replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_lower = round(float(np.percentile(bootstrap_means, 100 * alpha / 2)), 4)
        ci_upper = round(float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2))), 4)
    else:
        # Seed-agnostic models should report sample-level bootstrap CI from one
        # run, not a degenerate across-seed CI of replicated metrics.
        ref_seed = seeds[0]
        ref_metrics = per_seed[ref_seed]
        ci_lower = ref_metrics.get("f1_macro_ci_lower")
        ci_upper = ref_metrics.get("f1_macro_ci_upper")

        if ci_lower is None or ci_upper is None or float(ci_lower) == float(ci_upper):
            y_true, y_pred = load_predictions_for_model(
                model_name,
                language,
                results_dir=results_dir,
                cache_dir=cache_dir,
                seed=None,
            )
            ci_lower, ci_upper = bootstrap_ci(y_true, y_pred)
        else:
            ci_lower = round(float(ci_lower), 4)
            ci_upper = round(float(ci_upper), 4)

    per_class = {}
    label_names = ["negative", "neutral", "positive"]
    for label_name in label_names:
        f1_label_values = [float(per_seed[s]["per_class"][label_name]["f1"]) for s in seeds]
        per_class[label_name] = {
            "f1_mean": round(float(np.mean(f1_label_values)), 4),
            "f1_std": round(float(np.std(f1_label_values, ddof=1 if len(f1_label_values) > 1 else 0)), 4),
        }

    first_seed = seeds[0]
    n_samples = int(per_seed[first_seed].get("n_samples", 0))
    latency_values = [
        float(per_seed[s]["avg_latency_seconds"])
        for s in seeds
        if per_seed[s].get("avg_latency_seconds") is not None
    ]

    aggregated = {
        "metadata": {
            "model": model_name,
            "language": language,
            "seeds": seeds,
            "n_samples": n_samples,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "f1_macro_mean": round(float(np.mean(f1_values)), 4),
        "f1_macro_std": round(float(np.std(f1_values, ddof=1 if len(f1_values) > 1 else 0)), 4),
        "f1_macro_per_seed": {str(seed): round(float(per_seed[seed]["f1_macro"]), 4) for seed in seeds},
        "f1_macro_ci_lower": ci_lower,
        "f1_macro_ci_upper": ci_upper,
        "accuracy_mean": round(float(np.mean(acc_values)), 4),
        "accuracy_std": round(float(np.std(acc_values, ddof=1 if len(acc_values) > 1 else 0)), 4),
        "avg_latency_seconds_mean": (
            round(float(np.mean(latency_values)), 4) if latency_values else None
        ),
        "avg_latency_seconds_std": (
            round(float(np.std(latency_values, ddof=1 if len(latency_values) > 1 else 0)), 4) if latency_values else None
        ),
        "per_class": per_class,
    }

    out_path = Path(results_dir) / "aggregated" / f"{model_name}_{language}.json"
    _atomic_write_json(aggregated, out_path)
    logger.info(f"Aggregated seed results saved to {out_path}")
    return str(out_path)


# --------------------------------------------------------------------------- #
#  Prediction Loading (for confusion matrices / post-hoc analysis)              #
# --------------------------------------------------------------------------- #

def load_predictions_for_model(
    model_name: str,
    language: str,
    results_dir: str = "results/eval_scores/",
    cache_dir: str = "results/api_cache/",
    seed: int | None = None,
) -> tuple[list[int], list[int]]:
    """Load y_true and y_pred for a given model×language from saved results.

    For API models (llama70b, gemini25pro):
        Load from cache_dir/{api_name}/{language}/predictions.json.
    For majority_baseline:
        Load y_true from subsample, generate y_pred as majority class.
    For local models (qwen25_base, qwen25_finetuned):
        Load from cache_dir/{model_name}/{language}/predictions.json.

    Returns:
        (y_true, y_pred): Lists of integer labels.
    """
    if model_name == "majority_baseline":
        # Reconstruct from subsample
        from src.dataset import load_subsample_indices, load_all_splits, get_stratified_subsample

        indices_data = load_subsample_indices()
        full_data = load_all_splits([language])
        test_ds = get_stratified_subsample(
            full_data[language]["test"],
            indices_override=indices_data["indices"][language],
        )
        y_true: list[int] = [int(x) for x in test_ds["label"]]
        counter: Counter[int] = Counter(y_true)
        majority: int = counter.most_common(1)[0][0]
        y_pred: list[int] = [majority] * len(y_true)
        return y_true, y_pred

    # For API models
    if model_name in _API_MODEL_DIR_MAP:
        api_name = _API_MODEL_DIR_MAP[model_name]
        cache_path = Path(cache_dir) / api_name / language / "predictions.json"
    elif model_name == "xlm_roberta_base":
        if seed is None:
            raise ValueError("seed must be provided for xlm_roberta_base predictions")
        cache_path = Path(cache_dir) / "xlm_roberta_base" / f"seed_{seed}" / language / "predictions.json"
    elif model_name == "qwen25_finetuned":
        if seed is None:
            raise ValueError(
                "seed parameter is required for qwen25_finetuned in load_predictions_for_model(). "
                "Pass seed=42, 123, or 456."
            )
        cache_path = Path(cache_dir) / model_name / f"seed_{seed}" / language / "predictions.json"
    else:
        # Local models — predictions stored in api_cache/
        cache_path = Path(cache_dir) / model_name / language / "predictions.json"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {cache_path}. "
            f"Run evaluation for {model_name}/{language} first."
        )

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", {})
    y_true: list[int] = []
    y_pred: list[int] = []
    for idx in sorted(predictions.keys(), key=int):
        entry = predictions[idx]
        y_true.append(int(entry["true_label"]))
        y_pred.append(int(entry["pred_label"]))

    return y_true, y_pred

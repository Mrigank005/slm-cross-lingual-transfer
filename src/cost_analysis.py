"""
cost_analysis.py — Cost-performance comparison across models.

Computes inference costs from actual token usage and published pricing,
generates cost-performance table and scatter plot.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Published pricing (per 1M tokens)
PRICING = {
    "llama-3.3-70b-versatile": {
        "input_per_1m": 0.59,
        "output_per_1m": 0.79,
        "provider": "Groq",
    },
    "gemini-2.5-pro": {
        "input_per_1m": 1.25,   # Check current Google pricing at publication time
        "output_per_1m": 10.00,
        "provider": "Google AI Studio",
    },
}

MODEL_PARAMS = {
    "majority_baseline": {"params": "N/A", "params_numeric": 0},
    "qwen25_base": {"params": "3B", "params_numeric": 3e9},
    "qwen25_finetuned": {"params": "3B", "params_numeric": 3e9},
    "llama70b": {"params": "70B", "params_numeric": 70e9},
    "gemini25pro": {"params": "Undisclosed", "params_numeric": 200e9},  # estimated for plot sizing
    "xlm_roberta_base": {"params": "125M", "params_numeric": 125e6},
}


def _count_tokens_from_cache(
    model_api_name: str,
    languages: list[str],
    cache_dir: str = "results/api_cache/",
) -> dict:
    """Count total input/output tokens from cached predictions.

    Returns:
        {"total_tokens": int, "total_samples": int, "per_language": {...}}
    """
    total_tokens = 0
    total_samples = 0
    per_lang = {}

    for lang in languages:
        cache_path = Path(cache_dir) / model_api_name / lang / "predictions.json"
        if not cache_path.exists():
            per_lang[lang] = {"tokens": 0, "samples": 0}
            continue

        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        predictions = data.get("predictions", {})
        lang_tokens = sum(p.get("tokens_used", 0) for p in predictions.values())
        lang_samples = len(predictions)

        per_lang[lang] = {"tokens": lang_tokens, "samples": lang_samples}
        total_tokens += lang_tokens
        total_samples += lang_samples

    return {
        "total_tokens": total_tokens,
        "total_samples": total_samples,
        "per_language": per_lang,
    }


def compute_cost_table(
    results_dir: str = "results/eval_scores/aggregated/",
    cache_dir: str = "results/api_cache/",
    languages: list[str] | None = None,
) -> pd.DataFrame:
    """Build cost-performance comparison table.

    Columns: model, params, avg_f1_macro, cost_actual, cost_per_1k,
             training_cost, total_cost, provider.

    Args:
        results_dir: Directory with eval_scores JSONs.
        cache_dir: Directory with API cache JSONs.
        languages: List of languages. Default: all 4.

    Returns:
        DataFrame with one row per model.
    """
    if languages is None:
        languages = ["english", "hindi", "arabic", "spanish"]

    models = [
        "majority_baseline",
        "qwen25_base",
        "qwen25_finetuned",
        "llama70b",
        "gemini25pro",
        "xlm_roberta_base",
    ]
    rows = []

    for model in models:
        # Load F1 scores across languages
        f1_scores = []
        for lang in languages:
            path = Path(results_dir) / f"{model}_{lang}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                f1 = data.get("f1_macro_mean", data.get("f1_macro"))
                if f1 is not None:
                    f1_scores.append(f1)

        avg_f1 = round(np.mean(f1_scores), 4) if f1_scores else None

        # Cost calculation
        if model in ("majority_baseline", "qwen25_base", "qwen25_finetuned", "xlm_roberta_base"):
            cost_actual = 0.0
            cost_per_1k = 0.0
            provider = "Self-hosted"
            training_cost = 0.0  # Colab free tier
        elif model == "llama70b":
            token_data = _count_tokens_from_cache(
                "llama-3.3-70b-versatile", languages, cache_dir
            )
            pricing = PRICING["llama-3.3-70b-versatile"]
            # Approximate split: 80% input, 20% output tokens
            input_tokens = token_data["total_tokens"] * 0.8
            output_tokens = token_data["total_tokens"] * 0.2
            cost_actual = (
                input_tokens / 1e6 * pricing["input_per_1m"]
                + output_tokens / 1e6 * pricing["output_per_1m"]
            )
            n_samples = max(token_data["total_samples"], 1)
            cost_per_1k = round(cost_actual / n_samples * 1000, 4)
            cost_actual = round(cost_actual, 4)
            provider = pricing["provider"]
            training_cost = 0.0  # Zero-shot
        elif model == "gemini25pro":
            token_data = _count_tokens_from_cache(
                "gemini-2.5-pro", languages, cache_dir
            )
            pricing = PRICING["gemini-2.5-pro"]
            input_tokens = token_data["total_tokens"] * 0.8
            output_tokens = token_data["total_tokens"] * 0.2
            cost_actual = (
                input_tokens / 1e6 * pricing["input_per_1m"]
                + output_tokens / 1e6 * pricing["output_per_1m"]
            )
            n_samples = max(token_data["total_samples"], 1)
            cost_per_1k = round(cost_actual / n_samples * 1000, 4)
            cost_actual = round(cost_actual, 4)
            provider = pricing["provider"]
            training_cost = 0.0
        else:
            cost_actual = None
            cost_per_1k = None
            provider = "Unknown"
            training_cost = 0.0

        rows.append({
            "model": model,
            "params": MODEL_PARAMS.get(model, {}).get("params", "?"),
            "avg_f1_macro": avg_f1,
            "cost_actual": cost_actual,
            "cost_per_1k_samples": cost_per_1k,
            "training_cost": training_cost,
            "total_cost": round((cost_actual or 0) + training_cost, 4),
            "provider": provider,
        })

    df = pd.DataFrame(rows)
    logger.info(f"Cost table:\n{df.to_string(index=False)}")
    return df


def plot_cost_vs_performance(
    df: pd.DataFrame,
    save_path: str = "results/figures/cost_vs_f1.pdf",
    figsize: tuple[int, int] = (9, 7),
    dpi: int = 300,
) -> None:
    """Scatter plot: cost per 1K samples (x, log scale) vs avg F1 Macro (y).

    Point size proportional to parameter count. Annotated with model names.

    Args:
        df: DataFrame from compute_cost_table().
        save_path: Path to save figure.
    """
    plot_df = df.dropna(subset=["avg_f1_macro"]).copy()
    # Exclude majority baseline from cost plot (cost=0, not meaningful)
    plot_df = plot_df[plot_df["model"] != "majority_baseline"]

    if len(plot_df) < 2:
        logger.warning("Not enough data for cost vs performance plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Replace 0 cost with small epsilon for log scale
    costs = plot_df["cost_per_1k_samples"].replace(0, 0.001)

    # Point sizes based on parameter count
    sizes = []
    for model in plot_df["model"]:
        params_numeric = float(MODEL_PARAMS.get(model, {}).get("params_numeric", 1e9))
        sizes.append(max(params_numeric / 1e9 * 20, 50))  # Scale: 1B = 20pt, min 50

    colors = {
        "qwen25_base": "#2196F3",
        "qwen25_finetuned": "#4CAF50",
        "llama70b": "#FF9800",
        "gemini25pro": "#F44336",
        "xlm_roberta_base": "#9C27B0",
    }
    c = [colors.get(m, "#999") for m in plot_df["model"]]

    ax.scatter(costs, plot_df["avg_f1_macro"], s=sizes, c=c, alpha=0.8, edgecolors="white", linewidths=1.5, zorder=3)

    # Labels
    display_names = {
        "qwen25_base": "Qwen 3B\n(zero-shot)",
        "qwen25_finetuned": "Qwen 3B\n(QLoRA)",
        "llama70b": "Llama 70B\n(zero-shot)",
        "gemini25pro": "Gemini 2.5 Pro\n(zero-shot)",
        "xlm_roberta_base": "XLM-R-base\n(EN fine-tuned)",
    }

    for _, row in plot_df.iterrows():
        name = display_names.get(row["model"], row["model"])
        ax.annotate(
            name,
            (costs[row.name], row["avg_f1_macro"]),
            textcoords="offset points",
            xytext=(15, 5),
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1K Samples (USD, log scale)", fontsize=13)
    ax.set_ylabel("Average F1 Macro", fontsize=13)
    ax.set_title("Cost-Performance Tradeoff", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Cost vs performance plot saved to {save_path}")

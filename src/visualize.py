"""
visualize.py — Paper-ready figures and tables for the 5×4 evaluation matrix.

Generates LaTeX/Markdown results tables, grouped bar charts, confusion matrix
grids, and exports all figures in publication-ready format.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

MODELS = [
    "majority_baseline",
    "qwen25_base",
    "qwen25_finetuned",
    "llama70b",
    "gemini25pro",
    "xlm_roberta_base",
]
LANGUAGES = ["english", "spanish", "hindi", "arabic"]  # ordered by cross-lingual transfer difficulty

MODEL_DISPLAY = {
    "majority_baseline": "Majority Baseline",
    "qwen25_base": "Qwen2.5 3B (zero-shot)",
    "qwen25_finetuned": "Qwen2.5 3B QLoRA (ours)",
    "llama70b": "Llama 3.3 70B (zero-shot)",
    "gemini25pro": "Gemini 2.5 Pro (zero-shot)",
    "xlm_roberta_base": "XLM-R-base (EN fine-tuned)",
}


def _load_results(results_dir: str = "results/eval_scores/aggregated/") -> dict:
    """Load all result JSONs into nested dict: {model: {lang: metrics}}."""
    data = {}
    for model in MODELS:
        data[model] = {}
        for lang in LANGUAGES:
            path = Path(results_dir) / f"{model}_{lang}.json"
            if path.exists():
                with open(path, "r") as f:
                    data[model][lang] = json.load(f)
            else:
                warnings.warn(
                    f"Missing aggregated result file: {path}. "
                    f"Metric will be 0.0 — figures may be incorrect. "
                    f"Run aggregation (Phase 6) before generating figures.",
                    UserWarning,
                    stacklevel=2,
                )
    return data


# --------------------------------------------------------------------------- #
#  Results Table                                                                #
# --------------------------------------------------------------------------- #

def generate_results_table(
    results_dir: str = "results/eval_scores/aggregated/",
    include_ci: bool = True,
    include_latency: bool = True,
) -> tuple[str, str]:
    """Generate 5×4 results table in Markdown and LaTeX.

    Rows: models. Columns: languages. Cells: F1 Macro [CI lower, CI upper].

    Args:
        results_dir: Directory with eval_scores JSONs.
        include_ci: Include bootstrap confidence intervals.
        include_latency: Include average latency column.

    Returns:
        (markdown_str, latex_str)
    """
    results_path = Path(results_dir)
    if results_path.name != "aggregated":
        results_path = results_path / "aggregated"

    data = _load_results(str(results_path))
    seed_dependent_models = {"qwen25_finetuned", "xlm_roberta_base"}

    def _get_latency_for_cell(model: str, lang: str, aggregated_metrics: dict) -> float | None:
        seed_root = results_path.parent

        if model in seed_dependent_models:
            seeds = aggregated_metrics.get("metadata", {}).get("seeds", [42, 123, 456])
            vals = []
            for s in seeds:
                p = seed_root / f"seed_{s}" / f"{model}_{lang}.json"
                if not p.exists():
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                v = m.get("avg_latency_seconds")
                if v is not None:
                    vals.append(float(v))
            if vals:
                return float(np.mean(vals))
        else:
            # Seed-agnostic models are persisted under seed_42 for compatibility.
            p = seed_root / "seed_42" / f"{model}_{lang}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                v = m.get("avg_latency_seconds")
                if v is not None:
                    return float(v)

        v = aggregated_metrics.get("avg_latency_seconds_mean")
        return float(v) if v is not None else None

    # Build Markdown
    cols = ["Model"] + [l.capitalize() for l in LANGUAGES]
    if include_latency:
        cols.append("Avg Latency (s)")

    md_lines = []
    md_lines.append("| " + " | ".join(cols) + " |")
    md_lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for model in MODELS:
        row = [MODEL_DISPLAY.get(model, model)]
        latencies = []

        for lang in LANGUAGES:
            metrics = data.get(model, {}).get(lang, {})
            f1_mean = metrics.get("f1_macro_mean")
            f1_std = metrics.get("f1_macro_std")
            if f1_mean is None:
                # Backward compatibility with non-aggregated files
                legacy_f1 = metrics.get("f1_macro")
                if legacy_f1 is None:
                    row.append("—")
                else:
                    row.append(f"{legacy_f1:.4f}")
            elif model in seed_dependent_models:
                row.append(f"{f1_mean:.4f} ± {f1_std:.4f}")
            else:
                if include_ci:
                    ci_l = metrics.get("f1_macro_ci_lower", "?")
                    ci_u = metrics.get("f1_macro_ci_upper", "?")
                    row.append(f"{f1_mean:.4f} [{ci_l}, {ci_u}]")
                else:
                    row.append(f"{f1_mean:.4f}")

            lat = _get_latency_for_cell(model, lang, metrics)
            if lat is not None:
                latencies.append(lat)

        if include_latency:
            if latencies:
                row.append(f"{np.mean(latencies):.3f}")
            else:
                row.append("—")

        md_lines.append("| " + " | ".join(row) + " |")

    markdown = "\n".join(md_lines)

    # Build LaTeX
    n_cols = len(LANGUAGES) + 1 + (1 if include_latency else 0)
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Cross-lingual sentiment classification results (F1 Macro with 95\% bootstrap CI).}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "c" * (n_cols - 1) + "}",
        r"\toprule",
    ]

    header = " & ".join(cols) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")

    for model in MODELS:
        row_parts = [MODEL_DISPLAY.get(model, model)]
        latencies = []

        for lang in LANGUAGES:
            metrics = data.get(model, {}).get(lang, {})
            f1_mean = metrics.get("f1_macro_mean")
            f1_std = metrics.get("f1_macro_std")
            if f1_mean is None:
                legacy_f1 = metrics.get("f1_macro")
                if legacy_f1 is None:
                    row_parts.append("—")
                else:
                    row_parts.append(f"{legacy_f1:.4f}")
            elif model in seed_dependent_models:
                value = f"{f1_mean:.4f} ± {f1_std:.4f}"
                if model == "qwen25_finetuned":
                    value = rf"\textbf{{{f1_mean:.4f}}} ± {f1_std:.4f}"
                row_parts.append(value)
            else:
                if include_ci:
                    ci_l = metrics.get("f1_macro_ci_lower", "?")
                    ci_u = metrics.get("f1_macro_ci_upper", "?")
                    row_parts.append(f"{f1_mean:.4f} [{ci_l}, {ci_u}]")
                else:
                    row_parts.append(f"{f1_mean:.4f}")

            lat = _get_latency_for_cell(model, lang, metrics)
            if lat is not None:
                latencies.append(lat)

        if include_latency:
            if latencies:
                row_parts.append(f"{np.mean(latencies):.3f}")
            else:
                row_parts.append("—")

        latex_lines.append(" & ".join(row_parts) + r" \\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(latex_lines)

    # Save to files
    md_path = results_path.parent / "results_table.md"
    tex_path = results_path.parent / "results_table.tex"
    md_path.write_text(markdown, encoding="utf-8")
    tex_path.write_text(latex, encoding="utf-8")
    logger.info(f"Results table saved to {md_path} and {tex_path}")

    return markdown, latex


# --------------------------------------------------------------------------- #
#  Grouped Bar Chart                                                            #
# --------------------------------------------------------------------------- #

SHORT_DISPLAY = {
    "majority_baseline": "Majority",
    "qwen25_base": "Qwen2.5 3B",
    "qwen25_finetuned": "Qwen2.5 3B QLoRA",
    "llama70b": "Llama 3.3 70B",
    "gemini25pro": "Gemini 2.5 Pro",
    "xlm_roberta_base": "XLM-R (EN FT)",
}

# Paper figures are drawn at their printed size (ACL column = 3.03 in, text width = 6.3 in)
# in a Times-like serif, so that text in the figure matches the caption size.
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}


def plot_f1_grouped_bar(
    results_dir: str = "results/eval_scores/aggregated/",
    save_path: str = "results/figures/f1_grouped_bar.pdf",
    figsize: tuple[float, float] = (3.3, 2.35),
    dpi: int = 300,
) -> None:
    """Grouped bar chart: x=language, bars=models, y=macro-F1 with seed-SD error bars.

    Drawn at ACL single-column size; include with width=\columnwidth.
    """
    data = _load_results(results_dir)
    plot_models = [m for m in MODELS if m != "majority_baseline"]
    x = np.arange(len(LANGUAGES))
    width = 0.16
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=figsize)
        for i, model in enumerate(plot_models):
            f1_vals, std_errors = [], []
            for lang in LANGUAGES:
                metrics = data.get(model, {}).get(lang, {})
                if "f1_macro_mean" in metrics:
                    f1 = metrics.get("f1_macro_mean", 0)
                    per_seed = metrics.get("f1_macro_per_seed") or []
                    if isinstance(per_seed, dict):
                        per_seed = list(per_seed.values())
                    std = float(np.std(per_seed, ddof=1)) if len(set(per_seed)) > 1 else 0.0
                else:
                    f1, std = metrics.get("f1_macro", 0), 0.0
                f1_vals.append(f1)
                std_errors.append(std)
            ax.bar(x + i * width, f1_vals, width, label=SHORT_DISPLAY.get(model, model),
                   color=colors[i], yerr=std_errors, capsize=1.5,
                   error_kw={"linewidth": 0.6, "capthick": 0.6})

        ax.set_ylabel("Macro-F1", fontsize=8)
        ax.set_xticks(x + width * (len(plot_models) - 1) / 2)
        ax.set_xticklabels([l.capitalize() for l in LANGUAGES], fontsize=8)
        ax.tick_params(axis="y", labelsize=7, length=2)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 0.85)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  frameon=False, handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
        fig.tight_layout(pad=0.2)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    logger.info(f"Grouped bar chart saved to {save_path}")


# --------------------------------------------------------------------------- #
#  Confusion Matrix Grid                                                        #
# --------------------------------------------------------------------------- #

def plot_confusion_matrices_grid(
    results_dir: str = "results/eval_scores/aggregated/",
    cache_dir: str = "results/api_cache/",
    save_path: str = "results/figures/confusion_matrices_grid.pdf",
    figsize: tuple[float, float] = (6.3, 8.0),
    dpi: int = 300,
    seeds: tuple[int, ...] = (42, 123, 456),
) -> None:
    """6x4 grid of confusion matrices (rows=models, cols=languages), drawn at ACL text width.

    For the seed-dependent models (Qwen QLoRA, XLM-R) each matrix is the mean over all
    seeds, matching how every other number in the paper is reported.
    """
    from sklearn.metrics import confusion_matrix as cm_func
    from src.evaluate import load_predictions_for_model

    short = ["neg", "neu", "pos"]
    seeded = {"qwen25_finetuned", "xlm_roberta_base"}
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(MODELS), len(LANGUAGES), figsize=figsize)
        for i, model in enumerate(MODELS):
            for j, lang in enumerate(LANGUAGES):
                ax = axes[i, j]
                seed_args = seeds if model in seeded else (None,)
                mats = []
                for s in seed_args:
                    y_true, y_pred = load_predictions_for_model(model, lang, results_dir, cache_dir, seed=s)
                    mats.append(cm_func(y_true, y_pred, labels=[0, 1, 2]))
                cm = np.mean(mats, axis=0)
                fmt = "d" if len(mats) == 1 else ".1f"
                if len(mats) == 1:
                    cm = cm.astype(int)
                sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", vmin=0, vmax=290,
                            xticklabels=short if i == len(MODELS) - 1 else False,
                            yticklabels=short if j == 0 else False,
                            ax=ax, cbar=False, square=True, linewidths=0.4, linecolor="white",
                            annot_kws={"fontsize": 7})
                ax.tick_params(length=0, labelsize=6.5, pad=1.5)
                if j == 0:
                    ax.set_yticklabels(short, rotation=0)
                if i == 0:
                    ax.set_title(lang.capitalize(), fontsize=9, fontweight="bold", pad=4)
                if j == 0:
                    ax.set_ylabel(SHORT_DISPLAY.get(model, model), fontsize=8, fontweight="bold", labelpad=4)
                if i == len(MODELS) - 1:
                    ax.set_xlabel("predicted", fontsize=7, labelpad=2)
        fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    logger.info(f"Confusion matrix grid saved to {save_path}")


def generate_all_individual_confusion_matrices(
    results_dir: str = "results/eval_scores/aggregated/",
    cache_dir: str = "results/api_cache/",
    save_dir: str = "results/confusion_matrices/",
    seed: int = 42,
    dpi: int = 300,
) -> None:
    """Generate one confusion matrix PDF per model×language pair."""
    from src.evaluate import load_predictions_for_model, compute_confusion_matrix

    for model in MODELS:
        for lang in LANGUAGES:
            seed_arg = seed if model in {"qwen25_finetuned", "xlm_roberta_base"} else None
            y_true, y_pred = load_predictions_for_model(
                model,
                lang,
                results_dir=results_dir,
                cache_dir=cache_dir,
                seed=seed_arg,
            )
            compute_confusion_matrix(
                y_true,
                y_pred,
                save_path=f"{save_dir}/{model}_{lang}.pdf",
                title=f"{MODEL_DISPLAY.get(model, model)} — {lang.capitalize()}",
                figsize=(7, 6),
                dpi=dpi,
            )


# XLM-R excluded from fertility plot — it uses a distinct multilingual
# SentencePiece tokenizer; cross-tokenizer fertility comparison is not
# meaningful on a single axis.


# --------------------------------------------------------------------------- #
#  Export All Figures                                                            #
# --------------------------------------------------------------------------- #

def export_all_figures(
    results_dir: str = "results/",
    dpi: int = 300,
    fmt: str = "pdf",
) -> None:
    """Generate all publication-ready figures.

    Args:
        results_dir: Base results directory.
        dpi: Resolution.
        fmt: File format (pdf or png).
    """
    eval_dir = f"{results_dir}/eval_scores/aggregated"
    cache_dir = f"{results_dir}/api_cache"

    logger.info("Generating all figures...")

    generate_results_table(eval_dir)
    plot_f1_grouped_bar(eval_dir, f"{results_dir}/figures/f1_grouped_bar.{fmt}", dpi=dpi)
    plot_confusion_matrices_grid(eval_dir, cache_dir,
                                 f"{results_dir}/figures/confusion_matrices_grid.{fmt}", dpi=dpi)
    generate_all_individual_confusion_matrices(
        results_dir=eval_dir,
        cache_dir=cache_dir,
        save_dir=f"{results_dir}/confusion_matrices",
        dpi=dpi,
    )

    logger.info("All figures exported!")

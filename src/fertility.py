"""
fertility.py — Token fertility analysis across languages.

Computes average tokens per whitespace word using the Qwen2.5 tokenizer,
and correlates fertility with cross-lingual F1 Macro performance.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)


def compute_fertility(texts: list[str], tokenizer) -> float:
    """Compute average tokens per whitespace-word across all texts.

    fertility = sum(n_tokens) / sum(n_words)

    Args:
        texts: List of text strings.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Float fertility score (tokens per word).
    """
    total_tokens = 0
    total_words = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        total_tokens += len(tokens)
        total_words += len(words)

    if total_words == 0:
        return 0.0

    fertility = total_tokens / total_words
    return round(fertility, 4)


def fertility_analysis(
    datasets_dict: dict,
    tokenizer,
    languages: list[str],
    results_dir: str = "results/eval_scores/aggregated/",
    model_name: str = "qwen25_finetuned",
) -> pd.DataFrame:
    """Compute fertility for each language and correlate with F1 Macro.

    Loads F1 scores from saved result JSONs.

    Args:
        datasets_dict: {lang: Dataset} — the test subsamples.
        tokenizer: Qwen2.5 tokenizer.
        languages: List of language codes.
        results_dir: Where eval_scores JSONs are stored.
        model_name: Model to load F1 scores for.

    Returns:
        DataFrame with columns: language, fertility, f1_macro, f1_delta_from_english.
    """
    import json

    rows = []
    english_f1 = None

    for lang in languages:
        texts = datasets_dict[lang]["text"]
        fert = compute_fertility(texts, tokenizer)

        # Load F1 score
        result_path = Path(results_dir) / f"{model_name}_{lang}.json"
        f1_macro = None
        if result_path.exists():
            with open(result_path, "r") as f:
                data = json.load(f)
            f1_macro = data.get("f1_macro_mean", data.get("f1_macro"))

        if lang == "english":
            english_f1 = f1_macro

        rows.append({
            "language": lang,
            "fertility": fert,
            "f1_macro": f1_macro,
        })

    df = pd.DataFrame(rows)

    # Compute delta from English
    if english_f1 is not None:
        df["f1_delta_from_english"] = df["f1_macro"].apply(
            lambda x: round(x - english_f1, 4) if x is not None else None
        )
    else:
        df["f1_delta_from_english"] = None

    logger.info(f"Fertility analysis:\n{df.to_string(index=False)}")
    return df


def plot_fertility_vs_f1(
    df: pd.DataFrame,
    save_path: str = "results/figures/fertility_vs_f1.pdf",
    figsize: tuple[int, int] = (8, 6),
    dpi: int = 300,
) -> None:
    """Scatter plot: fertility (x) vs F1 Macro (y), annotated with language names.

    Includes Pearson correlation coefficient and regression line.

    Args:
        df: DataFrame from fertility_analysis().
        save_path: Path to save figure.
        figsize: Figure size.
        dpi: Resolution.
    """
    # Filter rows with valid F1 scores
    plot_df = df.dropna(subset=["f1_macro"]).copy()

    if len(plot_df) < 2:
        logger.warning("Not enough data points for fertility vs F1 plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        plot_df["fertility"],
        plot_df["f1_macro"],
        s=120,
        c="#2196F3",
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    # Annotate points
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["language"].capitalize(),
            (row["fertility"], row["f1_macro"]),
            textcoords="offset points",
            xytext=(10, 8),
            fontsize=11,
            fontweight="bold",
        )

    # Pearson correlation
    r, p_value = stats.pearsonr(plot_df["fertility"], plot_df["f1_macro"])

    # Regression line
    z = np.polyfit(plot_df["fertility"], plot_df["f1_macro"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_df["fertility"].min() - 0.1, plot_df["fertility"].max() + 0.1, 100)
    ax.plot(x_line, p(x_line), "--", color="#FF5722", alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Token Fertility (tokens/word)", fontsize=13)
    ax.set_ylabel("F1 Macro", fontsize=13)
    # No in-figure title: the caption carries the title in ACL/EMNLP format. The
    # correlation is annotated inside the axes instead, where it reads as data
    # rather than as a duplicate heading.
    ax.annotate(
        f"Pearson $r$ = {r:.3f} ($p$ = {p_value:.3f}, $n$ = {len(plot_df)})",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Fertility vs F1 plot saved to {save_path}")

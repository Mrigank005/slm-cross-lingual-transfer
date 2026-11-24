"""
dataset.py — Data loading, formatting, stratified sampling, and index management.

Handles cardiffnlp/tweet_sentiment_multilingual dataset for cross-lingual
sentiment transfer research. All functions are deterministic (seed=42).
"""

import json
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #

DATASET_NAME = "cardiffnlp/tweet_sentiment_multilingual"
LANGUAGES = ["english", "hindi", "arabic", "spanish"]
LABEL_NAMES = ["negative", "neutral", "positive"]

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_MAP_INV = {"negative": 0, "neutral": 1, "positive": 2}

# Import canonical prompts — single source of truth is src/prompts.py
from src.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

# Backward-compatible alias (used by format_for_sft formatting)
USER_PROMPT_TEMPLATE = "Text: {text}"  # kept for legacy; prefer build_user_prompt()


# --------------------------------------------------------------------------- #
#  Data Loading                                                                 #
# --------------------------------------------------------------------------- #

def load_all_splits(
    languages: list[str] | None = None,
) -> dict[str, dict[str, Dataset]]:
    """Load train/validation/test splits for specified languages.

    Args:
        languages: List of language codes. Defaults to LANGUAGES.

    Returns:
        Nested dict: {language: {"train": Dataset, "validation": Dataset, "test": Dataset}}

    Example:
        >>> data = load_all_splits(["english", "hindi"])
        >>> len(data["english"]["train"])
        1838
    """
    if languages is None:
        languages = LANGUAGES

    result = {}
    for lang in languages:
        logger.info(f"Loading dataset for language: {lang}")
        try:
            ds = load_dataset(DATASET_NAME, lang)
            train_ds = ds["train"]
            val_ds = ds["validation"]
            test_ds = ds["test"]
        except RuntimeError as e:
            # datasets>=3.0 disables script-based dataset loading for some repos.
            # Fallback to the Hub's generated parquet exports (script-free path).
            if "Dataset scripts are no longer supported" not in str(e):
                raise

            logger.debug(
                "Script-based loader unavailable for %s (%s). "
                "Falling back to parquet exports.",
                DATASET_NAME,
                lang,
            )
            train_ds = _load_split_from_parquet(DATASET_NAME, lang, "train")
            val_ds = _load_split_from_parquet(DATASET_NAME, lang, "validation")
            test_ds = _load_split_from_parquet(DATASET_NAME, lang, "test")

        result[lang] = {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
        logger.info(
            f"  {lang}: train={len(train_ds)}, "
            f"val={len(val_ds)}, test={len(test_ds)}"
        )
    return result


@lru_cache(maxsize=1)
def _get_parquet_index(dataset_name: str) -> dict[tuple[str, str], list[str]]:
    """Fetch Hub parquet index from datasets-server and map (config, split)->urls."""
    endpoint = (
        "https://datasets-server.huggingface.co/parquet?dataset="
        f"{quote(dataset_name, safe='')}"
    )
    with urlopen(endpoint, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    files = payload.get("parquet_files", [])
    index: dict[tuple[str, str], list[str]] = {}
    for row in files:
        config = row.get("config")
        split = row.get("split")
        url = row.get("url")
        if not config or not split or not url:
            continue
        index.setdefault((config, split), []).append(url)
    return index


def _load_split_from_parquet(dataset_name: str, config: str, split: str) -> Dataset:
    """Load one split from Hub parquet exports without executing dataset scripts."""
    parquet_index = _get_parquet_index(dataset_name)
    urls = parquet_index.get((config, split), [])
    if not urls:
        raise RuntimeError(
            f"No parquet export found for dataset={dataset_name}, "
            f"config={config}, split={split}."
        )

    # The resulting parquet dataset has a single logical split named "train".
    return load_dataset("parquet", data_files=urls, split="train")


def get_label_map() -> tuple[dict[int, str], dict[str, int]]:
    """Return (int→str, str→int) label mappings.

    Returns:
        (LABEL_MAP, LABEL_MAP_INV)
    """
    return LABEL_MAP, LABEL_MAP_INV


# --------------------------------------------------------------------------- #
#  Instruction Formatting (ChatML via tokenizer)                                #
# --------------------------------------------------------------------------- #

_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")


def _decode_unicode_escapes(text: str) -> str:
    """Decode literal unicode escape sequences like ``\\u2019`` into characters.

    This is a no-throw helper for defensive preprocessing. If decoding cannot
    be performed safely, the original text is returned unchanged.
    """
    if "\\" not in text:
        return text

    def _replace(match: re.Match) -> str:
        hex_part = match.group(1) or match.group(2)
        try:
            return chr(int(hex_part, 16))
        except (TypeError, ValueError, OverflowError):
            return match.group(0)

    try:
        return _UNICODE_ESCAPE_RE.sub(_replace, text)
    except Exception:
        return text

def format_for_sft(example: dict, tokenizer) -> dict:
    """Format a single example into ChatML instruction format for SFTTrainer.

    Uses tokenizer.apply_chat_template() to produce the correct special tokens.
    DO NOT manually construct <|im_start|>/<|im_end|> strings.

    Args:
        example: Dict with keys "text" and "label" (int 0/1/2).
        tokenizer: Qwen2.5 tokenizer with chat template support.

    Returns:
        Dict with added "formatted_text" key containing the full conversation.

    Raises:
        ValueError: If text is empty or whitespace-only.
    """
    text = _decode_unicode_escapes(example["text"])
    if not text or not text.strip():
        raise ValueError(
            f"Empty or whitespace-only text found in dataset "
            f"(label={example.get('label')}).  "
            f"Filter empty rows before calling format_for_sft."
        )

    label_text = LABEL_MAP[example["label"]]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
        {"role": "assistant", "content": label_text},
    ]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    return {**example, "formatted_text": formatted}


def format_dataset_for_sft(dataset: Dataset, tokenizer) -> Dataset:
    """Apply format_for_sft to an entire dataset.

    Args:
        dataset: HuggingFace Dataset with "text" and "label" columns.
        tokenizer: Qwen2.5 tokenizer.

    Returns:
        Dataset with added "formatted_text" column.
    """
    return dataset.map(
        lambda ex: format_for_sft(ex, tokenizer),
        desc="Formatting for SFT",
    )


# --------------------------------------------------------------------------- #
#  Train/Val Preparation                                                        #
# --------------------------------------------------------------------------- #

def prepare_english_train_val(
    dataset_dict: dict[str, dict[str, Dataset]],
    tokenizer,
) -> tuple[Dataset, Dataset]:
    """Prepare English train and validation sets for SFT training.

    Uses the existing cardiffnlp validation split (323 samples) rather than
    re-splitting the training data.

    Args:
        dataset_dict: Output of load_all_splits().
        tokenizer: Qwen2.5 tokenizer for formatting.

    Returns:
        (train_dataset, val_dataset): Both formatted for SFT with "formatted_text".
    """
    eng = dataset_dict["english"]
    train_ds = format_dataset_for_sft(eng["train"], tokenizer)
    val_ds = format_dataset_for_sft(eng["validation"], tokenizer)

    logger.info(f"English train: {len(train_ds)} samples, val: {len(val_ds)} samples")
    return train_ds, val_ds


# --------------------------------------------------------------------------- #
#  Stratified Subsampling                                                       #
# --------------------------------------------------------------------------- #

def get_stratified_subsample(
    dataset: Dataset,
    n_samples: int = 200,
    seed: int = 42,
    indices_override: list[int] | None = None,
) -> Dataset:
    """Get a stratified random subsample preserving class distribution.

    Args:
        dataset: HuggingFace Dataset (a test split for one language).
        n_samples: Number of samples to select. Default 200.
        seed: Random seed for reproducibility. Default 42.
        indices_override: If provided, select these exact indices instead of
            sampling. Used when loading from subsample_indices.json.

    Returns:
        Subsampled Dataset with n_samples rows.

    Raises:
        ValueError: If n_samples > len(dataset).
        ValueError: If any index in indices_override is out of bounds.
    """
    if indices_override is not None:
        # Validate indices
        max_idx = len(dataset) - 1
        out_of_bounds = [i for i in indices_override if i < 0 or i > max_idx]
        if out_of_bounds:
            raise ValueError(
                f"Out of bounds indices: {out_of_bounds[:5]}... "
                f"(dataset has {len(dataset)} samples)"
            )
        # Validate uniqueness
        if len(set(indices_override)) != len(indices_override):
            raise ValueError(
                f"Duplicate indices found in indices_override: "
                f"{len(indices_override)} total, {len(set(indices_override))} unique"
            )
        # Validate expected count
        if len(indices_override) != n_samples:
            logger.debug(
                f"indices_override has {len(indices_override)} indices, "
                f"expected {n_samples}. Using provided indices as-is."
            )
        return dataset.select(indices_override)

    if n_samples > len(dataset):
        raise ValueError(
            f"n_samples ({n_samples}) > dataset size ({len(dataset)})"
        )

    labels = np.array(dataset["label"])
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    # Compute proportional allocation per class
    allocations = {}
    for label, count in zip(unique_labels, counts):
        allocations[label] = round(n_samples * count / total)

    # Adjust rounding to ensure sum == n_samples exactly
    diff = n_samples - sum(allocations.values())
    if diff != 0:
        # Add/remove from the largest class
        largest_label = unique_labels[np.argmax(counts)]
        allocations[largest_label] += diff

    # Sample indices per class
    rng = np.random.default_rng(seed)
    selected_indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        sampled = rng.choice(label_indices, size=allocations[label], replace=False)
        selected_indices.extend(sampled.tolist())

    # Sort ascending for deterministic ordering
    selected_indices.sort()

    return dataset.select(selected_indices)


def _get_class_distribution(dataset: Dataset, indices: list[int]) -> dict[str, int]:
    """Count per-class samples for given indices."""
    labels = [dataset[i]["label"] for i in indices]
    dist = {}
    for name, code in LABEL_MAP_INV.items():
        dist[name] = labels.count(code)
    return dist


# --------------------------------------------------------------------------- #
#  Subsample Index Management                                                   #
# --------------------------------------------------------------------------- #

def save_subsample_indices(
    datasets_dict: dict[str, Dataset],
    n_samples: int = 200,
    seed: int = 42,
    save_path: str = "results/eval_scores/subsample_indices.json",
) -> dict:
    """Generate and save stratified subsample indices for all languages.

    This function is called ONCE before any evaluation. It generates the
    canonical subsample indices and writes them to disk. All subsequent
    evaluation runs load from this file.

    Args:
        datasets_dict: {"english": test_dataset, "hindi": test_dataset, ...}
        n_samples: Samples per language. Default 200.
        seed: Random seed. Default 42.
        save_path: Path to write JSON.

    Returns:
        Dict matching the saved schema.

    Raises:
        FileExistsError: If save_path already exists. Never silently overwrite.
            To regenerate, delete the file manually first.
    """
    path = Path(save_path)
    if path.exists():
        raise FileExistsError(
            f"Subsample indices file already exists at {save_path}. "
            "Delete it manually to regenerate."
        )

    result = {
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size_per_language": n_samples,
        "total_samples": n_samples * len(datasets_dict),
        "indices": {},
        "class_distribution": {},
    }

    for lang, dataset in datasets_dict.items():
        # Generate subsample to get the indices
        subsample = get_stratified_subsample(dataset, n_samples, seed)

        # Recover the selected indices by matching
        # Since get_stratified_subsample uses deterministic seed, we can
        # reproduce the index selection logic here
        labels = np.array(dataset["label"])
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        allocations = {}
        for label, count in zip(unique_labels, counts):
            allocations[label] = round(n_samples * count / total)

        diff = n_samples - sum(allocations.values())
        if diff != 0:
            largest_label = unique_labels[np.argmax(counts)]
            allocations[largest_label] += diff

        rng = np.random.default_rng(seed)
        selected_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled = rng.choice(label_indices, size=allocations[label], replace=False)
            selected_indices.extend(sampled.tolist())

        selected_indices.sort()
        result["indices"][lang] = selected_indices
        result["class_distribution"][lang] = _get_class_distribution(
            dataset, selected_indices
        )

        logger.info(
            f"  {lang}: {len(selected_indices)} indices, "
            f"distribution={result['class_distribution'][lang]}"
        )

    # Write atomically
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)

    logger.info(f"Subsample indices saved to {save_path}")
    return result


def load_subsample_indices(
    load_path: str = "results/eval_scores/subsample_indices.json",
) -> dict:
    """Load previously saved subsample indices.

    Raises:
        FileNotFoundError: If file does not exist. NEVER silently regenerate.
            The user must explicitly call save_subsample_indices() first.

    Returns:
        Parsed JSON dict with keys: seed, created_at, sample_size_per_language,
        total_samples, indices, class_distribution.
    """
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Subsample indices file not found at {load_path}. "
            "Run save_subsample_indices() first (typically in notebook 01). "
            "NEVER silently regenerate — indices must be canonical."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        f"Loaded subsample indices: {data['total_samples']} total samples, "
        f"seed={data['seed']}, created={data['created_at']}"
    )
    return data

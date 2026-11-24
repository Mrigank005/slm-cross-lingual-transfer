"""
train.py — QLoRA training pipeline for Qwen2.5-3B-Instruct.

Uses Unsloth + TRL SFTTrainer with Drive checkpointing and W&B logging.
Designed for Google Colab free tier (T4 GPU, 16GB VRAM).

Import guidance
---------------
Unsloth must be imported BEFORE transformers, trl, peft so that its
monkey-patches take effect on all downstream classes. The top-level
``import unsloth`` is done conditionally to avoid crashing on machines
that don't have it installed (e.g. local linting or evaluation-only
environments).
"""

import gc
import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import yaml
import wandb

# ── Unsloth: import first, but soft-fail for non-GPU environments ──────────
try:
    import unsloth  # noqa: F401 — must be imported before transformers/trl
    _UNSLOTH_AVAILABLE = True
except ImportError:
    _UNSLOTH_AVAILABLE = False

from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

# ── DataCollatorForCompletionOnlyLM: resolve from trl or use fallback  ─────
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    except ImportError:
        DataCollatorForCompletionOnlyLM = None  # set below after class def


class _FallbackCollator:
    """Response-only loss masking collator — used when TRL doesn't export one.

    Keeps labels only for the assistant completion segment that starts
    after ``response_template``. Prompt and padding tokens are masked to
    ``ignore_index`` (-100).

    Handles:
    - Variable-length sequences via tokenizer padding.
    - Nested tensor shapes from notebook verification calls.
    - Alternate templates with/without trailing newline.
    - Fail-fast if the template is never found (prevents silent 0-loss training).
    """

    def __init__(self, response_template: str, tokenizer, ignore_index: int = -100):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

        self.template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        alt_template = response_template.rstrip("\n")
        self.alt_template_ids = (
            tokenizer.encode(alt_template, add_special_tokens=False)
            if alt_template != response_template
            else []
        )

        if not self.template_ids and not self.alt_template_ids:
            raise ValueError(
                f"response_template '{response_template}' tokenization "
                f"returned empty IDs — cannot locate assistant turn."
            )

        _logger = logging.getLogger(__name__)
        _logger.info(
            f"FallbackCollator initialized: template_ids={self.template_ids}, "
            f"alt_template_ids={self.alt_template_ids}"
        )

    @staticmethod
    def _find_subsequence(seq: list[int], subseq: list[int]) -> int:
        """Return start index of first occurrence, or -1."""
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i:i + m] == subseq:
                return i
        return -1

    def __call__(self, features):
        # ── Normalize each feature to plain lists ──────────────────────
        normalized = []
        for f in features:
            item = {}
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key not in f:
                    continue
                value = f[key]
                if isinstance(value, torch.Tensor):
                    # Squeeze extra batch dims: (1, seq) → (seq,)
                    while value.dim() > 1:
                        value = value.squeeze(0)
                    value = value.tolist()
                elif isinstance(value, list) and value and isinstance(value[0], list):
                    value = value[0]
                item[key] = value
            normalized.append(item)

        # ── Pad to uniform length ────────────────────────────────────
        batch = self.tokenizer.pad(
            normalized,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Mask padding tokens
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = self.ignore_index

        # ── Apply response-template masking per row ──────────────────
        template_misses = 0
        for row_idx in range(input_ids.size(0)):
            seq = input_ids[row_idx].tolist()

            # Try primary template
            pos = self._find_subsequence(seq, self.template_ids)
            template_len = len(self.template_ids)

            # Try alternate (without trailing \n)
            if pos == -1 and self.alt_template_ids:
                pos = self._find_subsequence(seq, self.alt_template_ids)
                template_len = len(self.alt_template_ids)

            if pos == -1:
                template_misses += 1
                labels[row_idx, :] = self.ignore_index
                continue

            # Mask everything before and including the template
            start = pos + template_len
            labels[row_idx, :start] = self.ignore_index

        # ── Fail-fast guards ──────────────────────────────────────────
        active_tokens = (labels != self.ignore_index).sum().item()

        if template_misses == input_ids.size(0):
            raise RuntimeError(
                f"CRITICAL: Assistant response template IDs "
                f"{self.template_ids} were NOT found in ANY of the "
                f"{input_ids.size(0)} sequences in this batch.\n"
                f"This means the collator will mask ALL tokens and the "
                f"model will learn nothing.\n"
                f"Check that format_for_sft() produces valid Qwen2.5 "
                f"ChatML with the '<|im_start|>assistant\\n' turn present."
            )

        if active_tokens == 0:
            raise RuntimeError(
                f"All labels are masked (-100). No training signal "
                f"remains after loss masking. template_misses = "
                f"{template_misses}/{input_ids.size(0)}."
            )

        batch["labels"] = labels
        return batch


# If TRL didn't provide it, register our fallback
if DataCollatorForCompletionOnlyLM is None:
    DataCollatorForCompletionOnlyLM = _FallbackCollator


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Config Loading                                                               #
# --------------------------------------------------------------------------- #

def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# --------------------------------------------------------------------------- #
#  Model Loading                                                                #
# --------------------------------------------------------------------------- #

def load_model_and_tokenizer(config: dict) -> tuple:
    """Load Qwen2.5-3B-Instruct via Unsloth in 4-bit with LoRA adapters.

    Clears CUDA cache before loading to maximize available VRAM on T4.
    """
    from unsloth import FastLanguageModel

    # Free any cached VRAM from previous loads/retries
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_cfg = config["model"]
    lora_cfg = config["lora"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=model_cfg.get("dtype"),
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",  # Unsloth optimized
        random_state=config["training"]["seed"],
    )

    logger.info(
        f"Model loaded: {model_cfg['name']}, "
        f"LoRA r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, "
        f"targets={lora_cfg['target_modules']}"
    )

    return model, tokenizer


# --------------------------------------------------------------------------- #
#  W&B Setup                                                                    #
# --------------------------------------------------------------------------- #

def setup_wandb(config: dict) -> str:
    """Initialize Weights & Biases with resume support.

    Persists run_id to drive_path/wandb_run_id.json on first run.
    On resume, loads run_id and continues the same W&B run.

    Returns:
        run_id str.
    """
    wandb_cfg = config.get("wandb", {})
    drive_path = Path(config["checkpointing"]["drive_path"])
    run_id_file = drive_path / "wandb_run_id.json"

    if run_id_file.exists():
        saved = json.loads(run_id_file.read_text())
        run_id = saved["run_id"]
        run = wandb.init(
            id=run_id,
            resume="allow",
            project=wandb_cfg.get("project", "qwen-sentiment-crosslingual"),
            config=config,
        )
        logger.info(f"Resuming W&B run: {run_id}")
    else:
        run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            project=wandb_cfg.get("project", "qwen-sentiment-crosslingual"),
            entity=wandb_cfg.get("entity"),
            config=config,
            name=f"qwen25-3b-qlora-sentiment-{run_suffix}",
            tags=["qwen2.5", "qlora", "sentiment", "cross-lingual"],
        )
        run_id = run.id
        drive_path.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(json.dumps({"run_id": run_id}))
        logger.info(f"Started new W&B run: {run_id}")

    return run_id


# --------------------------------------------------------------------------- #
#  Trainer Creation                                                             #
# --------------------------------------------------------------------------- #

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: dict,
) -> SFTTrainer:
    """Create SFTTrainer with evaluation, checkpointing, and early stopping.

    Uses the DataCollatorForCompletionOnlyLM to ensure only assistant
    response tokens contribute to the loss (prompt tokens are masked).
    """
    train_cfg = config["training"]
    ckpt_cfg = config["checkpointing"]

    output_dir = ckpt_cfg.get("drive_path", "./checkpoints")

    # Hard-fail if output_dir doesn't exist or isn't writable
    validate_output_dir(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
        seed=train_cfg["seed"],
        report_to=train_cfg.get("report_to", "wandb"),
        optim="adamw_8bit",  # Memory-efficient optimizer for QLoRA
    )

    # Response-only loss masking: only compute loss on the assistant response
    response_template = "<|im_start|>assistant\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    early_stopping_patience = train_cfg.get("early_stopping_patience", 2)

    # ── CRITICAL: Rename formatted_text → text ──────────────────────
    # Unsloth's patched SFTTrainer ignores dataset_text_field and always
    # tokenizes the "text" column. Our datasets have both:
    #   - "text" = raw tweet (NOT what we want)
    #   - "formatted_text" = ChatML with <|im_start|>assistant\n
    # We must replace "text" with "formatted_text" so Unsloth tokenizes
    # the correct column containing the response template.
    for ds_name, ds in [("train", train_dataset), ("val", val_dataset)]:
        if "formatted_text" not in ds.column_names:
            raise RuntimeError(
                f"{ds_name} dataset is missing 'formatted_text' column. "
                f"Available columns: {ds.column_names}. "
                f"Ensure prepare_english_train_val() was called first."
            )
    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c != "formatted_text"]
    )
    train_dataset = train_dataset.rename_column("formatted_text", "text")
    val_dataset = val_dataset.remove_columns(
        [c for c in val_dataset.column_names if c != "formatted_text"]
    )
    val_dataset = val_dataset.rename_column("formatted_text", "text")
    logger.info(
        f"Dataset columns renamed: formatted_text → text "
        f"(train={train_dataset.column_names}, val={val_dataset.column_names})"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=config["model"]["max_seq_length"],
        packing=False,  # No packing for classification tasks
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
        )],
    )

    logger.info(
        f"Trainer created: max_epochs={train_cfg['num_train_epochs']}, "
        f"effective_batch={train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}, "
        f"lr={train_cfg['learning_rate']}, "
        f"early_stopping_patience={early_stopping_patience}"
    )

    return trainer


# --------------------------------------------------------------------------- #
#  Training with Resume                                                         #
# --------------------------------------------------------------------------- #

def _find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint in a directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    checkpoints = list(ckpt_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None

    valid_checkpoints = []
    for p in checkpoints:
        try:
            step_num = int(p.name.split("-")[-1])
            valid_checkpoints.append((step_num, p))
        except ValueError:
            logger.warning(f"Skipping malformed checkpoint directory: {p.name}")
            continue

    if not valid_checkpoints:
        return None

    valid_checkpoints.sort(key=lambda x: x[0])
    latest = str(valid_checkpoints[-1][1])
    logger.info(f"Found latest checkpoint: {latest}")
    return latest


def train_with_resume(
    trainer: SFTTrainer,
    config: dict,
    checkpoint_dir: str | None = None,
) -> tuple:
    """Train, auto-resuming from latest checkpoint if session was interrupted.

    Handles 3 states:
      1. training_complete.json exists → skip training, load best checkpoint
      2. Checkpoints exist, no marker → resume interrupted training
      3. No checkpoints → train from scratch

    After training finishes, writes training_complete.json so future
    calls to this function skip training entirely.

    Returns:
        (model, tokenizer): Best checkpoint model and tokenizer.
    """
    if checkpoint_dir is None:
        checkpoint_dir = trainer.args.output_dir

    ckpt_path = Path(checkpoint_dir)
    completion_marker = ckpt_path / "training_complete.json"

    # ── State 1: Training already completed in a prior run ─────────
    if completion_marker.exists():
        marker_data = json.loads(completion_marker.read_text())
        best_ckpt = marker_data.get("best_checkpoint")
        best_metric = marker_data.get("best_metric")
        logger.info(
            f"Training already completed (found {completion_marker}). "
            f"Skipping training. Best checkpoint: {best_ckpt}, "
            f"best val loss: {best_metric}"
        )
        print(
            f"✅ Training already completed in a prior run.\n"
            f"   Best checkpoint: {best_ckpt}\n"
            f"   Best val loss: {best_metric}\n"
            f"   Skipping training — loading best checkpoint directly."
        )

        if best_ckpt and Path(best_ckpt).exists():
            from unsloth import FastLanguageModel

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model, tokenizer = FastLanguageModel.from_pretrained(
                best_ckpt,
                max_seq_length=config["model"]["max_seq_length"],
                load_in_4bit=config["model"]["load_in_4bit"],
            )
            return model, tokenizer
        else:
            logger.warning(
                f"Marker says best_ckpt={best_ckpt} but path does not "
                f"exist. Deleting marker and re-training."
            )
            completion_marker.unlink()
            # Fall through to State 2/3

    # ── State 2/3: Train (from scratch or resume) ──────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        logger.info(f"VRAM free before training: {vram_free:.2f} GB")

    resume_from = _find_latest_checkpoint(checkpoint_dir)

    if resume_from:
        logger.info(f"Resuming training from: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        logger.info("Starting training from scratch")
        trainer.train()

    # ── Post-training: reload best checkpoint ──────────────────────
    from unsloth import FastLanguageModel

    best_ckpt = trainer.state.best_model_checkpoint
    best_metric = getattr(trainer.state, "best_metric", None)

    if best_ckpt:
        ckpt_dir = Path(best_ckpt)
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [f for f in required_files if not (ckpt_dir / f).exists()]
        if missing:
            raise RuntimeError(
                f"Checkpoint at {best_ckpt} is missing required adapter files: "
                f"{missing}\n"
                f"Available files: {list(ckpt_dir.iterdir())}\n"
                f"Cannot reload best checkpoint."
            )
        logger.info(f"Checkpoint integrity verified: {best_ckpt}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Reloading best checkpoint from: {best_ckpt}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            best_ckpt,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
        )
        logger.info(
            f"Best checkpoint: {best_ckpt}, "
            f"Best val loss: {best_metric:.4f}"
        )
    else:
        model = trainer.model
        tokenizer = trainer.tokenizer
        logger.warning(
            "No best checkpoint found — using final epoch weights. "
            "Check that evaluation_strategy and save_strategy are both 'epoch'."
        )

    # ── Write completion marker so future calls skip training ──────
    marker = {
        "best_checkpoint": best_ckpt,
        "best_metric": best_metric,
        "completed_at": datetime.now().isoformat(),
        "total_steps": trainer.state.global_step,
    }
    ckpt_path.mkdir(parents=True, exist_ok=True)
    completion_marker.write_text(json.dumps(marker, indent=2))
    logger.info(f"Training completion marker written to {completion_marker}")

    logger.info("Training complete!")
    return model, tokenizer


# --------------------------------------------------------------------------- #
#  Validation Utilities                                                        #
# --------------------------------------------------------------------------- #

def validate_output_dir(output_dir: str) -> None:
    """Verify output_dir exists and is writable before training starts.

    Raises:
        RuntimeError: If path does not exist or is not writable.
    """
    path = Path(output_dir)
    if not path.exists():
        raise RuntimeError(
            f"Output directory does not exist: {output_dir}\n"
            f"If using Google Drive, ensure it is mounted first:\n"
            f"  from google.colab import drive\n"
            f"  drive.mount('/content/drive')"
        )
    # Test writability
    test_file = path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        raise RuntimeError(
            f"Output directory is not writable: {output_dir}\n"
            f"Error: {e}"
        )


def log_class_distribution(dataset, split_name: str) -> None:
    """Log label distribution and warn if severely imbalanced."""
    counts = Counter(dataset["label"])
    total = len(dataset)
    label_names = {0: "negative", 1: "neutral", 2: "positive"}

    print(f"\nClass distribution — {split_name} ({total} samples):")
    for label_id in sorted(counts.keys()):
        count = counts[label_id]
        pct = count / total * 100
        print(f"  {label_names[label_id]:10s} ({label_id}): {count:5d} ({pct:.1f}%)")

    for label_id, count in counts.items():
        if count / total < 0.20:
            logger.warning(
                f"Class imbalance: '{label_names[label_id]}' "
                f"is only {count/total*100:.1f}% of {split_name}."
            )


# --------------------------------------------------------------------------- #
#  Model Export                                                                  #
# --------------------------------------------------------------------------- #

def push_to_hub(
    model,
    tokenizer,
    repo_id: str,
    save_method: str = "merged_16bit",
) -> None:
    """Merge LoRA adapters and push to HuggingFace Hub."""
    logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")

    if save_method == "merged_16bit":
        model.push_to_hub_merged(
            repo_id,
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    logger.info(f"Model pushed to: https://huggingface.co/{repo_id}")


def save_locally(
    model,
    tokenizer,
    save_dir: str,
    save_method: str = "merged_16bit",
) -> None:
    """Save merged model locally."""
    logger.info(f"Saving model locally to: {save_dir}")

    if save_method == "merged_16bit":
        model.save_pretrained_merged(save_dir, tokenizer, save_method="merged_16bit")
    else:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    logger.info(f"Model saved to {save_dir}")

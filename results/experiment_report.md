# Experiment Report — Cross-Lingual Sentiment Transfer via QLoRA Fine-Tuning

**Research Question:** Can a small generative language model (Qwen2.5 3B), fine-tuned only on English sentiment data via QLoRA, transfer to unseen languages, and how do its failure modes compare to encoder baselines and large frontier models?

---

## 1. Experiment Overview

| Item | Value |
|---|---|
| **Date training completed** | 2026-04-11 (evaluation aggregated at 2026-04-11T05:00:37 UTC, per JSON metadata) |
| **Dataset** | `cardiffnlp/tweet_sentiment_multilingual` |
| **Languages** | English, Hindi, Arabic, Spanish |
| **Test samples per language** | 870 (perfectly balanced 3-class: 290 neg / 290 neu / 290 pos) |
| **Primary metric** | F1 Macro with 95% bootstrap CI (1,000 resamples) |
| **Hardware** | Google Colab T4 GPU, 16 GB VRAM |
| **Total wall-clock time** | Training: ~3h total (Qwen QLoRA ~2h15m, XLM-R ~45m). Evaluation: ~5h across 2 days (Gemini rate limits). |

### HuggingFace Model Repositories

| Seed | Repository URL |
|---|---|
| 42 | `https://huggingface.co/Mrigank005/SLM-sentiment-crosslingual-seed-42` |
| 123 | `https://huggingface.co/Mrigank005/SLM-sentiment-crosslingual-seed-123` |
| 456 | `https://huggingface.co/Mrigank005/SLM-sentiment-crosslingual-seed-456` |

*(Template from `configs/training_config.yaml`: `Mrigank005/SLM-sentiment-crosslingual-seed-{seed}`)*

---

## 2. Training Configuration

### 2.1 Qwen2.5 3B — QLoRA Fine-Tuning

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct` |
| Max sequence length | 512 |
| Load in 4-bit | true |
| Dtype | null (auto-detect: float16 on T4) |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA target modules | `["q_proj", "v_proj"]` |
| LoRA dropout | 0.05 |
| LoRA bias | none |
| LoRA task type | CAUSAL_LM |
| Seeds | [42, 123, 456] |
| Num train epochs (max) | 10 |
| Per-device train batch size | 4 |
| Gradient accumulation steps | 4 |
| **Effective batch size** | **16** |
| Learning rate | 2.0e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| fp16 | true |
| Logging steps | 10 |
| Evaluation strategy | epoch |
| Save strategy | epoch |
| Load best model at end | true |
| Metric for best model | eval_loss |
| Greater is better | false |
| Save total limit | 3 |
| Early stopping patience | 2 |
| Report to | wandb |

### 2.2 XLM-R-base — Full English Fine-Tuning

| Parameter | Value |
|---|---|
| Base model | `xlm-roberta-base` |
| Num labels | 3 |
| Max sequence length | 128 |
| Learning rate | 2.0e-5 |
| Num train epochs (max) | 10 |
| Per-device train batch size | 32 |
| Gradient accumulation steps | 1 |
| **Effective batch size** | **32** |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Evaluation strategy | epoch |
| Save strategy | epoch |
| Load best model at end | true |
| Metric for best model | eval_loss |
| Early stopping patience | 2 |
| Seeds | [42, 123, 456] |

---

## 3. Training Results

### 3.1 Qwen2.5 3B QLoRA (3 Seeds)

| Seed | Best Epoch | Best Val Loss | Test F1 (seed eval) |
|---|---|---|---|
| 42 | 3 | 0.1907 | 0.7206 |
| 123 | 2 | 0.1955 | 0.7326 |
| 456 | 2 | 0.1889 | 0.7200 |

**Clarification:** The F1 values above are from seed-level test evaluation JSONs (`results/eval_scores/seed_*/qwen25_finetuned_english.json`), not validation-set F1 captured during training.

**Reviewer note:** In the captured QLoRA W&B run summaries, `eval/loss` was logged but `eval/f1_macro` was not, so training-time validation F1 is not directly recoverable from those summaries.

**Training curves:** 11 plots are present in `results/training_curves/`:
- `eval-loss.png` — Validation loss across training
- `eval-runtime.png` — Evaluation runtime per epoch
- `eval-smaples_per_second.png` (sic — typo in filename) — Evaluation samples/second
- `eval-steps_per_second.png` — Evaluation steps/second
- `train-epoch.png` — Training epoch progression
- `train-global_step.png` — Global step progression
- `train-grad_norm.png` — Gradient norm during training
- `train-learning_rate.png` — Learning rate schedule (cosine with warmup)
- `train-loss.png` — Training loss curve
- `val-loss-per-epoch.png` — Validation loss per epoch (simple line plot)
- `val_loss_per_epoch_last_seed.png` — Validation loss per epoch for the final seed

**Note:** Early stopping with patience=2 on eval_loss was configured. The model that loaded at training end was the checkpoint with the lowest validation loss.

### 3.2 XLM-R-base (3 Seeds)

| Seed | Best Epoch | Best Val Loss | Val F1 |
|---|---|---|---|
| 42 | 2 | 0.7470 | 0.644 |
| 123 | 3 |  0.7249 | 0.670 |
| 456 | 3 | 0.8007 | 0.656 |

**Note:** XLM-R was trained on the same English training split (1,839 samples) with early stopping patience=2 on eval_loss.

---

## 4. Dataset Statistics

*(Extracted from Notebook 01 output cells)*

### 4.1 Split Sizes

| Language | Train N | Validation N | Test N |
|---|---|---|---|
| English | 1,839 | 324 | 870 |
| Hindi | 1,839 | 324 | 870 |
| Arabic | 1,839 | 324 | 870 |
| Spanish | 1,839 | 324 | 870 |

### 4.2 Class Distribution (Test Set)

All four languages have **perfectly balanced** test sets:
- Negative: 290 (33.3%)
- Neutral: 290 (33.3%)
- Positive: 290 (33.3%)

### 4.3 Text Length Statistics (Test Set)

| Language | Avg Chars | Max Chars | Avg Words | Max Words |
|---|---|---|---|---|
| English | 92.8 | 142 | 15.0 | 32 |
| Hindi | 71.5 | 408 | 14.2 | 86 |
| Arabic | 85.8 | 140 | 14.1 | 29 |
| Spanish | 84.5 | 138 | 15.8 | 33 |

### 4.4 Token Fertility (Qwen2.5 Tokenizer — Test Set)

| Language | Avg Tokens | Max Tokens | Fertility (tokens/word) |
|---|---|---|---|
| English | 24.6 | 58 | 1.633 |
| Hindi | 24.2 | 125 | 1.703 |
| Arabic | 36.5 | 104 | 2.592 |
| Spanish | 25.0 | 56 | 1.582 |

**Note on Hindi:** The Hindi subset is **100% Romanized** (Latin script) in the test set (0 Devanagari out of 870 samples) and 99.9% Romanized in the training set (1 Devanagari out of 1,839 samples). Consequently, the fertility of 1.703 reflects Latin-script tokenization, not Devanagari. This should be noted as a limitation: Hindi does not test true non-Latin script transfer. XLM-R fertility is excluded from these calculations since XLM-R uses its own SentencePiece tokenizer.

---

## 5. Main Results — 6×4 F1 Macro Table

> **CI notation:** For seed-dependent models (Qwen QLoRA, XLM-R), values are reported as **mean ± sample std** (n−1 denominator) across the 3 seeds. For seed-agnostic models (majority, Qwen base, Llama, Gemini), values are reported as **score [lower, upper]** from bootstrapping the single-run n=870 prediction cache.

| Model | English | Spanish | Hindi | Arabic | Avg Latency (s) |
|---|---|---|---|---|---|
| Majority Baseline | 0.1667 [0.1538, 0.1785] | 0.1667 [0.1538, 0.1785] | 0.1667 [0.1538, 0.1785] | 0.1667 [0.1538, 0.1785] | N/A |
| Qwen2.5 3B (zero-shot) | 0.6752 [0.6443, 0.7068] | 0.6253 [0.5928, 0.6566] | 0.4851 [0.4503, 0.5205] | 0.4880 [0.4530, 0.5234] | 0.057 |
| **Qwen2.5 3B QLoRA (ours)** | **0.7244 ± 0.0071** | **0.6759 ± 0.0169** | **0.5652 ± 0.0027** | **0.5456 ± 0.0177** | 0.058 |
| Llama 3.3 70B (zero-shot) | 0.7137 [0.6832, 0.7411] | 0.6680 [0.6373, 0.6972] | 0.6484 [0.6172, 0.6809] | 0.7334 [0.7043, 0.7622] | 0.702 |
| Gemini 2.5 Pro (zero-shot) | 0.6950 [0.6654, 0.7236] | 0.7313 [0.7032, 0.7600] | 0.6798 [0.6485, 0.7123] | 0.7015 [0.6723, 0.7322] | 2.751 |
| XLM-R-base (EN fine-tuned) | 0.6710 ± 0.0050 | 0.5750 ± 0.0446 | 0.5299 ± 0.0049 | 0.6147 ± 0.0192 | 0.003 |

**Latency notes:**
- Qwen models (base and QLoRA): ~0.05–0.06 s/sample (local T4 inference)
- Llama 70B: ~0.70 s/sample (Groq API)
- Gemini 2.5 Pro: ~2.75 s/sample (Google AI Studio API)
- XLM-R: ~0.003 s/sample (local encoder inference)
- Majority baseline: no inference required

---

## 6. Secondary Metrics Table

### 6.1 Accuracy

| Model | English | Spanish | Hindi | Arabic |
|---|---|---|---|---|
| Majority Baseline | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| Qwen2.5 3B (zero-shot) | 0.6701 | 0.6207 | 0.5023 | 0.5126 |
| Qwen2.5 3B QLoRA | 0.7256 ± 0.0058 | 0.6809 ± 0.0121 | 0.5686 ± 0.0044 | 0.5575 ± 0.0127 |
| Llama 3.3 70B | 0.7195 | 0.6805 | 0.6483 | 0.7310 |
| Gemini 2.5 Pro | 0.6931 | 0.7322 | 0.6770 | 0.6977 |
| XLM-R-base | 0.6812 ± 0.0041 | 0.6073 ± 0.0252 | 0.5352 ± 0.0059 | 0.6169 ± 0.0197 |

### 6.2 Per-Class F1 Scores

*(From aggregated JSONs; mean ± std for seed-dependent models, mean for seed-agnostic models)*

| Model | Language | Neg F1 | Neu F1 | Pos F1 | F1 Macro |
|---|---|---|---|---|---|
| Majority Baseline | English | 0.5000 | 0.0000 | 0.0000 | 0.1667 |
| Majority Baseline | Spanish | 0.5000 | 0.0000 | 0.0000 | 0.1667 |
| Majority Baseline | Hindi | 0.5000 | 0.0000 | 0.0000 | 0.1667 |
| Majority Baseline | Arabic | 0.5000 | 0.0000 | 0.0000 | 0.1667 |
| Qwen2.5 3B Base | English | 0.7526 | 0.5884 | 0.6845 | 0.6752 |
| Qwen2.5 3B Base | Spanish | 0.6285 | 0.5592 | 0.6882 | 0.6253 |
| Qwen2.5 3B Base | Hindi | 0.4418 | 0.5487 | 0.4648 | 0.4851 |
| Qwen2.5 3B Base | Arabic | 0.4646 | 0.5645 | 0.4348 | 0.4880 |
| Qwen2.5 3B QLoRA | English | 0.7782 ± 0.0048 | 0.6061 ± 0.0274 | 0.7890 ± 0.0181 | 0.7244 ± 0.0071 |
| Qwen2.5 3B QLoRA | Spanish | 0.7209 ± 0.0164 | 0.5587 ± 0.0370 | 0.7481 ± 0.0066 | 0.6759 ± 0.0169 |
| Qwen2.5 3B QLoRA | Hindi | 0.5108 ± 0.0274 | 0.5727 ± 0.0134 | 0.6120 ± 0.0154 | 0.5652 ± 0.0027 |
| Qwen2.5 3B QLoRA | Arabic | 0.5068 ± 0.0527 | 0.5861 ± 0.0026 | 0.5437 ± 0.0309 | 0.5456 ± 0.0177 |
| Llama 3.3 70B | English | 0.7773 | 0.5833 | 0.7807 | 0.7137 |
| Llama 3.3 70B | Spanish | 0.7630 | 0.4909 | 0.7500 | 0.6680 |
| Llama 3.3 70B | Hindi | 0.6767 | 0.5675 | 0.7009 | 0.6484 |
| Llama 3.3 70B | Arabic | 0.7894 | 0.6761 | 0.7347 | 0.7334 |
| Gemini 2.5 Pro | English | 0.7700 | 0.6151 | 0.7000 | 0.6950 |
| Gemini 2.5 Pro | Spanish | 0.7762 | 0.6351 | 0.7826 | 0.7313 |
| Gemini 2.5 Pro | Hindi | 0.6975 | 0.6405 | 0.7016 | 0.6798 |
| Gemini 2.5 Pro | Arabic | 0.7543 | 0.6667 | 0.6837 | 0.7015 |
| XLM-R-base | English | 0.7394 ± 0.0088 | 0.5320 ± 0.0226 | 0.7416 ± 0.0175 | 0.6710 ± 0.0050 |
| XLM-R-base | Spanish | 0.6797 ± 0.0069 | 0.3615 ± 0.1065 | 0.6837 ± 0.0210 | 0.5750 ± 0.0446 |
| XLM-R-base | Hindi | 0.5308 ± 0.0216 | 0.4935 ± 0.0446 | 0.5654 ± 0.0666 | 0.5299 ± 0.0049 |
| XLM-R-base | Arabic | 0.6555 ± 0.0192 | 0.5514 ± 0.0220 | 0.6372 ± 0.0806 | 0.6147 ± 0.0192 |

---

## 7. Fine-Tuning Ablation (Base vs QLoRA vs XLM-R)

| Language | Qwen Base F1 | Qwen QLoRA F1 | XLM-R F1 | Winner |
|---|---|---|---|---|
| English | 0.6752 | **0.7244** ± 0.0071 | 0.6710 ± 0.0050 | **Qwen QLoRA** |
| Spanish | 0.6253 | **0.6759** ± 0.0169 | 0.5750 ± 0.0446 | **Qwen QLoRA** |
| Hindi | 0.4851 | **0.5652** ± 0.0027 | 0.5299 ± 0.0049 | **Qwen QLoRA** |
| Arabic | 0.4880 | 0.5456 ± 0.0177 | **0.6147** ± 0.0192 | **XLM-R** |

### Analysis

**English-only QLoRA tuning transfers positively to every language** relative to the untuned base model:
- English: +0.0492 (7.3%)
- Spanish: +0.0506 (8.1%)
- Hindi: +0.0801 (16.5%)
- Arabic: +0.0576 (11.8%)

**QLoRA vs. XLM-R comparison:**
- Qwen QLoRA wins on **3 of 4 languages** (English, Spanish, Hindi).
- XLM-R wins **Arabic** (0.6147 vs. 0.5456, +0.0691; 12.7% relative) under the shared English instruction. With an Arabic instruction QLoRA reaches 0.665 (Section 8.2).
- **Spanish** shows the largest QLoRA advantage: 0.6759 vs. 0.5750 (+0.1009). XLM-R's Spanish seed SD (0.0446) indicates instability, with neutral-class F1 at 0.3615 ± 0.1065 and only 14.5% of Spanish inputs predicted neutral.

---

## 8. Cross-Lingual Transfer Analysis

### 8.1 Neutral collapse and fragmentation

| Language | QLoRA F1 | Δ from English | Neutral predicted (%) | Parity vs. English translation |
|---|---|---|---|---|
| English | 0.7244 | — | 32.0 | 1 (ref.) |
| Spanish | 0.6759 | −0.0485 | 29.1 | 1.22 [1.21, 1.24] |
| Hindi | 0.5652 | −0.1592 | 55.2 | 1.26 [1.24, 1.28] |
| Arabic | 0.5456 | −0.1788 | 69.3 | 1.41 [1.39, 1.43] |

Test sets are balanced, so 33.3% neutral is the chance rate. English tuning recalibrates the model on English and Spanish but leaves it collapsed onto neutral on Hindi and Arabic. Parity (Qwen2.5 tokens of a tweet divided by tokens of its English translation, `src/parity.py`, translations in `results/translations/`) replaces tokens per word, which overstates Arabic because Arabic tweets have fewer, longer words. Romanized Hindi is fragmented only slightly more than Spanish yet collapses, so fragmentation does not explain the Hindi collapse.

### 8.2 Bias or loss of class separation? (label probabilities and target-language instructions)

Notebook `notebooks/06_label_probs_target_prompts.ipynb` re-ran all Qwen models, recording first-step label probabilities, and repeated the non-English evaluation with the instruction in the target language (label words kept in English). Outputs: `results/label_probs/` and `results/target_language_prompts/`, analysis: `src/label_prob_analysis.py`. Fine-tuned model, mean of 3 seeds:

| Language | AUC neg | AUC neu | AUC pos | F1, balanced decision rule | F1 / neutral %, target-language instruction |
|---|---|---|---|---|---|
| English | 0.929 | 0.797 | 0.948 | 0.738 | — |
| Spanish | 0.899 | 0.774 | 0.901 | 0.687 | 0.659 / 20.6 |
| Hindi | 0.797 | 0.718 | 0.810 | 0.599 | 0.577 / 34.0 |
| Arabic | 0.885 | 0.778 | 0.871 | 0.669 | 0.665 / 26.1 |

The balanced decision rule adds one offset per class until predictions are balanced (uses the known class balance, no item labels). On Arabic the model separates the classes almost as well as on Spanish, and either intervention recovers most of the lost F1: the collapse is a bias towards neutral. On Hindi the AUCs are lower and neither intervention helps much: class separation itself is weaker.

---

## 9. Scale vs. Specialization Analysis

| Model | Params | English | Spanish | Hindi | Arabic | Avg F1 |
|---|---|---|---|---|---|---|
| Qwen QLoRA (ours) | 3B | **0.7244** | 0.6759 | 0.5652 | 0.5456 | 0.6278 |
| Llama 3.3 70B | 70B | 0.7137 | 0.6680 | 0.6484 | **0.7334** | 0.6909 |
| Gemini 2.5 Pro | Unknown | 0.6950 | **0.7313** | **0.6798** | 0.7015 | **0.7019** |

### Key observations:

1. **English:** QLoRA (0.7244) is above Llama 70B (0.7137) and Gemini (0.6950); the Llama margin is within its CI, so this is parity-or-better rather than a decisive win.

2. **Arabic:** Llama 70B reaches 0.7334 and Gemini 0.7015, against 0.5456 for QLoRA under the English instruction. Neither shows the neutral collapse (Llama 39.8% neutral, Gemini 52.2%).

3. **Hindi (Romanized):** Llama (0.6484) and Gemini (0.6798) outperform QLoRA (0.5652) by 0.08–0.11, more than the whole English–Spanish drop.

4. **Spanish:** QLoRA (0.6759) is level with Llama (0.6680) and below Gemini (0.7313).

5. **Cost-performance tradeoff:** QLoRA reaches 89% of Gemini's average F1 at $0 inference cost.

---
## 10. Parse Failure Analysis

### API-based generative models

The `parse_failures.json` files in the API cache contain information about instances where the generative model's output could not be parsed into one of the three valid sentiment labels.

From the current artifacts:

| Model | Language | Parse Failure Count | Parse Failure Rate | Flagged |
|---|---|---|---|---|
| Qwen2.5 Base | English | 0 | 0.0% | No |
| Qwen2.5 Base | Hindi | 0 | 0.0% | No |
| Qwen2.5 Base | Arabic | 0 | 0.0% | No |
| Qwen2.5 Base | Spanish | 0 | 0.0% | No |
| Llama 3.3 70B | English | 0 | 0.0% | No |
| Llama 3.3 70B | Hindi | 0 | 0.0% | No |
| Llama 3.3 70B | Arabic | 0 | 0.0% | No |
| Llama 3.3 70B | Spanish | 0 | 0.0% | No |
| Gemini 2.5 Pro | English | 0 | 0.0% | No |
| Gemini 2.5 Pro | Hindi | 0 | 0.0% | No |
| Gemini 2.5 Pro | Arabic | 0 | 0.0% | No |
| Gemini 2.5 Pro | Spanish | 0 | 0.0% | No |

**Note:** API parse-failure artifacts are stored in `results/api_cache/*/*/parse_failures.json` and currently show zero failures for all Gemini and Llama language pairs. Because there are no recorded parse-failure instances in the current artifacts, parse-failure subtype analysis (for example verbose outputs or language mismatch) is not applicable for this run.

---

## 11. Cost Analysis

| Model | Params | Cost per 1K Samples | Hosting |
|---|---|---|---|
| Majority Baseline | N/A | $0 | N/A |
| Qwen2.5 3B (base) | 3B | $0 | Self-hosted (Colab T4) |
| Qwen2.5 3B QLoRA | 3B | $0 | Self-hosted (Colab T4) |
| Llama 3.3 70B | 70B | ~$0.03–0.06 | Groq API (free tier available) |
| Gemini 2.5 Pro | Undisclosed | ~$0 (free tier) / variable | Google AI Studio |
| XLM-R-base | 278M | $0 | Self-hosted (Colab T4) |

**Notes:**
- Groq API pricing: `llama-3.3-70b-versatile` is available on free tier with rate limits; paid tier pricing varies.
- Google AI Studio: Gemini 2.5 Pro has a free tier for development; production pricing is per-token.
- Self-hosted models on Colab T4 incur only compute costs (~$0 on free tier, ~$0.11/hr on Pro).

---

## 12. Confusion Matrix Summary

All **24 confusion matrix PDFs** are present in `results/confusion_matrices/`:

| Model | English | Spanish | Hindi | Arabic |
|---|---|---|---|---|
| Majority Baseline | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5 Base | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5 QLoRA | ✅ | ✅ | ✅ | ✅ |
| Llama 3.3 70B | ✅ | ✅ | ✅ | ✅ |
| Gemini 2.5 Pro | ✅ | ✅ | ✅ | ✅ |
| XLM-R-base | ✅ | ✅ | ✅ | ✅ |

### Structural Failure Mode Descriptions

1. **Arabic neutral-class bias (Qwen models):**
   - Qwen Base Arabic: Neutral F1 = 0.5645, while negative F1 = 0.4646 and positive F1 = 0.4348. The model over-predicts neutral, achieving high neutral recall (0.9276 on the n=870 evaluation set) at the expense of negative and positive recall (0.3172 and 0.2931 respectively).
   - Qwen QLoRA Arabic: Neutral F1 improves to 0.5861 ± 0.0026, but 69.3% of inputs are still predicted neutral and negative F1 (0.5068 ± 0.0527) and positive F1 (0.5437 ± 0.0309) remain depressed. The label-probability analysis (Section 8.2) shows this is a bias: the class information is still present.

2. **XLM-R's severe underperformance on Spanish neutral class:**
   - XLM-R Spanish neutral F1 = **0.3615 ± 0.1065** — dramatically lower than any other class for any model. Negative and positive F1 are reasonable, but the neutral class nearly collapses. The very high seed SD indicates instability.
   - A plausible explanation is distribution mismatch: XLM-R was fine-tuned on English neutral tweets, while Spanish neutral expressions may differ lexically and syntactically.

3. **Llama 70B Arabic:** Strong performance across all classes — Neg 0.7894, Neu 0.6761, Pos 0.7347, with no neutral collapse.

4. **Gemini 2.5 Pro:** Generally balanced across classes and languages, with its weakest class typically being neutral.

---

## 13. Error Analysis

`results/error_analysis/{english,spanish,hindi,arabic}_errors.csv` list the seed-42 QLoRA misclassifications
(columns `original_index`, `text`, `true_label`, `pred_label`, `failure_category`).

Annotated samples: `annotation_sample.csv` (10 Spanish, 10 Romanized Hindi, 10 Arabic, with English translations)
and `hindi_annotation_extension.csv` (40 further Hindi items, with notes; guide in `HINDI_ANNOTATION_GUIDE.md`).
Both files carry `seeds_wrong_of_3`, `majority_pred` and `retained`; the analysis uses the `retained` items, which at
least two of the three QLoRA seeds misclassify (8 Spanish, 8 Arabic, 46 Hindi).

### Failure categories (retained items)

| Category | Spanish | Hindi | Arabic |
|---|---:|---:|---:|
| tokenization_artifact | 1 | 0 | 4 |
| ambiguous_neutral | 3 | 12 | 0 |
| cultural_idiom | 2 | 4 | 2 |
| context_dependent | 1 | 14 | 2 |
| code_switching | 0 | 14 | 0 |
| sarcasm_irony | 1 | 2 | 0 |
| **Total** | 8 | 46 | 8 |

Hindi labels are by a native speaker; Spanish and Arabic labels were made with proficient non-native speakers.
Of the 38 retained items in the Hindi extension, 4 are entirely English and 14 have a debatable gold label.
Under the Arabic instruction 5 of the 8 Arabic items are classified correctly by at least two seeds (6 of 8 under the
balanced decision rule, including 3 of the 4 tokenization items); for Hindi the figures are 17 and 12 of 46. The
Arabic "tokenization artifact" label therefore describes how those errors look rather than what causes them.

---

## 14. Key Findings

1. **QLoRA matches or beats the frontier models on English** (0.7244 vs. Llama 0.7137, Gemini 0.6950).

2. **QLoRA beats XLM-R on 3 of 4 languages** (0.7244, 0.6759, 0.5652 vs. 0.6710, 0.5750, 0.5299); XLM-R wins Arabic (0.6147 vs. 0.5456) under the English instruction.

3. **English-only tuning transfers positively everywhere**, +0.0492 to +0.0801 macro-F1 over the base model.

4. **Predicted-class rates separate success from failure:** neutral share moves to 32.0% (EN) and 29.1% (ES) but stays at 55.2% (HI) and 69.3% (AR), against a 33.3% chance rate.

5. **Two failure profiles:** on Arabic the collapse is a bias over intact class information (AUC close to Spanish; balanced rule or Arabic instruction recovers F1 to 0.669 / 0.665); on Romanized Hindi class separation itself is weaker (balanced rule 0.599, Hindi instruction 0.577). Fragmentation explains neither.

6. **Fine-tuning barely improves the neutral class** (+0.016 on average vs. +0.057 negative and +0.105 positive).

7. **Gemini 2.5 Pro has the highest average F1** (0.7019), followed by Llama 70B (0.6909) and QLoRA (0.6278).

8. **Neutral is the weakest class in 16 of 20 non-trivial cells**; the four exceptions are the Qwen models on Hindi and Arabic, where the collapse inflates neutral recall.

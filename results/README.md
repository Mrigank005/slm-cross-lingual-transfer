# Results

All artifacts behind the numbers in the paper. Every file here is produced by the code in `src/` and the
notebooks in `notebooks/`.

| Folder / file | Contents |
|---|---|
| `api_cache/<model>/[seed_<s>/]<language>/predictions.json` | Per-item predictions (text, gold label, predicted label, raw model output, latency) for every model × language, and per seed for QLoRA and XLM-R. `parse_failures.json` files list unparseable outputs (none occurred). |
| `eval_scores/seed_<s>/` | Per-seed metrics (macro-F1, bootstrap CI, per-class P/R/F1, latency). Seed-agnostic models are stored identically under each seed for the aggregator. |
| `eval_scores/aggregated/` | Mean and sample standard deviation over seeds 42/123/456 (single-run models: point estimate and 95% bootstrap CI). |
| `eval_scores/results_table.{md,tex}` | Main macro-F1 table (paper Table 1). |
| `eval_scores/subsample_indices.json` | Indices of the 870 balanced test items per language. |
| `eval_scores/hindi_script_composition.json` | Script composition of the Hindi splits (100% Latin-script test set). |
| `label_probs/` | First-token label log-probabilities for the base model and the three QLoRA seeds with the English instruction (notebook 06); `summary.json` holds per-class AUC and balanced-decision-rule macro-F1 (`python -m src.label_prob_analysis`). |
| `target_language_prompts/` | Predictions with the instruction in Spanish, Romanized Hindi or Arabic (label words in English); `summary.json` compares with the English instruction. |
| `translations/` | English machine translations of the non-English test items (`gpt-oss-120b` via Groq) and the tokenizer-parity summary (`python -m src.parity`). |
| `error_analysis/<language>_errors.csv` | Seed-42 QLoRA misclassifications. |
| `error_analysis/annotation_sample.csv` | Manually annotated errors (10 per non-English language) with English translations and failure categories. |
| `error_analysis/hindi_annotation_extension.csv` | 40 further annotated Romanized Hindi errors with notes; `HINDI_ANNOTATION_GUIDE.md` is the guide used. In both annotation files, `seeds_wrong_of_3`, `majority_pred` and `retained` refer to the released predictions; the paper's error taxonomy uses the `retained` rows (misclassified by at least two of the three seeds). |
| `figures/` | Figures used in the paper (`f1_grouped_bar.pdf`, `confusion_matrices_grid.pdf`) and the cost–performance plot. |
| `confusion_matrices/` | One confusion matrix per model × language. |
| `training_curves/` | Training diagnostics exported from Weights & Biases. |
| `experiment_report.md` | Narrative report with all tables. |

Label encoding throughout: 0 = negative, 1 = neutral, 2 = positive.

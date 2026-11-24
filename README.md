# Two Failure Profiles in Zero-Shot Cross-Lingual Transfer of a Small Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements-local.txt)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-checkpoints-orange.svg)](https://huggingface.co/Mrigank005)

Code, predictions and annotations for the paper

> **Two Failure Profiles in Zero-Shot Cross-Lingual Transfer of a Small Language Model**
> Mrigank Singh (UPES Dehradun).
> *2nd Workshop on BHASHA: Benchmarks, Harmonization, Annotation, and Standardization for Human-Centric AI in
> Indian Languages*, co-located with ICON 2026, Guwahati, India.

The paper (PDF and LaTeX source) is in [`paper/`](paper/).

## Summary

We fine-tune Qwen2.5-3B-Instruct with QLoRA on **English-only** tweet sentiment and evaluate zero-shot transfer
to Spanish, Romanized Hindi and Arabic (`cardiffnlp/tweet_sentiment_multilingual`, 870 balanced test items per
language), against an untuned base model, a fully fine-tuned XLM-R encoder, Llama 3.3 70B and Gemini 2.5 Pro.

- English-only tuning transfers positively to every language (+0.049 to +0.080 macro-F1), and on English the 3B
  model matches or beats both frontier models.
- **Predicted-class rates on balanced data** separate success from failure: tuning recalibrates the model on
  English and Spanish (32% and 29% neutral predictions, chance 33%) but leaves it collapsed onto neutral on
  Romanized Hindi (55%) and Arabic (69%).
- The two collapses have **different profiles**. Label probabilities show that on Arabic the model still
  separates the classes almost as well as on Spanish; balancing its decisions, or giving the instruction in
  Arabic, recovers most of the lost macro-F1 (0.546 → 0.669 / 0.665). On Romanized Hindi, which the tokenizer
  fragments only slightly more than Spanish, class separation itself is weaker and neither intervention helps
  much.

| Macro-F1 | English | Spanish | Hindi (Rom.) | Arabic |
|---|---|---|---|---|
| Qwen2.5 3B (zero-shot) | 0.6752 | 0.6253 | 0.4851 | 0.4880 |
| **Qwen2.5 3B QLoRA (ours)**, mean ± SD of 3 seeds | **0.7244** ± 0.0071 | 0.6759 ± 0.0169 | 0.5652 ± 0.0027 | 0.5456 ± 0.0177 |
| Llama 3.3 70B (zero-shot) | 0.7137 | 0.6680 | 0.6484 | 0.7334 |
| Gemini 2.5 Pro (zero-shot) | 0.6950 | 0.7313 | 0.6798 | 0.7015 |
| XLM-R-base (EN fine-tuned), mean ± SD of 3 seeds | 0.6710 ± 0.0050 | 0.5750 ± 0.0446 | 0.5299 ± 0.0049 | 0.6147 ± 0.0192 |

Full tables with confidence intervals: [`results/eval_scores/results_table.md`](results/eval_scores/results_table.md)
and [`results/experiment_report.md`](results/experiment_report.md).

## Repository layout

```text
.
├── paper/                  LaTeX source, figures and PDF of the paper
├── configs/                training configuration (QLoRA and XLM-R)
├── notebooks/              pipeline notebooks 01–06 (Colab T4 for training/inference)
├── src/                    training, inference, evaluation, analysis and plotting code
├── results/
│   ├── api_cache/          per-item predictions of every model × language (× seed)
│   ├── eval_scores/        per-seed and aggregated metrics, results tables, test-subset indices
│   ├── label_probs/        first-token label probabilities (English instruction)
│   ├── target_language_prompts/  predictions with the instruction in the target language
│   ├── translations/       machine translations used for the tokenizer-parity measure
│   ├── error_analysis/     misclassification lists and the manual error annotations
│   ├── figures/            figures used in the paper
│   ├── confusion_matrices/ per-cell confusion matrices
│   └── training_curves/    training diagnostics
├── requirements-local.txt  local environment (analysis, API inference, plots)
├── requirements-colab.txt  Colab environment (training and local-model inference)
└── .env.example            template for API keys
```

Folder guides: [`notebooks/`](notebooks/README.md), [`src/`](src/README.md), [`results/`](results/README.md),
[`paper/`](paper/README.md).

## Models

| Model | Hugging Face Hub |
|---|---|
| Qwen2.5-3B QLoRA (merged), seeds 42 / 123 / 456 | [`Mrigank005/SLM-sentiment-crosslingual-seed-{42,123,456}`](https://huggingface.co/Mrigank005) |
| XLM-R-base, fully fine-tuned, seeds 42 / 123 / 456 | [`Mrigank005/xlm-roberta-sentiment-crosslingual-seed-{42,123,456}`](https://huggingface.co/Mrigank005) |

## Reproducing the results

All numbers in the paper can be recomputed from the released predictions without a GPU.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-local.txt
```

| Paper item | Command / source |
|---|---|
| Macro-F1, per-class F1, neutral shares; Figures 1–2 | `notebooks/04_evaluation.ipynb`, section 2 (runs locally from the caches: seed aggregation, tables, figures) |
| Label probabilities and target-language instructions (§5.1, Appendix G) | `python -m src.label_prob_analysis` |
| Tokenizer parity (Appendix H) | `python -m src.parity` (translations: `python -m src.translate_parity`, needs a Groq key) |
| Error taxonomy (§5.7, Appendix D) | `results/error_analysis/annotation_sample.csv`, `hindi_annotation_extension.csv` (rows with `retained = True`) |

Re-running inference: notebooks `02`/`03` train the models, `04` evaluates all configurations (local models on a
Colab T4; Llama and Gemini through their APIs, keys in `.env`), and `06` records label probabilities and runs the
target-language instructions. The QLoRA seed-42 predictions in `results/api_cache/` were written from the full
pass of notebook 06 with `python -m src.replace_seed42`. Note that the Llama model used
(`llama-3.3-70b-versatile` on Groq) has since been withdrawn by the provider, so its cached predictions are the
only way to reproduce that row exactly.

## Data

The tweets come from [`cardiffnlp/tweet_sentiment_multilingual`](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)
(Barbieri et al., 2022) and are subject to that dataset's terms. The cached predictions repeat the test-set texts
for convenience; pseudonymisation in the source data is partial (see the paper's Ethical Considerations).

## Citation

```bibtex
@inproceedings{singh2026twofailure,
  title     = {Two Failure Profiles in Zero-Shot Cross-Lingual Transfer of a Small Language Model},
  author    = {Singh, Mrigank},
  booktitle = {Proceedings of the 2nd Workshop on Benchmarks, Harmonization, Annotation, and Standardization
               for Human-Centric AI in Indian Languages (BHASHA 2026)},
  year      = {2026},
  note      = {Co-located with ICON 2026}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). Model checkpoints follow the licences of their
base models (Qwen2.5, XLM-R).

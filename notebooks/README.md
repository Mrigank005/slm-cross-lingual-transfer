# Notebooks

Run in order. Training and local-model inference need a GPU (the paper used a Colab T4); everything that only
reads the released caches runs on CPU.

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset inspection, class balance, text length and token statistics, balanced test-subset indices. |
| `02_finetune.ipynb` | QLoRA fine-tuning of Qwen2.5-3B-Instruct on English (seeds 42/123/456), pushed to the Hub. |
| `03_finetune_xlmr.ipynb` | Full fine-tuning of XLM-R-base on English (seeds 42/123/456), pushed to the Hub. |
| `04_evaluation.ipynb` | Evaluation of all six configurations on the four test sets. Section 1 (Colab) runs the local models; section 2 (local) runs the API models, the majority baseline, seed aggregation, tables and figures. |
| `05_inference_demo.ipynb` | Interactive inference with a released checkpoint. |
| `06_label_probs_target_prompts.ipynb` | Colab: first-token label probabilities and target-language instructions for the Qwen models; analysed with `python -m src.label_prob_analysis`. |

Inference is cached per item (`results/api_cache/`), so interrupted runs resume and re-running a notebook reuses
existing predictions. Outputs have been cleared; the resulting artifacts are in `results/`.

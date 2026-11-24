# Source Code Guide

This folder contains modular Python components for data handling, training, inference, evaluation, analysis, and demo serving.

## Module Overview

- dataset.py: Dataset loading, split handling, and sampling utilities.
- train.py: QLoRA training pipeline for Qwen2.5 3B.
- train_xlmr.py: Baseline training pipeline for XLM-R.
- evaluate.py: Metrics, confidence intervals, and multi-seed aggregation.
- infer_local.py: Local model inference utilities.
- infer_groq.py: Groq API inference runner and cache integration.
- infer_gemini.py: Gemini API inference runner and cache integration.
- infer_xlmr.py: XLM-R inference and evaluation utilities.
- fertility.py: Tokens-per-word statistics under the Qwen2.5 tokenizer.
- cost_analysis.py: Runtime and cost-performance analysis.
- visualize.py: Plot and table generation from result artifacts.
- prompts.py: Prompt templates and label extraction helpers.
- app.py: Gradio application entry point for interactive demos.
- translate_parity.py: Machine translation of the non-English test items to English (Groq), cached in `results/translations/`.
- parity.py: Tokenizer parity (tokens of a tweet / tokens of its English translation) with bootstrap CIs.
- label_prob_analysis.py: Per-class AUC, balanced decision rule and target-language-instruction comparison from the notebook-06 outputs.
- replace_seed42.py: Writes the QLoRA seed-42 predictions from the notebook-06 full pass and recomputes the seed-42 metrics and aggregates.

## Design Notes

- Inference scripts use per-sample atomic caching to make reruns resumable and cost-aware.
- Evaluation code separates seed-dependent and seed-agnostic CI logic for fair reporting.
- Visualization modules are artifact-driven and read from `results/eval_scores`.

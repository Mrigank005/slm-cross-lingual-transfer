"""
app.py — Gradio demo for cross-lingual sentiment analysis.

Loads the fine-tuned Qwen2.5 3B model from HuggingFace Hub and provides
a web interface for sentiment prediction in English, Hindi, Arabic, and Spanish.
"""

import logging

import gradio as gr
import torch

from src.dataset import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, LABEL_MAP

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Configuration                                                                #
# --------------------------------------------------------------------------- #

MODEL_REPO_ID = "Mrigank005/SLM-sentiment-crosslingual"

EXAMPLES = {
    "English": [
        "I absolutely love this product, it works perfectly!",
        "The service was okay, nothing special.",
        "This is the worst experience I've ever had.",
    ],
    "Hindi": [
        "Yeh product bahut accha hai, mujhe bahut pasand aaya!",
        "Service theek-thaak thi, kuch khaas nahi.",
        "Yeh sabse bura experience tha.",
    ],
    "Arabic": [
        "هذا المنتج رائع جداً، أنا أحبه!",
        "الخدمة كانت عادية، لا شيء مميز.",
        "هذه أسوأ تجربة مررت بها.",
    ],
    "Spanish": [
        "¡Me encanta este producto, funciona perfectamente!",
        "El servicio estuvo bien, nada especial.",
        "Esta es la peor experiencia que he tenido.",
    ],
}

SENTIMENT_COLORS = {
    "positive": "#4CAF50",
    "neutral": "#FF9800",
    "negative": "#F44336",
}

# --------------------------------------------------------------------------- #
#  Model Loading                                                                #
# --------------------------------------------------------------------------- #

_model = None
_tokenizer = None


def _load_model():
    """Load model on first inference call."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    try:
        from unsloth import FastLanguageModel

        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_REPO_ID,
            max_seq_length=512,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(_model)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        _model.eval()

    logger.info(f"Model loaded: {MODEL_REPO_ID}")
    return _model, _tokenizer


# --------------------------------------------------------------------------- #
#  Prediction                                                                   #
# --------------------------------------------------------------------------- #

def predict_sentiment(text: str) -> tuple[str, str]:
    """Run sentiment prediction on input text.

    Args:
        text: Input text in any supported language.

    Returns:
        Tuple of (label, markdown_detail) for Gradio outputs.
    """
    if not text.strip():
        return "—", "Please enter some text."

    model, tokenizer = _load_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
    ]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_length = len(inputs.input_ids[0])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_length:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Normalize to label
    raw_lower = raw_output.lower().strip()
    label_map = {"positive": "positive", "neutral": "neutral", "negative": "negative",
                 "pos": "positive", "neu": "neutral", "neg": "negative"}

    label = "neutral"  # default
    for key, val in label_map.items():
        if key in raw_lower:
            label = val
            break

    color = SENTIMENT_COLORS.get(label, "#999")

    return label, f"**Sentiment: {label.upper()}**\n\nRaw model output: `{raw_output}`"


# --------------------------------------------------------------------------- #
#  Gradio Interface                                                             #
# --------------------------------------------------------------------------- #

def build_demo() -> gr.Blocks:
    """Build Gradio demo interface."""

    with gr.Blocks(
        title="Cross-Lingual Sentiment Analysis",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🌍 Cross-Lingual Sentiment Analysis
            ### QLoRA Fine-Tuned Qwen2.5 3B — Trained on English, Works Across Languages

            This model was fine-tuned on **English sentiment data only** using QLoRA
            and can classify sentiment in **English, Hindi, Arabic, and Spanish**.

            Enter any text below to get a sentiment prediction.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text",
                    placeholder="Type or paste text in any supported language...",
                    lines=3,
                )
                submit_btn = gr.Button("Analyze Sentiment", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Label(label="Predicted Sentiment")
                detail_output = gr.Markdown(label="Details")

        gr.Markdown("### 📝 Example Inputs")

        with gr.Row():
            for lang, examples in EXAMPLES.items():
                with gr.Column():
                    gr.Markdown(f"**{lang}**")
                    for ex in examples:
                        gr.Button(ex, size="sm").click(
                            fn=lambda x=ex: x,
                            outputs=text_input,
                        )

        submit_btn.click(
            fn=predict_sentiment,
            inputs=text_input,
            outputs=[label_output, detail_output],
        )

        text_input.submit(
            fn=predict_sentiment,
            inputs=text_input,
            outputs=[label_output, detail_output],
        )

        gr.Markdown(
            """
            ---
            **Model**: Qwen2.5-3B-Instruct + QLoRA (r=16, α=16) •
            **Training**: English sentiment data only (1,838 samples) •
            **Dataset**: cardiffnlp/tweet_sentiment_multilingual •
            **Paper**: *Cross-lingual sentiment transfer via QLoRA fine-tuned SLM*
            """
        )

    return demo


# --------------------------------------------------------------------------- #
#  Entry Point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False)

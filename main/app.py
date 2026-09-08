"""
Streamlit demo for FECL-EmotionAI.

  streamlit run app.py

Loads the project's OWN trained checkpoint (checkpoints/federated.pt by
default -- the actual federated-learning global model this project
produces). It never silently falls back to an unrelated pretrained emotion
classifier: if no checkpoint exists yet, it shows the exact command to
train one instead of faking a prediction.
"""

from pathlib import Path

import streamlit as st

from config import CHECKPOINT_DIR, get_default_config
from predict import load_model_and_encoder, predict_text

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "federated.pt"

st.set_page_config(page_title="FECL-EmotionAI", page_icon=":speech_balloon:")


@st.cache_resource
def get_encoder_and_model(checkpoint_path: str):
    config = get_default_config()
    encoder, model = load_model_and_encoder(Path(checkpoint_path), config)
    return encoder, model, config


def main():
    st.title("FECL-EmotionAI")
    st.caption("Federated Emotion Classification with Privacy -- text to emotion, using this project's own trained model.")

    checkpoint_path = DEFAULT_CHECKPOINT
    if not checkpoint_path.exists():
        st.error(
            "No trained checkpoint found at `checkpoints/federated.pt`.\n\n"
            "This demo only ever uses this project's own trained model -- it will "
            "not fall back to a different, unrelated pretrained emotion classifier.\n\n"
            "Train one first by running, from the project root:\n\n"
            "```\npython run_experiments.py\n```\n\n"
            "or, for just the federated model:\n\n"
            "```\npython train.py --mode federated\n```"
        )
        return

    encoder, model, config = get_encoder_and_model(str(checkpoint_path))

    text = st.text_area("Enter a sentence:", value="I can't believe how happy this makes me feel today.")
    if st.button("Predict emotion") and text.strip():
        label, confidence, all_probs = predict_text(text, encoder, model, config)
        st.subheader(f"Predicted emotion: **{label}**")
        st.write(f"Confidence: {confidence:.1%}")
        st.bar_chart(all_probs)


if __name__ == "__main__":
    main()

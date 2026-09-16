"""
Single-sentence inference: text -> predicted emotion + confidence.

  python predict.py "I can't believe how happy this makes me feel"

Uses the SAME frozen encoder + trained head pipeline as training/evaluation
(no unrelated pretrained emotion classifier is ever substituted in).
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from config import CHECKPOINT_DIR, Config, get_default_config
from model import ClassifierHead, FrozenEncoder


def predict_text(
    text: str, encoder: FrozenEncoder, model: ClassifierHead, config: Config
) -> Tuple[str, float, Dict[str, float]]:
    """Return (predicted_label, confidence, {label: probability})."""
    embedding = encoder.encode_single(text).unsqueeze(0)  # [1, embedding_dim]
    model.eval()
    with torch.no_grad():
        logits, _ = model(embedding)
        probs = F.softmax(logits, dim=1).squeeze(0)

    label_probs = {name: float(p) for name, p in zip(config.label_names, probs)}
    predicted_idx = int(probs.argmax())
    predicted_label = config.label_names[predicted_idx]
    confidence = float(probs[predicted_idx])
    return predicted_label, confidence, label_probs


def load_model_and_encoder(checkpoint_path: Path, config: Config):
    encoder = FrozenEncoder(config)
    model = ClassifierHead(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return encoder, model


def _cli():
    parser = argparse.ArgumentParser(description="Predict the emotion of a sentence.")
    parser.add_argument("text", type=str, help="Input sentence.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "federated.pt"),
        help="Path to a trained ClassifierHead checkpoint.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train one first, e.g.: python train.py --mode federated"
        )

    config = get_default_config()
    encoder, model = load_model_and_encoder(checkpoint_path, config)
    label, confidence, all_probs = predict_text(args.text, encoder, model, config)

    print(f"Text:      {args.text}")
    print(f"Predicted: {label}  (confidence={confidence:.3f})")
    print("All class probabilities:")
    for name, prob in sorted(all_probs.items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s} {prob:.3f}")


if __name__ == "__main__":
    _cli()

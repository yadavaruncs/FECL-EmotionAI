"""
Evaluation utilities: accuracy, macro-F1, and confusion matrix on a
held-out global test set. The SAME test set and metric code are used for
every approach (centralized / federated / federated+DP) so results are
directly comparable, and results are never reported on a client's local
training data.
"""

import argparse
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config import Config, CHECKPOINT_DIR, RESULTS_DIR, get_default_config
from model import ClassifierHead


@torch.no_grad()
def evaluate_model(
    model: ClassifierHead,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
) -> Dict:
    """Evaluate a trained classification head on precomputed frozen embeddings."""
    model.eval()
    inputs = torch.tensor(embeddings, dtype=torch.float32)
    logits, _ = model(inputs)
    predictions = logits.argmax(dim=1).numpy()

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=list(range(len(label_names))))

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "predictions": predictions,
    }


def plot_confusion_matrix(cm: np.ndarray, label_names: Sequence[str], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_checkpoint(checkpoint_path: Path, config: Config) -> ClassifierHead:
    model = ClassifierHead(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _cli():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "federated.pt"),
        help="Path to a ClassifierHead checkpoint (.pt).",
    )
    args = parser.parse_args()

    from data import load_all_embeddings  # local import to avoid unused cost when unused

    config = get_default_config()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train one first, e.g.: python train.py --mode federated"
        )

    embeddings = load_all_embeddings(config)
    test_embeddings, test_labels = embeddings["test"]

    model = load_checkpoint(checkpoint_path, config)
    metrics = evaluate_model(model, test_embeddings, test_labels, config.label_names)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(metrics["confusion_matrix"])

    out_path = RESULTS_DIR / f"confusion_matrix_{checkpoint_path.stem}.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"], config.label_names, f"Confusion Matrix: {checkpoint_path.stem}", out_path
    )
    print(f"Saved confusion matrix plot to {out_path}")


if __name__ == "__main__":
    _cli()

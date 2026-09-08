"""
Training entry point.

  python train.py --mode centralized
  python train.py --mode federated
  python train.py --mode federated_dp

Each mode trains the (frozen-encoder + trainable head) system and writes a
checkpoint to checkpoints/<mode>.pt. This is the script referenced by the
Streamlit demo (app.py) when no checkpoint is found yet.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CHECKPOINT_DIR, get_default_config
from data import load_all_embeddings
from evaluate import evaluate_model
from federated import run_federated_training
from model import ClassifierHead, supcon_loss


def train_centralized(train_embeddings, train_labels, config):
    """Standard (non-federated) training on the full pooled training set --
    the classical baseline that federated learning is compared against."""
    torch.manual_seed(config.seed)
    model = ClassifierHead(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    # Same total local-update budget as one federated client would see across
    # all rounds, so the comparison is not accidentally biased by "who saw the
    # data more times": num_clients clients x num_rounds x local_epochs.
    total_epochs = config.num_rounds * config.local_epochs
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    ce_loss_fn = nn.CrossEntropyLoss()
    for _ in range(total_epochs):
        for batch_embeddings, batch_labels in data_loader:
            optimizer.zero_grad()
            logits, projections = model(batch_embeddings)
            loss = ce_loss_fn(logits, batch_labels)
            if config.use_supcon:
                loss = loss + config.supcon_lambda * supcon_loss(
                    projections, batch_labels, config.supcon_temperature
                )
            loss.backward()
            optimizer.step()

    return model.state_dict()


def save_checkpoint(state_dict, name: str):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{name}.pt"
    torch.save(state_dict, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train the FECL-EmotionAI classification head.")
    parser.add_argument(
        "--mode",
        choices=["centralized", "federated", "federated_dp"],
        default="federated",
        help="Training regime to run.",
    )
    args = parser.parse_args()

    config = get_default_config()
    print("Loading data and computing frozen-encoder embeddings (cached after first run)...")
    embeddings = load_all_embeddings(config)
    train_embeddings, train_labels = embeddings["train"]
    val_embeddings, val_labels = embeddings["validation"]
    test_embeddings, test_labels = embeddings["test"]

    start = time.time()
    if args.mode == "centralized":
        state_dict = train_centralized(train_embeddings, train_labels, config)
    else:
        use_dp = args.mode == "federated_dp"
        state_dict, history, epsilon_log = run_federated_training(
            train_embeddings, train_labels, val_embeddings, val_labels, config, use_dp=use_dp
        )
        print("Per-round validation history:")
        for row in history:
            print(f"  round {row['round']}: acc={row['val_accuracy']:.4f} macro_f1={row['val_macro_f1']:.4f}")
        if epsilon_log:
            print(f"Final privacy spend: epsilon={epsilon_log[-1]:.3f} at delta={config.dp_delta}")
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    model = ClassifierHead(config)
    model.load_state_dict(state_dict)
    test_metrics = evaluate_model(model, test_embeddings, test_labels, config.label_names)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}  Test macro-F1: {test_metrics['macro_f1']:.4f}")

    checkpoint_path = save_checkpoint(state_dict, args.mode)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

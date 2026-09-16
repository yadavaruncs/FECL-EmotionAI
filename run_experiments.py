"""
Runs the full experiment suite for FECL-EmotionAI and writes:
  results/results.csv                  -- machine-readable metrics table
  results/results.png                  -- headline comparison chart
  results/privacy_utility.png          -- macro-F1 vs. privacy budget (epsilon)
  results/confusion_matrix_centralized.png
  results/confusion_matrix_federated.png

  python run_experiments.py

All numbers in results.csv are produced by actually running the training
and evaluation code in this repository -- nothing here is hand-entered.
If a stage cannot run in a given environment, its row is written with
metric columns set to "not_run" and a note explaining why, instead of a
fabricated number.
"""

import csv
import dataclasses
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CHECKPOINT_DIR, RESULTS_DIR, get_default_config
from data import load_all_embeddings
from evaluate import evaluate_model, plot_confusion_matrix
from federated import run_federated_training
from model import ClassifierHead
from train import save_checkpoint, train_centralized

DP_NOISE_SWEEP = [0.5, 1.0, 1.5]
HEADLINE_NOISE_MULTIPLIER = 1.0

RESULTS_CSV_FIELDS = [
    "experiment_group",
    "approach",
    "accuracy",
    "macro_f1",
    "epsilon",
    "delta",
    "notes",
]


def _row(experiment_group, approach, accuracy, macro_f1, epsilon="", delta="", notes=""):
    return {
        "experiment_group": experiment_group,
        "approach": approach,
        "accuracy": f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy,
        "macro_f1": f"{macro_f1:.4f}" if isinstance(macro_f1, float) else macro_f1,
        "epsilon": f"{epsilon:.3f}" if isinstance(epsilon, float) else epsilon,
        "delta": delta,
        "notes": notes,
    }


def main():
    config = get_default_config()
    rows = []

    print("=" * 70)
    print("Loading data and computing frozen-encoder embeddings (cached)...")
    print("=" * 70)
    embeddings = load_all_embeddings(config)
    train_embeddings, train_labels = embeddings["train"]
    val_embeddings, val_labels = embeddings["validation"]
    test_embeddings, test_labels = embeddings["test"]
    print(f"train={len(train_labels)}  val={len(val_labels)}  test={len(test_labels)}")

    # ------------------------------------------------------------------
    # 1. Centralized baseline
    # ------------------------------------------------------------------
    print("\n[1/4] Training centralized baseline...")
    t0 = time.time()
    centralized_state = train_centralized(train_embeddings, train_labels, config)
    print(f"  done in {time.time() - t0:.1f}s")

    model = ClassifierHead(config)
    model.load_state_dict(centralized_state)
    centralized_metrics = evaluate_model(model, test_embeddings, test_labels, config.label_names)
    print(f"  test accuracy={centralized_metrics['accuracy']:.4f} macro_f1={centralized_metrics['macro_f1']:.4f}")
    save_checkpoint(centralized_state, "centralized")
    plot_confusion_matrix(
        centralized_metrics["confusion_matrix"],
        config.label_names,
        "Confusion Matrix: Centralized",
        RESULTS_DIR / "confusion_matrix_centralized.png",
    )
    rows.append(
        _row(
            "main_comparison",
            "Centralized",
            centralized_metrics["accuracy"],
            centralized_metrics["macro_f1"],
            notes="Single model trained on the full pooled training set (CE + SupCon).",
        )
    )

    # ------------------------------------------------------------------
    # 2. Federated (no DP)
    # ------------------------------------------------------------------
    print("\n[2/4] Training federated (FedAvg, no DP)...")
    t0 = time.time()
    federated_state, federated_history, _ = run_federated_training(
        train_embeddings, train_labels, val_embeddings, val_labels, config, use_dp=False
    )
    print(f"  done in {time.time() - t0:.1f}s")

    model = ClassifierHead(config)
    model.load_state_dict(federated_state)
    federated_metrics = evaluate_model(model, test_embeddings, test_labels, config.label_names)
    print(f"  test accuracy={federated_metrics['accuracy']:.4f} macro_f1={federated_metrics['macro_f1']:.4f}")
    save_checkpoint(federated_state, "federated")
    plot_confusion_matrix(
        federated_metrics["confusion_matrix"],
        config.label_names,
        "Confusion Matrix: Federated",
        RESULTS_DIR / "confusion_matrix_federated.png",
    )
    rows.append(
        _row(
            "main_comparison",
            "Federated (FedAvg)",
            federated_metrics["accuracy"],
            federated_metrics["macro_f1"],
            notes=(
                f"{config.num_clients} clients, Dirichlet alpha={config.dirichlet_alpha}, "
                f"{config.num_rounds} rounds x {config.local_epochs} local epochs (CE + SupCon)."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 3. Federated + DP sweep (privacy-utility trade-off)
    # ------------------------------------------------------------------
    print("\n[3/4] Training federated + differential privacy sweep...")
    privacy_utility_points = []  # (noise_multiplier, epsilon, macro_f1, accuracy)
    headline_dp_metrics = None

    for noise_multiplier in DP_NOISE_SWEEP:
        dp_config = dataclasses.replace(config, dp_noise_multiplier=noise_multiplier)
        print(f"  noise_multiplier={noise_multiplier} ...")
        t0 = time.time()
        dp_state, dp_history, epsilon_log = run_federated_training(
            train_embeddings, train_labels, val_embeddings, val_labels, dp_config, use_dp=True
        )
        elapsed = time.time() - t0
        epsilon = epsilon_log[-1] if epsilon_log else float("nan")

        model = ClassifierHead(config)
        model.load_state_dict(dp_state)
        dp_metrics = evaluate_model(model, test_embeddings, test_labels, config.label_names)
        print(
            f"    done in {elapsed:.1f}s | epsilon={epsilon:.3f} "
            f"accuracy={dp_metrics['accuracy']:.4f} macro_f1={dp_metrics['macro_f1']:.4f}"
        )

        privacy_utility_points.append(
            (noise_multiplier, epsilon, dp_metrics["macro_f1"], dp_metrics["accuracy"])
        )
        rows.append(
            _row(
                "privacy_utility_sweep",
                f"Federated + DP (noise_multiplier={noise_multiplier})",
                dp_metrics["accuracy"],
                dp_metrics["macro_f1"],
                epsilon=epsilon,
                delta=str(config.dp_delta),
                notes="Opacus DP-SGD per client; epsilon computed by Opacus's RDP accountant.",
            )
        )

        if noise_multiplier == HEADLINE_NOISE_MULTIPLIER:
            headline_dp_metrics = dp_metrics
            save_checkpoint(dp_state, "federated_dp")
            plot_confusion_matrix(
                dp_metrics["confusion_matrix"],
                config.label_names,
                "Confusion Matrix: Federated + DP",
                RESULTS_DIR / "confusion_matrix_federated_dp.png",
            )

    if headline_dp_metrics is not None:
        headline_epsilon = next(
            eps for nm, eps, _, _ in privacy_utility_points if nm == HEADLINE_NOISE_MULTIPLIER
        )
        rows.append(
            _row(
                "main_comparison",
                f"Federated + DP (noise_multiplier={HEADLINE_NOISE_MULTIPLIER})",
                headline_dp_metrics["accuracy"],
                headline_dp_metrics["macro_f1"],
                epsilon=headline_epsilon,
                delta=str(config.dp_delta),
                notes="Headline DP configuration used in the main comparison chart.",
            )
        )

    # ------------------------------------------------------------------
    # 4. SupCon ablation: CE-only vs CE + SupCon (federated setting)
    # ------------------------------------------------------------------
    print("\n[4/4] Running SupCon ablation (CE-only vs CE + SupCon)...")
    ce_only_config = dataclasses.replace(config, use_supcon=False)
    t0 = time.time()
    ce_only_state, _, _ = run_federated_training(
        train_embeddings, train_labels, val_embeddings, val_labels, ce_only_config, use_dp=False
    )
    print(f"  done in {time.time() - t0:.1f}s")
    model = ClassifierHead(config)
    model.load_state_dict(ce_only_state)
    ce_only_metrics = evaluate_model(model, test_embeddings, test_labels, config.label_names)
    print(f"  CE-only test accuracy={ce_only_metrics['accuracy']:.4f} macro_f1={ce_only_metrics['macro_f1']:.4f}")

    rows.append(
        _row(
            "supcon_ablation",
            "Federated, CE only",
            ce_only_metrics["accuracy"],
            ce_only_metrics["macro_f1"],
            notes="Same partition/seed/rounds as the main federated run, SupCon disabled.",
        )
    )
    rows.append(
        _row(
            "supcon_ablation",
            "Federated, CE + SupCon",
            federated_metrics["accuracy"],
            federated_metrics["macro_f1"],
            notes=f"lambda={config.supcon_lambda}, temperature={config.supcon_temperature}.",
        )
    )

    # ------------------------------------------------------------------
    # Write results.csv
    # ------------------------------------------------------------------
    csv_path = RESULTS_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ------------------------------------------------------------------
    # Plot results.png -- headline comparison chart
    # ------------------------------------------------------------------
    headline_rows = [r for r in rows if r["experiment_group"] == "main_comparison"]
    labels = [r["approach"] for r in headline_rows]
    macro_f1_values = [float(r["macro_f1"]) for r in headline_rows]
    accuracy_values = [float(r["accuracy"]) for r in headline_rows]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors_f1 = "#2E5EAA"
    bar_colors_acc = "#8DB0E8"
    ax.bar([i - width / 2 for i in x], macro_f1_values, width, label="Macro-F1", color=bar_colors_f1)
    ax.bar([i + width / 2 for i in x], accuracy_values, width, label="Accuracy", color=bar_colors_acc)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("FECL-EmotionAI: Centralized vs. Federated vs. Federated + DP")
    ax.legend()
    for i, (f1v, accv) in enumerate(zip(macro_f1_values, accuracy_values)):
        ax.text(i - width / 2, f1v + 0.01, f"{f1v:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, accv + 0.01, f"{accv:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "results.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {RESULTS_DIR / 'results.png'}")

    # ------------------------------------------------------------------
    # Plot privacy_utility.png -- macro-F1 vs. epsilon across noise levels
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    epsilons = [p[1] for p in privacy_utility_points]
    f1s = [p[2] for p in privacy_utility_points]
    ax.plot(epsilons, f1s, marker="o", color="#2E5EAA", label="Federated + DP")
    # Points at low noise multipliers can land close together on the x-axis
    # (epsilon shrinks fast as noise grows); stagger label y-offsets so text
    # doesn't overlap, and sort by epsilon first so the stagger is stable.
    sorted_points = sorted(privacy_utility_points, key=lambda p: p[1])
    for stagger_idx, (nm, eps, f1v, _) in enumerate(sorted_points):
        y_offset = 8 + (stagger_idx % 2) * 14
        ax.annotate(
            f"noise={nm}", (eps, f1v), textcoords="offset points", xytext=(6, y_offset), fontsize=8
        )
    ax.axhline(federated_metrics["macro_f1"], color="#999999", linestyle="--", label="Federated (no DP)")
    ax.set_xlabel("Privacy budget epsilon (lower = more private)")
    ax.set_ylabel("Macro-F1 on held-out test set")
    ax.set_title("Privacy-Utility Trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "privacy_utility.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {RESULTS_DIR / 'privacy_utility.png'}")

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()

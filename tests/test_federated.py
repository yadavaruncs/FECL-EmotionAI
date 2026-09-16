"""Tests for federated.py: Dirichlet partitioning and FedAvg aggregation."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from federated import dirichlet_partition, fedavg_aggregate


def _class_proportions(labels, indices, num_classes):
    subset_labels = labels[indices]
    counts = np.bincount(subset_labels, minlength=num_classes)
    return counts / counts.sum()


def test_dirichlet_partition_covers_all_data_with_correct_client_count():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 6, size=1000)
    num_clients = 3

    client_indices = dirichlet_partition(labels, num_clients=num_clients, alpha=0.3, seed=42)

    assert len(client_indices) == num_clients
    all_indices = np.concatenate(client_indices)
    assert len(all_indices) == len(labels)
    # No overlap and every original index is used exactly once.
    assert sorted(all_indices.tolist()) == list(range(len(labels)))


def test_dirichlet_partition_is_more_non_iid_at_lower_alpha():
    rng = np.random.default_rng(1)
    # Balanced synthetic labels so any skew we see comes from the partition, not the data.
    labels = np.tile(np.arange(6), 500)
    rng.shuffle(labels)
    num_clients = 3

    low_alpha_clients = dirichlet_partition(labels, num_clients, alpha=0.05, seed=7)
    high_alpha_clients = dirichlet_partition(labels, num_clients, alpha=100.0, seed=7)

    def avg_max_class_proportion(client_indices):
        proportions = [_class_proportions(labels, idx, 6).max() for idx in client_indices]
        return float(np.mean(proportions))

    low_alpha_skew = avg_max_class_proportion(low_alpha_clients)
    high_alpha_skew = avg_max_class_proportion(high_alpha_clients)

    # Low alpha should concentrate each client's data into fewer classes,
    # i.e. a higher average "largest class proportion" than near-uniform alpha.
    assert low_alpha_skew > high_alpha_skew


def test_fedavg_matches_hand_computed_weighted_average():
    state_dict_a = {"weight": torch.tensor([1.0, 2.0]), "count": torch.tensor(5, dtype=torch.int64)}
    state_dict_b = {"weight": torch.tensor([3.0, 4.0]), "count": torch.tensor(9, dtype=torch.int64)}
    sample_counts = [10, 30]  # weights: 0.25, 0.75

    result = fedavg_aggregate([state_dict_a, state_dict_b], sample_counts)

    expected_weight = 0.25 * torch.tensor([1.0, 2.0]) + 0.75 * torch.tensor([3.0, 4.0])
    assert torch.allclose(result["weight"], expected_weight)
    # Non-floating-point tensors must not be averaged; they are copied from the first client.
    assert result["count"].item() == 5

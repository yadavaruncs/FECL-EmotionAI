"""
Federated learning core: Dirichlet non-IID client partitioning, local client
training, and sample-weighted FedAvg aggregation.

FedAvg formula (McMahan et al., 2017), whiteboard version:

    w_global <- sum_k ( n_k / n_total ) * w_k

where w_k are the parameters returned by client k after local training,
n_k is client k's number of local training samples, and n_total = sum_k n_k.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from evaluate import evaluate_model
from model import ClassifierHead, supcon_loss
from privacy import attach_privacy_engine, unwrap_state_dict


def dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int
) -> List[np.ndarray]:
    """Split sample indices across `num_clients` clients using a per-class
    Dirichlet(alpha) split, producing non-IID label distributions.

    Every sample is assigned to exactly one client (the partition covers the
    whole dataset with no overlap). Lower alpha -> each client's Dirichlet
    draw is more skewed towards a few classes -> stronger label imbalance
    between clients.
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return [np.array(idx, dtype=np.int64) for idx in client_indices]


def fedavg_aggregate(
    state_dicts: List[Dict[str, torch.Tensor]], sample_counts: List[int]
) -> Dict[str, torch.Tensor]:
    """Sample-weighted federated averaging.

    Only floating-point tensors are averaged as weighted sums; any
    non-floating-point tensor (e.g. an integer counter/buffer) is copied
    from the first client instead of being (incorrectly) averaged as a float.
    """
    total_samples = sum(sample_counts)
    weights = [n / total_samples for n in sample_counts]

    averaged: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key in state_dicts[0].keys():
        reference = state_dicts[0][key]
        if torch.is_floating_point(reference):
            weighted_sum = torch.zeros_like(reference, dtype=torch.float32)
            for weight, state_dict in zip(weights, state_dicts):
                weighted_sum += weight * state_dict[key].to(torch.float32)
            averaged[key] = weighted_sum.to(reference.dtype)
        else:
            averaged[key] = reference.clone()
    return averaged


def _train_local_model(
    global_state: Dict[str, torch.Tensor],
    embeddings: np.ndarray,
    labels: np.ndarray,
    config: Config,
    use_dp: bool,
    privacy_engine=None,
) -> Tuple[Dict[str, torch.Tensor], int, Optional[float], Optional[object]]:
    """Train one client's local copy of the classification head for one round.

    When `use_dp` is True, `privacy_engine` should be the SAME PrivacyEngine
    instance across all of this client's rounds so its accountant accumulates
    the client's cumulative privacy spend (see privacy.attach_privacy_engine).
    """
    model = ClassifierHead(config)
    model.load_state_dict(global_state)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    dataset = TensorDataset(
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    if use_dp:
        model, optimizer, data_loader, privacy_engine = attach_privacy_engine(
            model, optimizer, data_loader, config, privacy_engine=privacy_engine
        )

    ce_loss_fn = nn.CrossEntropyLoss()
    for _ in range(config.local_epochs):
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

    epsilon = None
    if use_dp:
        epsilon = privacy_engine.get_epsilon(delta=config.dp_delta)
        state_dict = unwrap_state_dict(model)
    else:
        state_dict = model.state_dict()

    state_dict = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
    return state_dict, len(labels), epsilon, privacy_engine


def run_federated_training(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    config: Config,
    use_dp: bool = False,
) -> Tuple[Dict[str, torch.Tensor], List[dict], List[float]]:
    """Run the full FedAvg loop.

    Returns (final_global_state_dict, per_round_val_history, epsilon_log).
    epsilon_log is empty when use_dp is False.
    """
    torch.manual_seed(config.seed)
    client_index_sets = dirichlet_partition(
        train_labels, config.num_clients, config.dirichlet_alpha, config.seed
    )

    global_model = ClassifierHead(config)
    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    history = []
    epsilon_log: List[float] = []
    # One persistent PrivacyEngine per client, reused every round, so each
    # client's accountant accumulates its TRUE cumulative privacy spend
    # across all rounds it participates in (see privacy.attach_privacy_engine).
    client_privacy_engines = [None] * config.num_clients

    for round_id in range(config.num_rounds):
        client_states = []
        client_sample_counts = []
        round_epsilons = []

        for client_id, client_indices in enumerate(client_index_sets):
            client_embeddings = train_embeddings[client_indices]
            client_labels = train_labels[client_indices]

            local_state, n_samples, epsilon, engine = _train_local_model(
                global_state,
                client_embeddings,
                client_labels,
                config,
                use_dp,
                privacy_engine=client_privacy_engines[client_id],
            )
            client_privacy_engines[client_id] = engine
            client_states.append(local_state)
            client_sample_counts.append(n_samples)
            if epsilon is not None:
                round_epsilons.append(epsilon)

        global_state = fedavg_aggregate(client_states, client_sample_counts)

        if round_epsilons:
            epsilon_log.append(max(round_epsilons))

        eval_model = ClassifierHead(config)
        eval_model.load_state_dict(global_state)
        metrics = evaluate_model(eval_model, val_embeddings, val_labels, config.label_names)
        history.append(
            {
                "round": round_id + 1,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
            }
        )

    return global_state, history, epsilon_log

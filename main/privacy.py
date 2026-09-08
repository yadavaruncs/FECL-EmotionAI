"""
Differential privacy for local client training, via Opacus DP-SGD.

This is REAL DP-SGD, not a hand-rolled noise hack:
  - per-sample gradient clipping to a fixed L2 norm (`dp_max_grad_norm`)
  - calibrated Gaussian noise added to the summed clipped gradients
    (`dp_noise_multiplier`)
  - privacy accounting via Opacus's RDP accountant, so the (epsilon, delta)
    reported in results.csv is the ACTUAL value computed by the accountant
    for the number of steps/epochs/sampling-rate actually used -- never a
    predetermined or hand-picked number.

Each client trains its own local copy of the classification head, so each
client accumulates its own, independent (epsilon, delta) privacy guarantee
with respect to its own local training data.
"""

import warnings
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

from config import Config


def attach_privacy_engine(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data_loader: DataLoader,
    config: Config,
    privacy_engine: PrivacyEngine = None,
) -> Tuple[nn.Module, optim.Optimizer, DataLoader, PrivacyEngine]:
    """Wrap a model/optimizer/dataloader for DP-SGD training with Opacus.

    A client participates in MULTIPLE federated rounds, and each round is
    another query against that client's local data -- so its privacy spend
    must accumulate across rounds, not reset every round. Pass in the SAME
    `privacy_engine` instance (one per client, created once before the
    round loop) across every round for that client so its accountant keeps
    a running history; a fresh engine is created only if none is given.

    Returns (private_model, private_optimizer, private_data_loader, engine).
    Call `engine.get_epsilon(delta=config.dp_delta)` after training to obtain
    the actual CUMULATIVE privacy spend for every step taken on this engine
    so far.
    """
    if privacy_engine is None:
        privacy_engine = PrivacyEngine()
    with warnings.catch_warnings():
        # Expected: we intentionally build a fresh DataLoader for the same
        # client subset every round (same logical dataset, new Python object).
        warnings.filterwarnings("ignore", message="PrivacyEngine detected new dataset object")
        private_model, private_optimizer, private_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=config.dp_noise_multiplier,
            max_grad_norm=config.dp_max_grad_norm,
        )
    return private_model, private_optimizer, private_loader, privacy_engine


def unwrap_state_dict(private_model: nn.Module) -> dict:
    """Opacus wraps the model in a GradSampleModule (params prefixed with
    `_module.`). Unwrap it so the returned state_dict has plain keys that
    match a fresh (non-DP) ClassifierHead, for FedAvg aggregation.
    """
    underlying = private_model._module if hasattr(private_model, "_module") else private_model
    return underlying.state_dict()

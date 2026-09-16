"""Tests for model.py: frozen encoder, trainable head, and SupCon loss."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_default_config
from model import ClassifierHead, FrozenEncoder, supcon_loss


def test_frozen_encoder_has_no_trainable_parameters():
    config = get_default_config()
    encoder = FrozenEncoder(config)
    params = list(encoder.encoder_module.parameters())
    assert len(params) > 0
    assert all(not p.requires_grad for p in params)


def test_classifier_head_forward_shapes():
    config = get_default_config()
    model = ClassifierHead(config)
    batch_size = 5
    embeddings = torch.randn(batch_size, config.embedding_dim)

    logits, projections = model(embeddings)

    assert logits.shape == (batch_size, len(config.label_names))
    assert projections.shape == (batch_size, config.projection_dim)
    # Projections should be L2-normalized (SupCon requires unit vectors).
    norms = projections.norm(dim=1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)


def test_supcon_loss_is_finite_and_nonnegative_with_positive_pairs():
    torch.manual_seed(0)
    projections = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss = supcon_loss(projections, labels, temperature=0.07)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_supcon_loss_handles_no_positive_pairs():
    torch.manual_seed(0)
    projections = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    labels = torch.tensor([0, 1, 2, 3])  # every label unique -> no positives for anyone

    loss = supcon_loss(projections, labels, temperature=0.07)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0

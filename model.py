"""
Model components for FECL-EmotionAI.

Two clearly separated pieces:

1. `FrozenEncoder` -- a pretrained DistilBERT wrapped so that ALL of its
   parameters have requires_grad=False. It is never trained; it is only used
   to turn raw text into a fixed-size sentence embedding (mean-pooled last
   hidden state). Fine-tuning a 66M-parameter transformer inside a
   federated + differential-privacy loop is unnecessary for this task and
   would make every experiment far too slow to run on a laptop/Colab -- see
   the "Engineering Decisions" section of the README.

2. `ClassifierHead` -- the ONLY trainable part of the system. A small shared
   trunk maps a frozen sentence embedding to a learned representation, which
   feeds both a classifier (class logits) and a small projection head used
   for supervised contrastive learning (SupCon). This is the module that is
   trained locally on each federated client and aggregated with FedAvg.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import Config


class FrozenEncoder:
    """Wraps a pretrained transformer encoder with all parameters frozen.

    Not an nn.Module subclass on purpose: it is never part of a trainable
    computation graph, never checkpointed, and never sent to clients in the
    federated loop -- only its (fixed) output embeddings are used.
    """

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
        self.model = AutoModel.from_pretrained(config.encoder_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def encoder_module(self) -> nn.Module:
        """Expose the underlying nn.Module (e.g. to assert requires_grad in tests)."""
        return self.model

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Mean-pool (attention-mask aware) last hidden states into one vector per text."""
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state  # [B, T, H]
            mask = encoded["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            all_embeddings.append(pooled)
        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_single(self, text: str) -> torch.Tensor:
        return self.encode_texts([text], batch_size=1)[0]


class ClassifierHead(nn.Module):
    """The only trainable model in the federated system.

    A shared trainable trunk produces one representation `h` from the frozen
    embedding; the classifier and projection heads are both thin linear
    layers on top of that SAME `h`. This is deliberate: if the classifier and
    projection heads instead read the frozen embedding independently, they
    would share no parameters at all, and SupCon's gradient (which only
    flows through the projection head) could never influence the
    classifier's predictions -- making SupCon a no-op in training instead of
    an actual joint objective. Sharing the trunk is what makes
    `CE + lambda * SupCon` a genuine joint loss.

    Input:  frozen sentence embedding [B, embedding_dim]
    Output: (logits [B, num_classes], normalized projection [B, projection_dim])
    """

    def __init__(self, config: Config):
        super().__init__()
        embedding_dim = config.embedding_dim
        hidden_dim = config.hidden_dim
        num_classes = len(config.label_names)

        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projection = nn.Linear(hidden_dim, config.projection_dim)

    def forward(self, embeddings: torch.Tensor):
        shared_representation = self.trunk(embeddings)
        logits = self.classifier(shared_representation)
        projections = F.normalize(self.projection(shared_representation), dim=1)
        return logits, projections


def supcon_loss(
    projections: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., 2020), simplified for one view per sample.

    For each anchor, positives are other samples in the batch sharing its label.
    Anchors with no positive in the batch are safely excluded (returns 0.0 loss
    if the whole batch has no positive pairs at all, instead of NaN).
    """
    device = projections.device
    batch_size = projections.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=device)

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - self_mask  # exclude self as its own positive

    has_positive = positive_mask.sum(dim=1) > 0
    if not torch.any(has_positive):
        return torch.tensor(0.0, device=device)

    similarity = torch.matmul(projections, projections.T) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()  # numerical stability
    exp_sim = torch.exp(similarity) * (1 - self_mask)  # exclude self from denominator
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    positive_counts = positive_mask.sum(dim=1).clamp(min=1e-12)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts

    loss_per_anchor = -mean_log_prob_pos[has_positive]
    return loss_per_anchor.mean()

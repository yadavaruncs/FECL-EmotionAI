

import os

# Works around a known OpenMP double-initialization crash (OMP Error #15)
# that occurs on some Windows/conda setups when both PyTorch and NumPy ship
# their own copies of the OpenMP runtime. Must be set before torch is
# imported anywhere in the project, so it lives at the top of config.py,
# which every other module imports first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# Filesystem layout (all paths are resolved relative to this file, so the
# project runs correctly no matter which directory it is launched from).

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"            # cached frozen-encoder embeddings
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"  # trained head checkpoints
RESULTS_DIR = PROJECT_ROOT / "results"         # results.csv, results.png, ...

for _d in (CACHE_DIR, CHECKPOINT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    
    # Dataset
    
    dataset_name: str = "dair-ai/emotion"
    label_names: Tuple[str, ...] = (
        "sadness", "joy", "love", "anger", "fear", "surprise",
    )

    
    # Frozen encoder
    
    encoder_name: str = "distilbert-base-uncased"
    max_seq_length: int = 64
    embedding_batch_size: int = 32
    embedding_dim: int = 768  # DistilBERT hidden size

    
    # Trainable classification / projection head
    
    hidden_dim: int = 128
    projection_dim: int = 128
    dropout: float = 0.2

    
    # Non-IID federated client partitioning (Dirichlet)
    
    num_clients: int = 3
    dirichlet_alpha: float = 0.3  # lower -> more label heterogeneity

    
    # Federated averaging (FedAvg)
    
    num_rounds: int = 5
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3

    
    # Supervised contrastive loss (SupCon)
    
    use_supcon: bool = True
    supcon_lambda: float = 0.1
    supcon_temperature: float = 0.07

    
    # Differential privacy (Opacus DP-SGD), used only when dp=True
    
    dp_noise_multiplier: float = 1.0
    dp_max_grad_norm: float = 1.0
    dp_delta: float = 1e-5

    
    # Misc
    
    seed: int = 42
    device: str = "cpu"


def get_default_config() -> Config:
    """Return a fresh default configuration."""
    return Config()

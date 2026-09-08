"""
Dataset loading and frozen-encoder feature extraction.

We use the real HuggingFace dataset `dair-ai/emotion` (6-class English
emotion classification, official train/validation/test splits, downloaded
automatically at runtime).

Because the transformer encoder is FROZEN (see model.py), its output for a
given sentence never changes during training. Running the encoder once per
sentence and caching the resulting embedding is therefore mathematically
identical to running the frozen encoder inside every training loop, but is
orders of magnitude faster -- this is what makes it practical to run
multiple federated rounds x multiple clients x multiple privacy levels on a
plain CPU/laptop. This is a deliberate engineering decision, documented in
the README.
"""

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from datasets import load_dataset

from config import Config, CACHE_DIR
from model import FrozenEncoder


def load_emotion_dataset(config: Config) -> Dict[str, Tuple[list, np.ndarray]]:
    """Download (if needed) and return the official dair-ai/emotion splits.

    Returns a dict: split_name -> (list_of_texts, label_array)
    """
    raw = load_dataset(config.dataset_name)
    splits = {}
    for split in ("train", "validation", "test"):
        texts = list(raw[split]["text"])
        labels = np.array(raw[split]["label"], dtype=np.int64)
        splits[split] = (texts, labels)
    return splits


def _cache_path(split: str, config: Config) -> Path:
    key = f"{config.dataset_name}-{config.encoder_name}-{config.max_seq_length}-{split}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"embeddings_{split}_{digest}.npz"


def get_frozen_embeddings(
    texts, labels: np.ndarray, split: str, config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (or load from cache) frozen-encoder embeddings for a split.

    Returns (embeddings [N, embedding_dim], labels [N]).
    """
    cache_file = _cache_path(split, config)
    if cache_file.exists():
        data = np.load(cache_file)
        return data["embeddings"], data["labels"]

    encoder = FrozenEncoder(config)
    embeddings = encoder.encode_texts(texts, batch_size=config.embedding_batch_size)
    embeddings = embeddings.numpy()

    np.savez(cache_file, embeddings=embeddings, labels=labels)
    return embeddings, labels


def load_all_embeddings(config: Config) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load raw text splits and return frozen embeddings for train/val/test."""
    splits = load_emotion_dataset(config)
    result = {}
    for split, (texts, labels) in splits.items():
        emb, lab = get_frozen_embeddings(texts, labels, split, config)
        result[split] = (emb, lab)
    return result

"""Tests for data.py: dataset loading and frozen-encoder embedding caching.

These tests use the real dair-ai/emotion dataset and the real frozen
DistilBERT encoder (downloaded automatically, cached locally by
HuggingFace after the first run -- consistent with how the rest of the
project works).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_default_config
from data import get_frozen_embeddings, load_emotion_dataset


def test_load_emotion_dataset_has_official_splits_with_no_leakage():
    config = get_default_config()
    splits = load_emotion_dataset(config)

    assert set(splits.keys()) == {"train", "validation", "test"}

    train_texts, train_labels = splits["train"]
    test_texts, test_labels = splits["test"]

    assert len(train_texts) > 0 and len(test_texts) > 0
    assert train_labels.min() >= 0
    assert train_labels.max() < len(config.label_names)

    # The official dair-ai/emotion splits are not perfectly deduplicated (a
    # small number of short, generic sentences repeat across splits), but
    # this project introduces no additional leakage: overlap must stay a
    # tiny fraction of the test set, not something our own split logic adds.
    overlap = set(train_texts) & set(test_texts)
    assert len(overlap) / len(test_texts) < 0.01


def test_get_frozen_embeddings_shape_and_cache_consistency():
    config = get_default_config()
    texts = [
        "i am feeling really happy about this",
        "this makes me so angry and frustrated",
        "i am scared of what might happen next",
    ]
    labels = np.array([1, 3, 4], dtype=np.int64)

    embeddings_first, labels_first = get_frozen_embeddings(texts, labels, "unit_test_split", config)
    embeddings_second, labels_second = get_frozen_embeddings(texts, labels, "unit_test_split", config)

    assert embeddings_first.shape == (3, config.embedding_dim)
    assert np.array_equal(labels_first, labels)
    # Second call must hit the cache and return identical embeddings.
    assert np.allclose(embeddings_first, embeddings_second)
    assert np.array_equal(labels_second, labels)

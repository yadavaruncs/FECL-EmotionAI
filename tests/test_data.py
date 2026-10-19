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


def test_get_frozen_embeddings_shape_and_cache_consistency(tmp_path):
    # tmp_path: a fresh, empty cache dir per run, so the first call below
    # really runs the encoder (instead of loading a file from an earlier run)
    # and the test never writes into the project's real .cache/.
    config = get_default_config()
    texts = [
        "i am feeling really happy about this",
        "this makes me so angry and frustrated",
        "i am scared of what might happen next",
    ]
    labels = np.array([1, 3, 4], dtype=np.int64)

    assert list(tmp_path.iterdir()) == []
    embeddings_first, labels_first = get_frozen_embeddings(
        texts, labels, "unit_test_split", config, cache_dir=tmp_path
    )
    assert len(list(tmp_path.glob("*.npz"))) == 1  # encoded, then cached

    embeddings_second, labels_second = get_frozen_embeddings(
        texts, labels, "unit_test_split", config, cache_dir=tmp_path
    )

    assert embeddings_first.shape == (3, config.embedding_dim)
    assert np.isfinite(embeddings_first).all()
    assert np.array_equal(labels_first, labels)
    # Second call must hit the cache and return identical embeddings.
    assert np.allclose(embeddings_first, embeddings_second)
    assert np.array_equal(labels_second, labels)
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_embedding_cache_key_depends_on_texts_not_just_split_name(tmp_path):
    config = get_default_config()
    labels = np.array([1], dtype=np.int64)

    happy, _ = get_frozen_embeddings(["i am so happy"], labels, "same_split", config, cache_dir=tmp_path)
    angry, _ = get_frozen_embeddings(["i am furious"], labels, "same_split", config, cache_dir=tmp_path)

    # Same split name, different sentences -> must be a separate cache entry
    # with freshly computed (different) embeddings, never the stale ones.
    assert len(list(tmp_path.glob("*.npz"))) == 2
    assert not np.allclose(happy, angry)

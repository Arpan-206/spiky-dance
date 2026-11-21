"""The baseline scorer: a statistical distance with the Model.score contract."""

from __future__ import annotations

import numpy as np
import pytest

from dance import baseline, score
from dance.contracts import FEATURE_DIM


def _clips(seed: int, n: int = 4, bins: int = 300) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.5, 0.05, (bins, 1, FEATURE_DIM)).astype(np.float32) for _ in range(n)]


def test_train_load_score_follow_the_contract(tmp_path):
    baseline.train(_clips(0) + [_clips(1, 1, 50)[0].repeat(2, axis=1)], tmp_path)
    assert (tmp_path / baseline.ARTEFACT).exists()
    model = baseline.load_model(tmp_path)
    feats = _clips(2, 1, 120)[0].repeat(3, axis=1)
    scores = model.score(feats)
    assert scores.shape == (120, 3) and scores.dtype == np.float32
    assert scores.min() >= 0 and scores.max() <= 1
    np.testing.assert_array_equal(scores, model.score(feats))


def test_held_out_normal_is_near_the_median_and_outliers_are_high(tmp_path):
    baseline.train(_clips(0), tmp_path)
    model = baseline.load_model(tmp_path)
    normal = model.score(_clips(5, 1)[0])
    odd = _clips(6, 1)[0]
    odd[:, :, 0] += 0.5
    assert 0.4 < normal.mean() < 0.6
    assert 0.05 < (normal >= 0.9).mean() < 0.2
    assert model.score(odd).mean() > 0.95


def test_missing_artefact_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        baseline.load_model(tmp_path)
    with pytest.raises(FileNotFoundError, match="dance.baseline"):
        score.load(tmp_path, "baseline")


def test_score_load_rejects_an_unknown_scorer(tmp_path):
    with pytest.raises(ValueError):
        score.load(tmp_path, "other")


def test_no_nan_on_empty_bins(tmp_path):
    baseline.train(_clips(0), tmp_path)
    scores = baseline.load_model(tmp_path).score(np.zeros((10, 2, FEATURE_DIM), dtype=np.float32))
    assert np.isfinite(scores).all()

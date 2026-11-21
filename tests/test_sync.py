"""The training-free sync score."""

from __future__ import annotations

import numpy as np

from dance.contracts import FEATURE_DIM
from dance.sync import WINDOW_BINS, sync_score


def _series(bins: int, seed: int) -> np.ndarray:
    """A smooth periodic feature series, (bins, 13)."""
    rng = np.random.default_rng(seed)
    t = np.arange(bins)[:, None]
    return 0.5 + 0.3 * np.sin(2 * np.pi * t / 36 + rng.uniform(0, 6, 13)) * rng.uniform(0.5, 1, 13)


def _clip(series: list[np.ndarray]) -> np.ndarray:
    feats = np.zeros((len(series[0]), len(series), FEATURE_DIM), dtype=np.float32)
    for d, s in enumerate(series):
        feats[:, d, :13] = s
    return feats


def test_one_dancer_has_no_sync_score():
    assert sync_score(_clip([_series(100, 0)])) is None


def test_shape_range_and_window():
    assert WINDOW_BINS == 61                                   # 2 s of 33 ms bins
    sync = sync_score(_clip([_series(200, 0), _series(200, 1), _series(200, 2)]))
    assert sync.shape == (200, 3) and sync.dtype == np.float32
    assert np.isfinite(sync).all() and sync.min() >= 0 and sync.max() <= 2


def test_dancers_in_step_score_low_and_the_odd_one_scores_high():
    same = _series(300, 0)
    noise = np.random.default_rng(3).normal(0, 0.01, (3, 300, 13))
    odd = _series(300, 7)                                      # other phases: not in step
    sync = sync_score(_clip([same + noise[0], odd, same + noise[1], same + noise[2]]))
    late = sync[100:]
    assert late[:, [0, 2, 3]].mean() < 0.1
    assert late[:, 1].mean() > 0.6
    assert int(np.argmax(late.mean(axis=0))) == 1


def test_opposite_motion_scores_near_two():
    a = _series(200, 0)
    sync = sync_score(_clip([a, 1.0 - a]))
    assert sync[100:].mean() > 1.9


def test_a_frozen_dancer_among_moving_dancers_scores_one():
    a = _series(200, 0)
    frozen = np.full_like(a, 0.5)
    sync = sync_score(_clip([a, frozen, a.copy()]))
    assert np.allclose(sync[100:, 1], 1.0)
    assert sync[100:, 0].mean() < 0.01


def test_empty_bins_give_no_nan():
    sync = sync_score(np.zeros((50, 2, FEATURE_DIM), dtype=np.float32))
    assert np.isfinite(sync).all()

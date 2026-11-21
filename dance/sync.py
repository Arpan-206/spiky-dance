"""Sync score: how far a dancer is out of step with the other dancers.

It needs no training. For each dancer and each bin:

    sync = 1 - correlation(own feature series, median series of the other dancers)

The correlation covers a rolling window of the last 2 seconds. It is computed
for each of the 13 features (not the deltas) and then averaged.

    0   the dancer moves like the others
    1   no relation to the others
    2   the exact opposite of the others

The score needs 2 or more dancers. For 1 dancer, `sync_score` returns None.

Rules for a feature that does not change inside the window:
  - constant for the dancer and for the others: correlation 1 (both are still);
  - constant for only one of the two: correlation 0.
The first bins have a shorter window. With fewer than MIN_BINS bins the score is 1.
"""

from __future__ import annotations

import numpy as np

from .contracts import BIN_MS

WINDOW_S = 2.0
WINDOW_BINS = int(round(WINDOW_S * 1000 / BIN_MS))   # 61 bins of 33 ms
MIN_BINS = 3
BASE_DIM = 13
EPS = 1e-10


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Sum over the last `window` rows, for each row. Shorter at the start."""
    c = np.cumsum(x, axis=0, dtype=np.float64)
    out = c.copy()
    out[window:] = c[window:] - c[:-window]
    return out


def sync_score(features: np.ndarray, window_bins: int = WINDOW_BINS) -> np.ndarray | None:
    """(bins, dancers, >= 13) features -> (bins, dancers) float32 sync scores, 0 to 2.

    Returns None when there are fewer than 2 dancers.
    """
    features = np.asarray(features)
    if features.ndim != 3 or features.shape[2] < BASE_DIM:
        raise ValueError(f"features must have shape (bins, dancers, >= {BASE_DIM}), got {features.shape}")
    bins, dancers = features.shape[:2]
    if dancers < 2:
        return None
    base = features[..., :BASE_DIM].astype(np.float64)
    n = np.minimum(np.arange(1, bins + 1), window_bins).astype(np.float64)[:, None]
    out = np.ones((bins, dancers), dtype=np.float32)
    for d in range(dancers):
        a = base[:, d]                                            # (bins, 13)
        b = np.median(np.delete(base, d, axis=1), axis=1)         # (bins, 13)
        sa, sb = _rolling_sum(a, window_bins), _rolling_sum(b, window_bins)
        var_a = _rolling_sum(a * a, window_bins) - sa * sa / n
        var_b = _rolling_sum(b * b, window_bins) - sb * sb / n
        cov = _rolling_sum(a * b, window_bins) - sa * sb / n
        still_a, still_b = var_a <= EPS, var_b <= EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.sqrt(var_a * var_b)
        corr = np.where(still_a & still_b, 1.0, np.where(still_a | still_b, 0.0, corr))
        corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
        score = 1.0 - corr.mean(axis=1)
        score[: MIN_BINS - 1] = 1.0
        out[:, d] = score
    return out

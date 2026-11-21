"""Contracts between the parts of the `dance` package.

This file is a specification. The bodies here only raise NotImplementedError.
Only the main agent changes this file.

Where each contract is implemented:
    extract_features  -> dance.features.extract_features
    train             -> dance.train.train
    load_model, Model -> dance.spiky.load_model, dance.spiky.Model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURE_DIM = 26   # numbers per time bin per dancer: 13 features + 13 deltas
BIN_MS = 33        # length of one time bin, in milliseconds


def extract_features(binned: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Binned events + dancer boxes -> features.

    binned: float32, shape (n_bins, 2, H, W). The layout of eventify's
        `frame_stream`: event counts per pixel, channel 0 = OFF, channel 1 = ON.
        One bin is one BIN_MS window.
    boxes: int64, shape (dancers, 4) for all bins, or (n_bins, dancers, 4).
        Columns x0, y0, x1, y1 in sensor pixels. x1 and y1 are exclusive.

    Returns float32, shape (n_bins, dancers, FEATURE_DIM). No NaN, also on an
    empty bin. Feature order, with coordinates normalised to the box:
        0      rate        log-scaled event count, 0..1
        1      polarity    (ON - OFF) / count, -1..1, 0 on an empty bin
        2      centroid_x  0..1, 0.5 on an empty bin
        3      centroid_y  0..1, 0.5 on an empty bin
        4..12  grid        3x3 event counts, row-major, sum 1; all 0 on an empty bin
        13..25 deltas      feature[k] - feature[k - 1] for the 13 features above,
                           in the same order. All 0 for the first bin of `binned`.
                           Range -1..1, and -2..2 for the polarity delta (index 14).
    """
    raise NotImplementedError


def train(feature_clips: list[np.ndarray], out_dir: str | Path) -> None:
    """Fit the spiking network and its calibration on normal clips only.

    feature_clips: one array per clean clip, each float32 with shape
        (bins_i, dancers_i, FEATURE_DIM), as returned by `extract_features`.
        Bins are in time order. Clips can differ in bins and in dancers.
    out_dir: folder for all artefacts (weights, whitening, baselines, ECDF,
        threshold). The folder is created. Old files in it are overwritten.

    Everything is fitted again from `feature_clips`. No artefact of an earlier
    model is read. Membrane state is carried across the bins of one clip. It is
    reset between clips and between dancers.
    """
    raise NotImplementedError


def load_model(model_dir: str | Path) -> "Model":
    """Load the artefacts that `train` saved in `model_dir`.

    Raises FileNotFoundError if the folder or any artefact is missing. There is
    no fallback and no mock scoring mode.
    """
    raise NotImplementedError


class Model:
    """A trained scorer."""

    def score(self, features: np.ndarray) -> np.ndarray:
        """Features of one clip -> unusualness scores.

        features: float32, shape (bins, dancers, FEATURE_DIM), bins in time order.

        Returns float32, shape (bins, dancers), values 0 to 1. A score is the
        percentile of the bin against the training clips: 0.5 is the training
        median, and 0.90 is the p90 threshold. Membrane state is carried across
        bins. It starts from zero on each call and is separate for each dancer.
        The same input always gives the same output.
        """
        raise NotImplementedError

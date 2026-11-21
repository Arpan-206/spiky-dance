"""Baseline scorer: a plain statistical distance, with no spiking network.

It implements the same contract as the spiking model (`dance/contracts.py`):
`train(feature_clips, out_dir)`, `load_model(model_dir)`, `Model.score`.

Method:
  1. Fit the mean and the covariance of the features of the clean clips.
  2. The raw score of a bin is its Mahalanobis distance to that mean. The
     Mahalanobis distance is the distance in units of the normal spread, with
     the links between features removed.
  3. The final score is the percentile of the raw score against the raw
     scores of the training bins (the same ECDF step as the spiking model).

Each bin is scored alone. The scorer keeps no state between bins.

Train: python -m dance.baseline CLIPS_DIR [--out models/dance] [--dancers N] [--max-seconds S]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .contracts import FEATURE_DIM

ARTEFACT = "baseline.npz"
RIDGE = 1e-3   # added to the covariance diagonal, so the inverse always exists


def _rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3 or features.shape[2] != FEATURE_DIM:
        raise ValueError(f"features must have shape (bins, dancers, {FEATURE_DIM}), got {features.shape}")
    return features.reshape(-1, FEATURE_DIM)


def _distance(rows: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    z = rows - mean
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z, precision, z), 0.0))


def train(feature_clips: list[np.ndarray], out_dir: str | Path) -> None:
    """Fit the baseline on clean clips and save `baseline.npz` in `out_dir`."""
    rows = [_rows(clip) for clip in feature_clips]
    rows = [r for r in rows if len(r)]
    if not rows:
        raise ValueError("no training bins: feature_clips is empty")
    x = np.concatenate(rows)
    if len(x) < 2 * FEATURE_DIM:
        raise ValueError(f"too few training bins: {len(x)}, need at least {2 * FEATURE_DIM}")
    mean = x.mean(axis=0)
    cov = np.cov((x - mean).T) + RIDGE * np.eye(FEATURE_DIM)
    precision = np.linalg.inv(cov)
    reference = np.sort(_distance(x, mean, precision))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / ARTEFACT,
        mean=mean,
        precision=precision,
        reference=reference,
        n_clips=np.int64(len(rows)),
    )


class BaselineModel:
    """A fitted baseline scorer. See `dance.contracts.Model`."""

    def __init__(self, mean: np.ndarray, precision: np.ndarray, reference: np.ndarray):
        self.mean, self.precision, self.reference = mean, precision, reference

    def score(self, features: np.ndarray) -> np.ndarray:
        """(bins, dancers, FEATURE_DIM) -> (bins, dancers) float32 percentiles, 0 to 1."""
        shape = np.asarray(features).shape[:2]
        dist = _distance(_rows(features), self.mean, self.precision)
        pct = np.searchsorted(self.reference, dist, side="right") / len(self.reference)
        return pct.reshape(shape).astype(np.float32)


def load_model(model_dir: str | Path) -> BaselineModel:
    """Load `baseline.npz`. Raise FileNotFoundError if it is missing."""
    path = Path(model_dir) / ARTEFACT
    if not path.exists():
        raise FileNotFoundError(f"baseline artefact not found: {path}")
    with np.load(path) as data:
        mean, precision, reference = data["mean"], data["precision"], data["reference"]
    if mean.shape != (FEATURE_DIM,):
        raise ValueError(
            f"{path} was fitted on {mean.shape[0]} features, but FEATURE_DIM is {FEATURE_DIM}. Train again."
        )
    return BaselineModel(mean, precision, reference)


def main(argv: list[str] | None = None) -> None:
    from . import stream
    from .features import clip_features, find_poses

    p = argparse.ArgumentParser(description="Fit the baseline scorer on a folder of clean clips")
    p.add_argument("clips_dir")
    p.add_argument("--out", default="models/dance")
    p.add_argument("--dancers", type=int, default=1, help="number of vertical strips when there are no poses")
    p.add_argument("--max-seconds", type=float, default=None)
    args = p.parse_args(argv)

    suffixes = stream.VIDEO_SUFFIXES | stream.HDF5_SUFFIXES
    clips = sorted(f for f in Path(args.clips_dir).iterdir() if f.suffix.lower() in suffixes)
    if not clips:
        raise SystemExit(f"error: no clips in {args.clips_dir}")
    feature_clips = []
    for clip in clips:
        poses_path = find_poses(clip)
        poses = np.load(poses_path) if poses_path else None
        feats, _ = clip_features(clip, poses, args.dancers, args.max_seconds)
        feature_clips.append(feats)
        print(f"{clip.name}: {feats.shape[0]} bins, {feats.shape[1]} dancer(s)")
    train(feature_clips, args.out)
    print(f"baseline fitted on {len(clips)} clips -> {Path(args.out) / ARTEFACT}")


if __name__ == "__main__":
    main()

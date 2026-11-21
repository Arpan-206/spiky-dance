"""Whitening and encoding of feature vectors for the spiking network.

PROVENANCE: copied from github.com/Arpan-206/mr-spiky, file `src/encode.py`.
Unchanged: the ZCA whitening maths of `fit_whitening` (centre, covariance,
symmetric inverse square root, epsilon 1e-3, rescale by 2 * q99), the
`whiten` formula `(W @ (x - mean) + 0.5).clamp(0, 1)`, DEFAULT_STEPS = 25,
rate coding with `snntorch.spikegen.rate`, graded-current sequence mode.

Changes for spiky-dance:
  1. The feature source. `normalize` now maps the dance features of
     `dance.contracts` to 0..1 (only polarity needs a change: -1..1 -> 0..1).
     The original divided code-review features by fixed caps.
  2. The functions take arrays or tensors of shape (..., 13), not lists.
  3. REMOVED `_load_whitening`, `_WHITENING_PATH` and the cache. The original
     read `models/whitening.pt` from a fixed path when no transform was given.
     Here `whiten` has no default: the caller must pass the transform that
     `dance.train.train` fitted. No function in this file reads a file.
  5. The dance features have 13 values and their 13 bin-to-bin deltas
     (FEATURE_DIM = 26). `normalize` maps a delta d to (d + 1) / 2, and the
     polarity delta to (d + 2) / 4.
  6. OPTIONAL, OFF BY DEFAULT: `encoding="onoff"`. The default "shift" is the
     formula of Mr. Spiky. It adds 0.5 to each whitened value, so each input
     has the mean 0.5 and a deviation below the mean LOWERS the input. With
     "onoff", each whitened value y gives two inputs, relu(y) and relu(-y),
     times ONOFF_GAIN, clamped to 0..1. Then the inputs are near 0 for a
     normal bin, and a deviation in EITHER direction raises an input.
     This is a change to Mr. Spiky's encoding. It is not one of the three
     network changes that the user allowed, thus it is not the default.
     The encoding name is saved in the whitening artefact.
  4. `encode_sequence` keeps the dancers as the batch dimension (the original
     had batch 1). `encode_rate` is `encode_batch` for a clip: it rate-codes
     each time bin and keeps the bins in time order.
"""

from __future__ import annotations

import numpy as np
import torch
from snntorch import spikegen

from .model import NUM_FEATURES

DEFAULT_STEPS = 25

BASE_FEATURES = 13  # dance.contracts: rate, polarity, centroid x, y, 3x3 grid
POLARITY_INDEX = 1  # dance.contracts: feature 1 is polarity, -1..1
if NUM_FEATURES not in (BASE_FEATURES, 2 * BASE_FEATURES):
    raise ImportError(
        f"dance.contracts.FEATURE_DIM = {NUM_FEATURES}: dance.spiky.encode.normalize "
        f"knows only {BASE_FEATURES} features, or {BASE_FEATURES} features + their deltas"
    )
HAS_DELTAS = NUM_FEATURES == 2 * BASE_FEATURES

ENCODINGS = ("shift", "onoff")
ONOFF_GAIN = 2.0  # fit_whitening puts q99 of |y| at 0.5: the gain maps it to 1.0


def encoded_dim(encoding: str) -> int:
    """Number of network inputs for an encoding."""
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
    return NUM_FEATURES if encoding == "shift" else 2 * NUM_FEATURES
_QUANTILE_MAX = 8_000_000  # torch.quantile accepts at most 16 million values


def normalize(features: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Dance features (..., FEATURE_DIM) -> float32 tensor with all values in 0..1."""
    x = torch.as_tensor(np.asarray(features, dtype=np.float32)).clone()
    if x.shape[-1] != NUM_FEATURES:
        raise ValueError(f"last dimension must be {NUM_FEATURES}, got shape {tuple(x.shape)}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("features contain NaN or infinite values")
    x[..., POLARITY_INDEX] = (x[..., POLARITY_INDEX] + 1.0) * 0.5
    if HAS_DELTAS:
        # Deltas: -1..1 -> 0..1. Polarity delta: -2..2 -> 0..1. No change = 0.5.
        d = x[..., BASE_FEATURES:]
        d.add_(1.0).mul_(0.5)
        k = BASE_FEATURES + POLARITY_INDEX
        x[..., k] = (x[..., k] - 0.5) * 0.5 + 0.5
    return x.clamp(0.0, 1.0)


def fit_whitening(
    normalized: torch.Tensor, epsilon: float = 1e-3, encoding: str = "shift"
) -> dict:
    """Compute a ZCA whitening transform from normalized feature vectors.

    normalized: (N, F), the output of `normalize` with all bins of all dancers
    of all clips as rows.

    Steps: center, compute covariance, take symmetric inverse square root.
    Applying W · (x - mean) yields x_white with identity covariance — every
    output dimension is uncorrelated with every other and has unit variance.

    The result is *rescaled* back into a roughly [0, 1] range (divided by
    2·q99 of the absolute whitened values) so it's a valid rate-coding input
    for the SNN.
    """
    X = normalized.reshape(-1, NUM_FEATURES).to(torch.float32)                # (N, F)
    mean = X.mean(dim=0)                                                     # (F,)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)                              # (F, F)
    # ZCA: W = Cov^(-1/2). Compute via eigen-decomposition (symmetric matrix).
    eigvals, eigvecs = torch.linalg.eigh(cov)                                # ascending
    inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals.clamp(min=epsilon)))
    W = eigvecs @ inv_sqrt @ eigvecs.T                                       # (F, F)
    # Rescale so whitened outputs land back in a similar range to normalized
    # ones (Bernoulli rate-coding wants inputs in [0, 1]).
    flat = (Xc @ W.T).abs().flatten()
    if flat.numel() > _QUANTILE_MAX:
        # torch.quantile has an input size limit: use an evenly spaced subset.
        flat = flat[:: flat.numel() // _QUANTILE_MAX + 1]
    scale = float(flat.quantile(0.99)) * 2.0
    W = W / max(scale, 1e-6)
    encoded_dim(encoding)  # checks the name
    return {"mean": mean, "W": W, "encoding": encoding}


def whiten(normalized: torch.Tensor, transform: dict) -> torch.Tensor:
    """Apply the whitening transform to normalized features (..., F).

    `transform` is required, with its "encoding" key. There is no default and
    no file is read. Returns (..., encoded_dim(encoding)) with values in 0..1.
    """
    encoding = transform["encoding"]
    x = normalized.to(torch.float32) - transform["mean"]
    y = x @ transform["W"].T
    if encoding == "shift":
        # Mr. Spiky: shift + clamp back into [0, 1] (whitened values are ~ mean-zero).
        return (y + 0.5).clamp(0.0, 1.0)
    if encoding == "onoff":
        return torch.cat([torch.relu(y), torch.relu(-y)], dim=-1).mul(ONOFF_GAIN).clamp(0.0, 1.0)
    raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")


def encode_sequence(features: np.ndarray | torch.Tensor, transform: dict) -> torch.Tensor:
    """Sequence encoding: each bin's whitened values are the graded input
    current at one timestep. (bins, dancers, F) -> (T=bins, B=dancers, F).

    The SNN then processes the bins *in order*, with LIF membrane state
    carrying context from bin i-1 into bin i. Each dancer is one batch item,
    thus each dancer has a separate membrane state.
    """
    return whiten(normalize(features), transform)


def encode_rate(
    features: np.ndarray | torch.Tensor,
    transform: dict,
    steps_per_bin: int,
) -> torch.Tensor:
    """Rate-coded encoding of one clip, in time order.

    (bins, dancers, F) -> (bins * steps_per_bin, dancers, F) binary spikes.
    Each bin becomes `steps_per_bin` Bernoulli samples (`spikegen.rate`, as in
    Mr. Spiky's `encode_batch`). The samples of bin i come before bin i + 1.
    Uses the global torch random generator.
    """
    encoded = whiten(normalize(features), transform)             # (T, B, F)
    T, B, F = encoded.shape
    if T == 0:
        return torch.zeros(0, B, F, dtype=torch.float32)
    spikes = spikegen.rate(encoded, num_steps=steps_per_bin)     # (K, T, B, F)
    return spikes.permute(1, 0, 2, 3).reshape(T * steps_per_bin, B, F)

"""`Model` and `load_model`: the trained scorer of spiky-dance.

PROVENANCE: this file is new. It is NOT copied from Mr. Spiky. It replaces
`src/infer.py` of github.com/Arpan-206/mr-spiky, which is not vendored.
The score path is the sequence-mode path of Mr. Spiky: whiten -> SparseSpikyNet ->
membrane attribution above the per-neuron baselines -> ECDF percentile.

There is no substitute scorer here. `load_model` raises FileNotFoundError when
the folder or any artefact is missing. No weights, whitening, baselines or
thresholds are built in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .calibrate import sequence_raw_scores
from .encode import encoded_dim
from .model import NUM_FEATURES, ecdf_rescale_tensor
from .sparse import SparseSpikyNet

WEIGHTS_FILE = "snn_weights.pt"
WHITENING_FILE = "whitening.pt"
BASELINES_FILE = "snn_baselines.pt"
ECDF_FILE = "snn_ecdf.pt"
THRESHOLD_FILE = "threshold.json"
ARTEFACT_FILES = (WEIGHTS_FILE, WHITENING_FILE, BASELINES_FILE, ECDF_FILE, THRESHOLD_FILE)


class Model:
    """A trained scorer. Make one with `load_model`."""

    def __init__(
        self,
        net: SparseSpikyNet,
        whitening: dict[str, torch.Tensor],
        hidden_baseline: torch.Tensor,
        output_baseline: torch.Tensor,
        ecdf: torch.Tensor,
        meta: dict,
    ) -> None:
        if ecdf.numel() == 0:
            raise ValueError("the ECDF reference is empty")
        self.net = net.eval()
        self.whitening = whitening
        self.hidden_baseline = hidden_baseline
        self.output_baseline = output_baseline
        self.ecdf = ecdf
        self.meta = meta
        self.threshold = float(meta["threshold"])

    def _check(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 3 or features.shape[2] != NUM_FEATURES:
            raise ValueError(
                f"features must have shape (bins, dancers, {NUM_FEATURES}), got {features.shape}"
            )
        return features

    def raw_score(self, features: np.ndarray) -> np.ndarray:
        """Features of one clip -> raw scores before the ECDF, shape (bins, dancers)."""
        raw = sequence_raw_scores(
            self.net, self._check(features), self.whitening,
            self.hidden_baseline, self.output_baseline,
        )
        return raw.numpy().astype(np.float32)

    def score(self, features: np.ndarray) -> np.ndarray:
        """Features of one clip -> unusualness scores.

        features: float32, shape (bins, dancers, FEATURE_DIM), bins in time order.
        Returns float32, shape (bins, dancers), values 0 to 1: the percentile
        of the bin against the training clips. Membrane state starts from zero
        on each call and is separate for each dancer. Deterministic.
        """
        raw = sequence_raw_scores(
            self.net, self._check(features), self.whitening,
            self.hidden_baseline, self.output_baseline,
        )
        return ecdf_rescale_tensor(raw, self.ecdf).numpy().astype(np.float32)


def load_model(model_dir: str | Path) -> Model:
    """Load the artefacts that `dance.train.train` saved in `model_dir`.

    Raises FileNotFoundError if the folder or any artefact is missing.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model folder not found: {model_dir}")
    missing = [name for name in ARTEFACT_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"model folder {model_dir} has missing artefacts: {', '.join(missing)}. "
            "Run dance.train again."
        )

    whitening = torch.load(model_dir / WHITENING_FILE, weights_only=True)
    net = SparseSpikyNet(input_dim=encoded_dim(whitening["encoding"]))
    net.load_state_dict(torch.load(model_dir / WEIGHTS_FILE, weights_only=True))
    net.eval()
    baselines = torch.load(model_dir / BASELINES_FILE, weights_only=True)
    ecdf = torch.load(model_dir / ECDF_FILE, weights_only=True)
    meta = json.loads((model_dir / THRESHOLD_FILE).read_text())
    return Model(
        net=net,
        whitening={"mean": whitening["mean"], "W": whitening["W"], "encoding": whitening["encoding"]},
        hidden_baseline=baselines["hidden"],
        output_baseline=baselines["output"],
        ecdf=ecdf["sorted_raw_scores"],
        meta=meta,
    )

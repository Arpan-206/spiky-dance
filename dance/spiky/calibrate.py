"""Calibration of Mr. Spiky on the normal corpus: baselines, ECDF, threshold.

ECDF = empirical cumulative distribution function: the sorted raw scores of
the normal corpus. A new raw score becomes its percentile rank in that list.

PROVENANCE: the method comes from github.com/Arpan-206/mr-spiky, file
`src/calibrate.py` (`_compute_baselines`, `_sequence_line_intensities`,
`_batch_intensities`, `_stats`, and the ECDF + p90 part of `calibrate`).
Unchanged: the logic. Baseline = mean firing rate per neuron in sequence mode
over the normal corpus. Raw score = `per_timestep_attribution` with membrane
traces and those baselines. ECDF reference = the sorted raw scores of the
normal corpus. Threshold = 0.90 in percentile space.

Changes for spiky-dance:
  1. The functions take feature arrays (bins, dancers, 13), not source-code
     strings. The original functions called the code-review feature extractor.
  2. The whitening transform is an explicit argument. No file is read.
  3. Dancers are the batch dimension. A baseline is the mean over bins and
     dancers. The original had batch 1.
  4b. The network is `SparseSpikyNet` (see `sparse.py`). Each neuron has its
     own threshold. `per_timestep_attribution` compares a membrane with the
     fixed threshold 1.0. Thus the membranes are shifted by each neuron's
     threshold offset before the call: the attribution code is not changed,
     and "at threshold" means the neuron's own threshold.
  4d. Added `run_sequence`: each dancer is run in a separate pass.
  4c. Added `activity`: the fraction of neurons that fire per bin.
  4. Removed the file paths, the labelled-data evaluation blocks, the
     "labeled accuracy sweep" fallback and the built-in contrast pairs.
"""

from __future__ import annotations

import numpy as np
import torch

from .encode import encode_sequence
from .model import LIF1_THRESHOLD, LIF2_THRESHOLD, SNNOutput, per_timestep_attribution
from .sparse import SparseSpikyNet

# After ECDF rescale, the threshold is the percentile cutoff.
# p90 = "top 10 % most unusual for the normal clips".
THRESHOLD = 0.90


def run_sequence(net: SparseSpikyNet, seq: torch.Tensor) -> SNNOutput:
    """Run (T, B, F) in sequence mode, each dancer (batch item) in a separate pass.

    A batched pass gives the same values up to float rounding. But the top-k
    choice between near-equal neurons can change with that rounding. With
    separate passes, the result for one dancer does not depend on the others.
    """
    with torch.no_grad():
        outs = [net(seq[:, d:d + 1]) for d in range(seq.shape[1])]
    fields = ("hidden_spikes", "output_spikes", "hidden_mem", "output_mem")
    return SNNOutput(**{f: torch.cat([getattr(o, f) for o in outs], dim=1) for f in fields})


def sequence_raw_scores(
    net: SparseSpikyNet,
    features: np.ndarray | torch.Tensor,
    whitening: dict[str, torch.Tensor],
    hidden_baseline: torch.Tensor | None = None,
    output_baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one clip through the SNN in *sequence mode*: one graded-current
    step per bin, one forward pass per clip. Returns raw scores (bins, dancers).

    Calibration and scoring both use this function, thus they are on the same
    scale. The membrane state starts from zero and is separate for each dancer.
    """
    seq = encode_sequence(features, whitening)          # (T, B, F)
    if seq.shape[0] == 0 or seq.shape[1] == 0:
        return torch.zeros(seq.shape[0], seq.shape[1], dtype=torch.float32)
    out = run_sequence(net, seq)
    thr1, thr2 = net.thresholds()
    return per_timestep_attribution(
        out.hidden_spikes, out.output_spikes,
        hidden_baseline=hidden_baseline,
        output_baseline=output_baseline,
        hidden_mem=out.hidden_mem - (thr1 - LIF1_THRESHOLD),
        output_mem=out.output_mem - (thr2 - LIF2_THRESHOLD),
    )


def compute_baselines(
    net: SparseSpikyNet,
    clips: list[np.ndarray],
    whitening: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean per-neuron firing rate across the corpus. Returns (hidden, output)
    both shape (H,) / (O,), values in [0, 1]."""
    h_sums: torch.Tensor | None = None
    o_sums: torch.Tensor | None = None
    total_steps = 0
    for clip in clips:
        seq = encode_sequence(clip, whitening)
        if seq.shape[0] == 0 or seq.shape[1] == 0:
            continue
        out = run_sequence(net, seq)
        # out.hidden_spikes: (T, B, H). Sum over T and B; accumulate per-neuron.
        h = out.hidden_spikes.sum(dim=(0, 1))  # (H,)
        o = out.output_spikes.sum(dim=(0, 1))  # (O,)
        h_sums = h if h_sums is None else h_sums + h
        o_sums = o if o_sums is None else o_sums + o
        total_steps += out.hidden_spikes.shape[0] * out.hidden_spikes.shape[1]
    if total_steps == 0 or h_sums is None or o_sums is None:
        raise RuntimeError("no baseline data — check corpus")
    return h_sums / total_steps, o_sums / total_steps


def corpus_raw_scores(
    net: SparseSpikyNet,
    clips: list[np.ndarray],
    whitening: dict[str, torch.Tensor],
    hidden_baseline: torch.Tensor,
    output_baseline: torch.Tensor,
) -> torch.Tensor:
    """All raw scores of the corpus as one flat tensor (one value per bin per dancer)."""
    parts = [
        sequence_raw_scores(net, clip, whitening, hidden_baseline, output_baseline).flatten()
        for clip in clips
    ]
    return torch.cat(parts) if parts else torch.zeros(0)


def activity(
    net: SparseSpikyNet,
    clips: list[np.ndarray],
    whitening: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Fraction of neurons that fire per bin in sequence mode, per layer.

    Also the smallest and the largest firing rate of a single neuron.
    """
    hidden, output = compute_baselines(net, clips, whitening)
    return {
        "hidden": float(hidden.mean()),
        "output": float(output.mean()),
        "hidden_neuron_min": float(hidden.min()),
        "hidden_neuron_max": float(hidden.max()),
        "output_neuron_min": float(output.min()),
        "output_neuron_max": float(output.max()),
    }


def stats(xs: torch.Tensor) -> dict[str, float]:
    """Summary of raw scores. Same keys as Mr. Spiky's `_stats`, plus min, p90, p99, max."""
    if xs.numel() == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    s = xs.flatten().sort().values

    def q(p: float) -> float:
        return float(s[min(int(p * (s.numel() - 1) + 0.5), s.numel() - 1)])

    return {
        "n": int(s.numel()),
        "mean": float(s.mean()),
        "std": float(s.std(unbiased=False)),
        "min": float(s[0]),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": float(s[-1]),
    }

"""The STDP rule of Mr. Spiky and its hyperparameters.

STDP = spike-timing-dependent plasticity: a local, unsupervised weight update
that depends on the order of input (pre) and output (post) spikes.

PROVENANCE: copied from github.com/Arpan-206/mr-spiky, file `src/train_stdp.py`.
Unchanged: `_stdp_update` and A_PLUS, A_MINUS, TAU_TRACE, W_MIN, W_MAX,
WEIGHT_DECAY, EPOCHS, BATCH_SIZE.

STDP rule:
    Δw⁺ = +A⁺ · pre_trace · post_spike                 (potentiation)
    Δw⁻ = -A⁻ · post_trace · pre_spike · current_w     (multiplicative depression)

Note from the original, still true: STDP on dense graded-current sequence
input saturated fc2 to zero. Thus training uses rate-coded binary spikes.
Calibration and scoring use graded-current sequence mode.

Changes for spiky-dance:
  1. Removed the corpus loaders, the file paths and the `train` loop of the
     code-review model. The dance training loop is in `dance/train.py`.
  2. BATCH_SIZE is kept for reference only. In spiky-dance the batch is the
     dancers of one clip.
"""

from __future__ import annotations

import torch

# STDP hyperparameters. Depression dominates potentiation and scales with the
# current weight (multiplicative depression) so weights don't stampede to W_MAX.
A_PLUS = 0.005
A_MINUS = 0.020
TAU_TRACE = 20.0
W_MIN = 0.0
W_MAX = 1.0
WEIGHT_DECAY = 0.001
EPOCHS = 6
BATCH_SIZE = 16


def _stdp_update(
    layer_weight: torch.Tensor,
    pre_spikes: torch.Tensor,   # (T, B, in) — B is always 1 in sequence mode
    post_spikes: torch.Tensor,  # (T, B, out)
) -> None:
    """Pair-based STDP, in-place. layer_weight shape: (out, in)."""
    T, B, in_dim = pre_spikes.shape
    _, _, out_dim = post_spikes.shape
    decay = float(torch.exp(torch.tensor(-1.0 / TAU_TRACE)))

    pre_trace = torch.zeros(B, in_dim)
    post_trace = torch.zeros(B, out_dim)
    dw = torch.zeros_like(layer_weight)

    for t in range(T):
        pre_trace = pre_trace * decay + pre_spikes[t]
        post_trace = post_trace * decay + post_spikes[t]

        pot = torch.bmm(post_spikes[t].unsqueeze(2), pre_trace.unsqueeze(1)).sum(0)
        dep = torch.bmm(post_trace.unsqueeze(2), pre_spikes[t].unsqueeze(1)).sum(0)
        dw += A_PLUS * pot - A_MINUS * dep * layer_weight

    with torch.no_grad():
        layer_weight.mul_(1.0 - WEIGHT_DECAY)
        layer_weight.add_(dw / max(B, 1))
        layer_weight.clamp_(W_MIN, W_MAX)

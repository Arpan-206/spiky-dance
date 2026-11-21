"""The spiking network of Mr. Spiky, and its attribution and ECDF functions.

PROVENANCE: copied from github.com/Arpan-206/mr-spiky, file `src/model.py`.
Unchanged: `SNNOutput`, `SpikyNet` (layers, sizes, BETA, thresholds, weight
init, `forward`), `per_timestep_attribution`, `ecdf_rescale`.

Changes for spiky-dance:
  1. `NUM_FEATURES` comes from `dance.contracts.FEATURE_DIM` (13). The original
     imported it from the code-review feature module.
  2. Removed `temporal_spike_attribution`. Only the line-independent path of
     the original used it. That path is not vendored.
  3. Added `init_state` and `forward_stateful`. They run the same operations
     as `SpikyNet.forward`, but take and return the membrane state. The class
     itself is not changed.
  4. Added `ecdf_rescale_tensor`. It gives the same values as `ecdf_rescale`
     for a whole tensor in one call.

This package has no substitute scorer and no built-in weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from snntorch import surrogate
from snntorch import Leaky

from dance.contracts import FEATURE_DIM

NUM_FEATURES = FEATURE_DIM

HIDDEN = 128  # bumped 16 → 64 → 128. Bigger hidden layer gives STDP more
              # neurons to specialize, so different neurons can learn different
              # 'flavors of taste' — makes the multi-axis output more selective
              # and provides more distinct firing patterns for context.
OUTPUT = 32   # bumped 4 → 16 → 32 for the same reason
BETA = 0.9    # membrane decay
LIF1_THRESHOLD = 1.0
LIF2_THRESHOLD = 1.0


@dataclass
class SNNOutput:
    hidden_spikes: torch.Tensor  # (T, B, HIDDEN)  — binary
    output_spikes: torch.Tensor  # (T, B, OUTPUT)  — binary
    hidden_mem: torch.Tensor     # (T, B, HIDDEN)  — continuous membrane
    output_mem: torch.Tensor     # (T, B, OUTPUT)  — continuous membrane


class SpikyNet(nn.Module):
    def __init__(self, input_dim: int = NUM_FEATURES, hidden: int = HIDDEN, output: int = OUTPUT):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(input_dim, hidden, bias=False)
        self.lif1 = Leaky(beta=BETA, spike_grad=spike_grad, threshold=LIF1_THRESHOLD)
        self.fc2 = nn.Linear(hidden, output, bias=False)
        self.lif2 = Leaky(beta=BETA, spike_grad=spike_grad, threshold=LIF2_THRESHOLD)

        # STDP-style init: small non-negative weights so early training is stable.
        nn.init.uniform_(self.fc1.weight, 0.0, 0.5)
        nn.init.uniform_(self.fc2.weight, 0.0, 0.5)

    def forward(self, spikes_in: torch.Tensor) -> SNNOutput:
        """spikes_in: (T, B, F). Returns SNNOutput with spike traces + membrane
        traces. Membranes are the continuous 'about-to-fire' potentials — used
        for smooth per-line attribution that doesn't collapse to discrete
        firing-rate values."""
        T, B, _ = spikes_in.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        h_spk_trace = []
        o_spk_trace = []
        h_mem_trace = []
        o_mem_trace = []
        for t in range(T):
            cur1 = self.fc1(spikes_in[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            h_spk_trace.append(spk1)
            o_spk_trace.append(spk2)
            h_mem_trace.append(mem1)
            o_mem_trace.append(mem2)

        return SNNOutput(
            hidden_spikes=torch.stack(h_spk_trace, dim=0),
            output_spikes=torch.stack(o_spk_trace, dim=0),
            hidden_mem=torch.stack(h_mem_trace, dim=0),
            output_mem=torch.stack(o_mem_trace, dim=0),
        )


# ---- Addition for spiky-dance (not in Mr. Spiky) ---------------------------

def init_state(net: SpikyNet) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero membrane state for both layers, as `SpikyNet.forward` starts with."""
    return net.lif1.init_leaky(), net.lif2.init_leaky()


def forward_stateful(
    net: SpikyNet,
    spikes_in: torch.Tensor,
    mem1: torch.Tensor,
    mem2: torch.Tensor,
) -> tuple[SNNOutput, torch.Tensor, torch.Tensor]:
    """Run `spikes_in` (T, B, F) from the given membrane state.

    The operations are the same as in `SpikyNet.forward`. The only difference:
    the membranes come in as arguments and go out as results. Thus a long
    sequence can be run in parts without a reset between the parts.
    T must be at least 1. Returns (SNNOutput, mem1, mem2).
    """
    T = spikes_in.shape[0]
    h_spk_trace = []
    o_spk_trace = []
    h_mem_trace = []
    o_mem_trace = []
    for t in range(T):
        cur1 = net.fc1(spikes_in[t])
        spk1, mem1 = net.lif1(cur1, mem1)
        cur2 = net.fc2(spk1)
        spk2, mem2 = net.lif2(cur2, mem2)
        h_spk_trace.append(spk1)
        o_spk_trace.append(spk2)
        h_mem_trace.append(mem1)
        o_mem_trace.append(mem2)
    out = SNNOutput(
        hidden_spikes=torch.stack(h_spk_trace, dim=0),
        output_spikes=torch.stack(o_spk_trace, dim=0),
        hidden_mem=torch.stack(h_mem_trace, dim=0),
        output_mem=torch.stack(o_mem_trace, dim=0),
    )
    return out, mem1, mem2


# ---- Vendored from Mr. Spiky, unchanged ------------------------------------

def ecdf_rescale(raw_scores: list[float], sorted_ref: torch.Tensor) -> list[float]:
    """Map raw SNN scores to their percentile rank in a reference distribution.

    `sorted_ref` is the sorted array of raw scores over the senior corpus.
    A raw score at the 50th-percentile of the ref becomes 0.5; at the 99th,
    0.99; below the lowest, 0.0. This turns the discrete/saturated raw output
    into a smooth [0,1] scale where the threshold means 'top-X% for senior code.'
    """
    if not raw_scores or sorted_ref.numel() == 0:
        return raw_scores
    n = float(sorted_ref.numel())
    out: list[float] = []
    for s in raw_scores:
        # right-search position gives count of ref-scores <= s
        pos = float(torch.searchsorted(sorted_ref, torch.tensor(s), right=True))
        out.append(max(0.0, min(1.0, pos / n)))
    return out


# ---- Addition for spiky-dance (not in Mr. Spiky) ---------------------------

def ecdf_rescale_tensor(raw_scores: torch.Tensor, sorted_ref: torch.Tensor) -> torch.Tensor:
    """`ecdf_rescale` for a tensor of any shape, in one call. Same values."""
    n = float(sorted_ref.numel())
    pos = torch.searchsorted(sorted_ref, raw_scores.contiguous(), right=True)
    return (pos.to(torch.float32) / n).clamp(0.0, 1.0)


# ---- Vendored from Mr. Spiky, unchanged ------------------------------------

def per_timestep_attribution(
    hidden_spikes: torch.Tensor,
    output_spikes: torch.Tensor | None = None,
    hidden_baseline: torch.Tensor | None = None,
    output_baseline: torch.Tensor | None = None,
    hidden_mem: torch.Tensor | None = None,
    output_mem: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sequence-mode attribution: one score per timestep (line).

    If membrane traces are given (hidden_mem, output_mem), we use those
    directly — membrane potentials are continuous, so scores don't collapse
    to the ~5 discrete values that binary spike averages produce. This is
    what makes per-line scores actually distribute smoothly across [0, 1].

    Membrane is *how close the neuron got to firing* — a strictly richer
    signal than the binary spike output. A neuron that hovers at 0.9 threshold
    but doesn't quite spike is still "excited by this line," and the
    membrane captures that.

    Fallback: if only spikes are given, we compute the mean-firing attribution
    (used to be the primary path). Still supported for legacy calls.

    hidden_baseline / output_baseline: per-neuron mean firing rate on the
    training corpus. Score is how much this line's activation exceeds those
    baselines — the SNN's "unusual for senior code" signal.

    Returns (T, B) in [0, 1].
    """
    T = hidden_spikes.shape[0]
    if T == 0:
        return torch.zeros(T, hidden_spikes.shape[1])

    if hidden_mem is not None:
        # Membrane potentials can range above 1.0 (they accumulate). Squash
        # via sigmoid centered at the LIF threshold so 0.5 = "at threshold."
        h_activation = torch.sigmoid(4.0 * (hidden_mem - LIF1_THRESHOLD))  # (T,B,H)
        if hidden_baseline is not None:
            # Baseline is a per-neuron firing rate; convert to same
            # sigmoid-activation space by using it as a scalar shift.
            base = hidden_baseline.view(1, 1, -1).clamp(0.0, 1.0)
            h_activation = (h_activation - base).clamp(min=0.0) / (1.0 - base + 1e-6)
        h_score = h_activation.mean(dim=2)  # (T, B)
        if output_mem is not None:
            o_activation = torch.sigmoid(4.0 * (output_mem - LIF2_THRESHOLD))
            if output_baseline is not None:
                base = output_baseline.view(1, 1, -1).clamp(0.0, 1.0)
                o_activation = (o_activation - base).clamp(min=0.0) / (1.0 - base + 1e-6)
            o_score = o_activation.mean(dim=2)
            return (0.5 * h_score + 0.5 * o_score).clamp(0.0, 1.0)
        return h_score.clamp(0.0, 1.0)

    # Legacy spike-only path.
    if hidden_baseline is not None:
        excess = (hidden_spikes - hidden_baseline.view(1, 1, -1)).clamp(min=0.0)
        max_excess = (1.0 - hidden_baseline).clamp(min=1e-6)
        hidden_score = (excess / max_excess.view(1, 1, -1)).mean(dim=2)
        if output_spikes is not None and output_baseline is not None:
            o_excess = (output_spikes - output_baseline.view(1, 1, -1)).clamp(min=0.0)
            o_max = (1.0 - output_baseline).clamp(min=1e-6)
            output_score = (o_excess / o_max.view(1, 1, -1)).mean(dim=2)
            return (0.6 * hidden_score + 0.4 * output_score).clamp(0.0, 1.0)
        return hidden_score.clamp(0.0, 1.0)

    hidden = hidden_spikes.mean(dim=2)
    if output_spikes is None or output_spikes.numel() == 0:
        return hidden
    out = output_spikes.mean(dim=2)
    return 0.7 * hidden + 0.3 * out

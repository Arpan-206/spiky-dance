"""`SparseSpikyNet`: the Mr. Spiky network with three sparse-activity mechanisms.

PROVENANCE: this file is new. It is NOT a copy of Mr. Spiky code. It starts
from `SpikyNet` of github.com/Arpan-206/mr-spiky (`src/model.py`, vendored in
`model.py`) and keeps: two layers of leaky integrate-and-fire (LIF) neurons,
FEATURE_DIM -> 128 -> 32, BETA = 0.9, no bias, non-negative weights with
uniform(0, 0.5) init, base threshold 1.0, reset by subtraction.

WHY: measured on real dance features, `SpikyNet` is saturated. With untrained
weights 99.7 % of the hidden neurons and 100 % of the output neurons fire in
each bin (input current per bin: mean 1.62, threshold 1.0). STDP then gives
all neurons the same update. Baselines become 1.0 and all raw scores become 0.

WHAT DIFFERS FROM MR. SPIKY (the user allowed exactly these three changes):
  1. Adaptive per-neuron threshold (homeostasis). Each neuron has an offset
     `theta`. Its threshold is 1.0 + theta. In training, theta goes up when
     the neuron fires more than TARGET_RATE and down when it fires less.
     For scoring, theta is frozen (it is saved with the weights).
  2. Top-k lateral inhibition per layer. In one step, at most
     TOPK_FRACTION of the neurons of a layer fire: those with the largest
     membrane excess above their threshold. A neuron above its threshold that
     loses is pulled back to its threshold.
  3. Per-neuron weight normalisation. After each STDP update, `normalize_rows`
     rescales the input weights of each neuron to a fixed sum.
Also different: the LIF step is written in plain torch (snntorch's `Leaky`
has no top-k), and the recorded membrane is the value before reset.

Scoring is deterministic: no random numbers are used outside training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import BETA, HIDDEN, LIF1_THRESHOLD, LIF2_THRESHOLD, NUM_FEATURES, OUTPUT, SNNOutput

TARGET_RATE = 0.10     # homeostasis target: a neuron fires in 10 % of the steps
TOPK_FRACTION = 0.15   # lateral inhibition: at most 15 % of a layer fires in one step
THETA_LR = 0.02        # homeostasis step size
MIN_THRESHOLD = 0.05   # a threshold never goes below this value
INIT_MEAN_WEIGHT = 0.25  # mean of the uniform(0, 0.5) init; sets the row sum


def normalize_rows(weight: torch.Tensor) -> None:
    """Per-neuron weight normalisation, in place. weight: (out, in).

    Each row (the input weights of one neuron) is scaled to the sum
    INIT_MEAN_WEIGHT * in. Weights stay in 0..1.
    """
    target = INIT_MEAN_WEIGHT * weight.shape[1]
    weight.mul_(target / weight.sum(dim=1, keepdim=True).clamp(min=1e-6))
    weight.clamp_(0.0, 1.0)


def _lif_topk(
    cur: torch.Tensor, mem: torch.Tensor, thr: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One LIF step with top-k lateral inhibition.

    cur, mem: (B, N). thr: (N,). Returns (spikes, potential, new membrane).
    `potential` is the membrane after integration and before reset.
    """
    pot = BETA * mem + cur
    exc = pot - thr
    fire = exc > 0
    if k < exc.shape[1]:
        kth = exc.topk(k, dim=1).values[:, -1:]
        fire = fire & (exc >= kth)
    # Winners: reset by subtraction. Losers above threshold: pulled back to it.
    new_mem = torch.where(fire, pot - thr, torch.minimum(pot, thr))
    return fire.to(cur.dtype), pot, new_mem


class SparseSpikyNet(nn.Module):
    def __init__(self, input_dim: int = NUM_FEATURES, hidden: int = HIDDEN, output: int = OUTPUT):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, output, bias=False)
        # STDP-style init, as in Mr. Spiky: small non-negative weights.
        nn.init.uniform_(self.fc1.weight, 0.0, 0.5)
        nn.init.uniform_(self.fc2.weight, 0.0, 0.5)
        with torch.no_grad():
            normalize_rows(self.fc1.weight)
            normalize_rows(self.fc2.weight)
        # Adaptive threshold offsets. They are saved with the weights.
        self.register_buffer("theta1", torch.zeros(hidden))
        self.register_buffer("theta2", torch.zeros(output))
        self.k1 = max(1, int(round(TOPK_FRACTION * hidden)))
        self.k2 = max(1, int(round(TOPK_FRACTION * output)))

    def thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (LIF1_THRESHOLD + self.theta1).clamp(min=MIN_THRESHOLD),
            (LIF2_THRESHOLD + self.theta2).clamp(min=MIN_THRESHOLD),
        )

    def init_state(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero membrane state for `batch` items (one item per dancer)."""
        return (
            torch.zeros(batch, self.fc1.out_features),
            torch.zeros(batch, self.fc2.out_features),
        )

    @torch.no_grad()
    def run(
        self,
        spikes_in: torch.Tensor,
        mem1: torch.Tensor,
        mem2: torch.Tensor,
        adapt: bool = False,
    ) -> tuple[SNNOutput, torch.Tensor, torch.Tensor]:
        """Run (T, B, F) from the given membrane state. T must be at least 1.

        adapt=True changes theta1 and theta2 (homeostasis). Use it only in
        training. Returns (SNNOutput, mem1, mem2). The membrane traces in
        SNNOutput are the potentials before reset.
        """
        h_spk, o_spk, h_pot, o_pot = [], [], [], []
        for t in range(spikes_in.shape[0]):
            thr1, thr2 = self.thresholds()
            spk1, pot1, mem1 = _lif_topk(self.fc1(spikes_in[t]), mem1, thr1, self.k1)
            spk2, pot2, mem2 = _lif_topk(self.fc2(spk1), mem2, thr2, self.k2)
            if adapt:
                self.theta1 += THETA_LR * (spk1.mean(dim=0) - TARGET_RATE)
                self.theta2 += THETA_LR * (spk2.mean(dim=0) - TARGET_RATE)
            h_spk.append(spk1)
            o_spk.append(spk2)
            h_pot.append(pot1)
            o_pot.append(pot2)
        out = SNNOutput(
            hidden_spikes=torch.stack(h_spk, dim=0),
            output_spikes=torch.stack(o_spk, dim=0),
            hidden_mem=torch.stack(h_pot, dim=0),
            output_mem=torch.stack(o_pot, dim=0),
        )
        return out, mem1, mem2

    def forward(self, spikes_in: torch.Tensor) -> SNNOutput:
        """spikes_in: (T, B, F), from zero membrane state, thresholds frozen."""
        mem1, mem2 = self.init_state(spikes_in.shape[1])
        out, _, _ = self.run(spikes_in, mem1, mem2, adapt=False)
        return out

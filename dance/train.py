"""Train the spiking network (SNN) of spiky-dance on normal clips only.

Library use:
    from dance.train import train
    train(feature_clips, "models/dance")

Command line:
    uv run python -m dance.train CLIPS_DIR [--out models/dance] [--dancers N] [--max-seconds S]

`train` fits five artefacts again from the given clips, in this order:
    1. whitening.pt       ZCA whitening of the features
    2. snn_weights.pt     SparseSpikyNet weights from STDP (unsupervised), and
                          the adaptive threshold offsets of the neurons
    3. snn_baselines.pt   mean firing rate per neuron on the normal clips
    4. snn_ecdf.pt        sorted raw scores of the normal clips
    5. threshold.json     threshold 0.90 (p90) and metadata

From Mr. Spiky (see `dance/spiky`), not changed: the STDP rule and its
hyperparameters, ZCA whitening, the membrane attribution above per-neuron
baselines, the ECDF percentile calibration and the p90 threshold. No artefact
of an earlier model is read.

Different from Mr. Spiky (design change, see `dance/spiky/sparse.py`):
    - adaptive per-neuron threshold (homeostasis), target 10 % firing;
    - top-k lateral inhibition per layer, at most 15 % of a layer fires;
    - per-neuron weight normalisation after each STDP update.
Mr. Spiky's network was saturated on dance features: each neuron fired in
each bin, so all raw scores were 0.

How the membrane state is used in training:
    - Each time bin is rate-coded into STEPS_PER_BIN binary spike steps. The
      bins stay in time order.
    - The membrane state is carried across all steps of one clip.
    - The membrane state starts from zero for each clip. Each dancer is one
      batch item, thus each dancer has a separate membrane state.
    - `_stdp_update` runs one time per STDP_WINDOW = 25 steps, as in Mr. Spiky.
    - After STDP, the weights are frozen. The thresholds then settle in
      sequence mode (one graded-current step per bin), the mode of scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dance.contracts import BIN_MS, FEATURE_DIM
from dance.spiky import calibrate as cal
from dance.spiky.encode import (
    DEFAULT_STEPS,
    ENCODINGS,
    encode_rate,
    encode_sequence,
    encoded_dim,
    fit_whitening,
    normalize,
)
from dance.spiky.scorer import (
    BASELINES_FILE,
    ECDF_FILE,
    THRESHOLD_FILE,
    WEIGHTS_FILE,
    WHITENING_FILE,
)
from dance.spiky.sparse import TARGET_RATE, TOPK_FRACTION, SparseSpikyNet, normalize_rows
from dance.spiky.stdp import EPOCHS, _stdp_update

STEPS_PER_BIN = 5            # rate-coded spike steps per time bin in training
STDP_WINDOW = DEFAULT_STEPS  # steps per `_stdp_update` call (25, as in Mr. Spiky)
SETTLE_STEPS = 20_000        # sequence-mode steps for the thresholds to settle
MIN_SAMPLES = 2              # minimum bins x dancers in the whole corpus
ACTIVITY_TARGET = (0.05, 0.15)  # wanted fraction of neurons that fire per bin


class DegenerateCalibrationError(RuntimeError):
    """The raw scores of the normal clips have only one distinct value.

    Then the ECDF maps each input to the same percentile, and the model cannot
    tell normal from abnormal. `train` raises this error and writes no artefact.
    """


def _check_clips(feature_clips: list[np.ndarray]) -> list[np.ndarray]:
    clips: list[np.ndarray] = []
    for i, clip in enumerate(feature_clips):
        arr = np.asarray(clip, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != FEATURE_DIM:
            raise ValueError(
                f"clip {i}: shape must be (bins, dancers, {FEATURE_DIM}), got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"clip {i}: features contain NaN or infinite values")
        if arr.shape[0] > 0 and arr.shape[1] > 0:
            clips.append(arr)
    n = sum(c.shape[0] * c.shape[1] for c in clips)
    if n < MIN_SAMPLES:
        raise ValueError(
            f"training needs at least {MIN_SAMPLES} bins in total, got {n} "
            f"from {len(feature_clips)} clips"
        )
    return clips


def _weight_stats(w: torch.Tensor) -> dict:
    return {
        "mean": float(w.mean()),
        "std": float(w.std()),
        "min": float(w.min()),
        "max": float(w.max()),
        "fraction_zero": float((w <= 1e-6).float().mean()),
        "fraction_at_max": float((w >= 1.0 - 1e-6).float().mean()),
        "histogram_10_bins_0_to_1": torch.histc(w, 10, 0.0, 1.0).int().tolist(),
    }


def _train_stdp(
    net: SparseSpikyNet,
    clips: list[np.ndarray],
    whitening: dict[str, torch.Tensor],
    epochs: int,
    steps_per_bin: int,
    verbose: bool,
) -> None:
    """Unsupervised STDP over the clips, membrane carried across bins.

    Homeostasis is on. The rows of a weight matrix are normalised after each
    STDP update. Changes `net` in place.
    """
    for epoch in range(epochs):
        order = torch.randperm(len(clips)).tolist()
        for i in order:
            spikes = encode_rate(clips[i], whitening, steps_per_bin)   # (T*K, B, F)
            mem1, mem2 = net.init_state(spikes.shape[1])               # reset per clip
            for start in range(0, spikes.shape[0], STDP_WINDOW):
                chunk = spikes[start:start + STDP_WINDOW]
                out, mem1, mem2 = net.run(chunk, mem1, mem2, adapt=True)
                _stdp_update(net.fc1.weight.data, chunk, out.hidden_spikes)
                normalize_rows(net.fc1.weight.data)
                _stdp_update(net.fc2.weight.data, out.hidden_spikes, out.output_spikes)
                normalize_rows(net.fc2.weight.data)
        if verbose:
            w1, w2 = net.fc1.weight.data, net.fc2.weight.data
            print(
                f"epoch {epoch + 1}/{epochs}  fc1 mean={w1.mean():.3f} std={w1.std():.3f}  "
                f"fc2 mean={w2.mean():.3f} std={w2.std():.3f}",
                file=sys.stderr,
            )


def _settle_thresholds(
    net: SparseSpikyNet,
    clips: list[np.ndarray],
    whitening: dict[str, torch.Tensor],
    settle_steps: int,
) -> None:
    """Homeostasis only, in sequence mode, weights frozen. Changes the thresholds."""
    total = sum(c.shape[0] for c in clips)
    for _ in range(max(1, math.ceil(settle_steps / max(total, 1)))):
        for clip in clips:
            seq = encode_sequence(clip, whitening)
            mem1, mem2 = net.init_state(seq.shape[1])
            net.run(seq, mem1, mem2, adapt=True)


def train(
    feature_clips: list[np.ndarray],
    out_dir: str | Path,
    *,
    epochs: int = EPOCHS,
    seed: int = 0,
    steps_per_bin: int = STEPS_PER_BIN,
    settle_steps: int = SETTLE_STEPS,
    encoding: str = "onoff",
    regions: list[str] | None = None,
    verbose: bool = False,
) -> None:
    """Fit the spiking network and its calibration on normal clips only.

    feature_clips: one float32 array (bins_i, dancers_i, FEATURE_DIM) per clean clip.
    out_dir: folder for all artefacts. It is created. Old files are overwritten.
    epochs: passes over the clips for STDP (default 6, as in Mr. Spiky).
    seed: seed for the weight init, the clip order and the rate coding.
    steps_per_bin: rate-coded spike steps per time bin in training.
    settle_steps: sequence-mode steps for the adaptive thresholds to settle.
    encoding: "onoff" (default) gives two inputs for each whitened value, so
        a deviation in either direction raises an input. It is not from Mr.
        Spiky: see `dance/spiky/encode.py`, change 6. "shift" is the input
        encoding of Mr. Spiky. It is opt-in, and it does not detect a freeze.
    regions: optional, how the dancer boxes of each clip were found ("poses"
        or "strips"). It is only recorded in threshold.json.

    Raises DegenerateCalibrationError, and writes no artefact, if the raw
    scores of the normal clips have only one distinct value.
    """
    clips = _check_clips(feature_clips)
    if epochs < 1 or steps_per_bin < 1 or settle_steps < 0:
        raise ValueError("epochs and steps_per_bin must be at least 1, settle_steps at least 0")
    encoded_dim(encoding)  # checks the name
    out_dir = Path(out_dir)

    rng_state = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # the tensors are tiny: one thread is faster
    t0 = time.time()
    try:
        # 1. Whitening, from all bins of all dancers of all clips.
        rows = torch.cat([normalize(c).reshape(-1, FEATURE_DIM) for c in clips])
        whitening = fit_whitening(rows, encoding=encoding)

        # 2. STDP weights, then the thresholds for sequence mode.
        net = SparseSpikyNet(input_dim=encoded_dim(encoding))
        net.eval()
        _train_stdp(net, clips, whitening, epochs, steps_per_bin, verbose)
        _settle_thresholds(net, clips, whitening, settle_steps)

        # 3. Per-neuron baselines, in sequence mode.
        hidden_baseline, output_baseline = cal.compute_baselines(net, clips, whitening)
        activity = cal.activity(net, clips, whitening)

        # 4. ECDF reference: sorted raw scores of the normal clips.
        raw = cal.corpus_raw_scores(net, clips, whitening, hidden_baseline, output_baseline)
        sorted_raw = raw.sort().values.to(torch.float32)
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng_state)

    thr1, thr2 = net.thresholds()
    for name, t in (("fc1", net.fc1.weight.data), ("fc2", net.fc2.weight.data),
                    ("thresholds", torch.cat([thr1, thr2])), ("ecdf", sorted_raw)):
        if not bool(torch.isfinite(t).all()):
            raise RuntimeError(f"training gave non-finite values in {name}")

    n_distinct = int(torch.unique(sorted_raw).numel())
    if n_distinct <= 1:
        raise DegenerateCalibrationError(
            "calibration failed: all "
            f"{sorted_raw.numel()} raw scores of the normal clips are equal "
            f"({float(sorted_raw[0]):.6f}). Fraction of neurons that fire per bin: "
            f"{activity['hidden']:.4f} in the hidden layer and {activity['output']:.4f} in "
            f"the output layer (target {ACTIVITY_TARGET[0]} to {ACTIVITY_TARGET[1]}). "
            f"fc1 weights: mean {float(net.fc1.weight.data.mean()):.4f}, std "
            f"{float(net.fc1.weight.data.std()):.4f}. fc2 weights: mean "
            f"{float(net.fc2.weight.data.mean()):.4f}, std {float(net.fc2.weight.data.std()):.4f}. "
            f"No artefact was written to {out_dir}."
        )

    # 5. Threshold and metadata.
    raw_stats = cal.stats(sorted_raw)
    lo, hi = ACTIVITY_TARGET
    meta = {
        "threshold": cal.THRESHOLD,
        "n_samples": raw_stats["n"],
        "method": "anomaly: p90 of baseline-relative membrane attribution over the "
                  "normal clips (sequence-SNN, ECDF percentile)",
        "normal_stats": raw_stats,
        "n_distinct_raw_scores": n_distinct,
        "n_clips": len(clips),
        "n_clips_given": len(feature_clips),
        "bins_per_clip": [int(c.shape[0]) for c in clips],
        "dancers_per_clip": [int(c.shape[1]) for c in clips],
        "regions": sorted(set(regions)) if regions else [],
        "regions_per_clip": list(regions) if regions else [],
        "bin_ms": BIN_MS,
        "feature_dim": FEATURE_DIM,
        "encoding": encoding,
        "epochs": epochs,
        "seed": seed,
        "steps_per_bin": steps_per_bin,
        "stdp_window": STDP_WINDOW,
        "settle_steps": settle_steps,
        "network": "SparseSpikyNet: adaptive thresholds, top-k inhibition, row-normalised weights",
        "target_rate": TARGET_RATE,
        "topk_fraction": TOPK_FRACTION,
        "activity": activity,
        "activity_in_target": bool(lo <= activity["hidden"] <= hi and lo <= activity["output"] <= hi),
        "thresholds": {
            "hidden_mean": float(thr1.mean()), "hidden_min": float(thr1.min()),
            "hidden_max": float(thr1.max()),
            "output_mean": float(thr2.mean()), "output_min": float(thr2.min()),
            "output_max": float(thr2.max()),
        },
        "weights": {
            "fc1": _weight_stats(net.fc1.weight.data),
            "fc2": _weight_stats(net.fc2.weight.data),
        },
        "baselines": {
            "hidden_mean": float(hidden_baseline.mean()),
            "hidden_max": float(hidden_baseline.max()),
            "output_mean": float(output_baseline.mean()),
            "output_max": float(output_baseline.max()),
        },
        "train_seconds": round(time.time() - t0, 2),
        "source": "STDP rule, whitening, attribution, calibration: github.com/Arpan-206/mr-spiky",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(whitening, out_dir / WHITENING_FILE)
    torch.save(net.state_dict(), out_dir / WEIGHTS_FILE)
    torch.save({"hidden": hidden_baseline, "output": output_baseline}, out_dir / BASELINES_FILE)
    torch.save({"sorted_raw_scores": sorted_raw}, out_dir / ECDF_FILE)
    (out_dir / THRESHOLD_FILE).write_text(json.dumps(meta, indent=2))


# ---- Command line -----------------------------------------------------------

def find_clips(clips_dir: str | Path) -> list[Path]:
    """Video files and eventify HDF5 files in a folder, sorted by name.

    If a video and an events file have the same stem, only the events file is
    used (it is the same clip, already converted).
    """
    from dance import stream

    clips_dir = Path(clips_dir)
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"clips folder not found: {clips_dir}")
    suffixes = set(stream.VIDEO_SUFFIXES) | set(stream.HDF5_SUFFIXES)
    found = sorted(
        p for p in clips_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith(".")
    )
    events_stems = {p.stem for p in found if p.suffix.lower() in stream.HDF5_SUFFIXES}
    return [
        p for p in found
        if p.suffix.lower() in stream.HDF5_SUFFIXES or p.stem not in events_stems
    ]


def build_features(paths: list[Path], dancers: int, max_seconds: float | None, jobs: int) -> list[tuple[np.ndarray, dict]]:
    """Features of all clips. Clips run in parallel worker processes."""
    from concurrent.futures import ProcessPoolExecutor

    from dance.features import clip_features_job

    work = [(str(path), dancers, max_seconds) for path in paths]
    if jobs <= 1 or len(work) == 1:
        return [clip_features_job(job) for job in work]
    with ProcessPoolExecutor(max_workers=min(jobs, len(work))) as pool:
        return list(pool.map(clip_features_job, work))


def main(argv: list[str] | None = None) -> None:
    """One command to train again on a folder of clean clips.

    One clip is held out. The scorers are fitted on the other clips. Then the
    held-out clip is scored, so you can see that a normal clip gets a mean score
    near 0.4 to 0.5 and about 10 percent of its bins above the threshold.
    """
    import os
    import time

    p = argparse.ArgumentParser(
        description="Train the scorers on a folder of clean (normal) clips, with a held-out check."
    )
    p.add_argument("clips_dir", help="folder with video clips or eventify .h5 files")
    p.add_argument("--out", default="models/dance", help="folder for the model artefacts")
    p.add_argument("--scorer", choices=("baseline", "spiking", "both"), default="both",
                   help="which scorer to fit")
    p.add_argument("--dancers", type=int, default=1,
                   help="number of vertical strips when a clip has no poses file")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="use only the first S seconds of each clip")
    p.add_argument("--holdout", default=None,
                   help="file name of the clip to hold out; default: the last clip by name")
    p.add_argument("--no-holdout", action="store_true", help="train on all clips, no held-out check")
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                   help="worker processes for the event and feature work")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="spiking scorer: training epochs")
    p.add_argument("--seed", type=int, default=0, help="spiking scorer: random seed")
    p.add_argument("--encoding", choices=ENCODINGS, default="onoff",
                   help="spiking scorer. onoff (default): rectified, see encode.py; "
                        "shift: the input encoding of Mr. Spiky, opt-in")
    args = p.parse_args(argv)

    from dance import baseline

    start = time.perf_counter()
    try:
        paths = find_clips(args.clips_dir)
    except FileNotFoundError as err:
        raise SystemExit(f"error: {err}")
    if not paths:
        raise SystemExit(
            f"error: no clips in {args.clips_dir}. "
            "Put video files or eventify .h5 files in that folder."
        )

    held: Path | None = None
    if not args.no_holdout:
        if args.holdout is not None:
            matches = [q for q in paths if q.name == args.holdout]
            if not matches:
                raise SystemExit(f"error: held-out clip {args.holdout!r} is not in {args.clips_dir}")
            held = matches[0]
        elif len(paths) >= 2:
            held = paths[-1]
        else:
            print("Only 1 clip: no held-out check. Give 2 or more clips for the check.")

    built = build_features(paths, args.dancers, args.max_seconds, args.jobs)
    clips: list[np.ndarray] = []
    regions: list[str] = []
    held_features: np.ndarray | None = None
    for path, (f, info) in zip(paths, built):
        role = "held out" if path == held else "train"
        print(f"{path.name}: {f.shape[0]} bins, {f.shape[1]} dancer(s), regions from {info['regions']}, {role}")
        if path == held:
            held_features = f
        else:
            clips.append(f)
            regions.append(str(info["regions"]))
    features_s = time.perf_counter() - start

    rows: list[tuple[str, float, float]] = []
    try:
        if args.scorer in ("baseline", "both"):
            baseline.train(clips, args.out)
            if held_features is not None:
                s = baseline.load_model(args.out).score(held_features)
                rows.append(("baseline", float(s.mean()), float((s >= 0.90).mean())))
        if args.scorer in ("spiking", "both"):
            from dance.spiky import load_model

            train(clips, args.out, epochs=args.epochs, seed=args.seed, encoding=args.encoding,
                  regions=regions, verbose=False)
            if held_features is not None:
                s = load_model(args.out).score(held_features)
                rows.append(("spiking", float(s.mean()), float((s >= 0.90).mean())))
    except (ValueError, DegenerateCalibrationError) as err:
        raise SystemExit(f"error: {err}")

    total_s = time.perf_counter() - start
    print(f"\nModels saved in {args.out}. Trained on {len(clips)} clip(s).")
    if args.scorer in ("spiking", "both"):
        meta = json.loads((Path(args.out) / THRESHOLD_FILE).read_text())
        act = meta["activity"]
        print(f"Spiking scorer: encoding {meta['encoding']}, neurons that fire per bin: "
              f"hidden {act['hidden']:.1%}, output {act['output']:.1%} (target 5% to 15%).")
    if rows:
        print(f"\nHeld-out normal clip: {held.name}. It was not used for training.")
        print("| Scorer | Mean score | Fraction above p90 |")
        print("|---|---|---|")
        for name, mean, fraction in rows:
            print(f"| {name} | {mean:.3f} | {fraction:.3f} |")
        print("Expected for a normal clip: a mean near 0.4 to 0.5 and a fraction near 0.10.")
        for name, mean, fraction in rows:
            if not (0.30 <= mean <= 0.60) or fraction > 0.25:
                print(f"NOTE: the {name} scorer is outside that range. The held-out clip differs from "
                      "the training clips, or there are too few training clips.")
    print(f"Time: {total_s:.1f} s (events and features {features_s:.1f} s).")


if __name__ == "__main__":
    main()

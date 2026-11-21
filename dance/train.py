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
    """Video files and eventify HDF5 files directly in a folder, sorted by name.

    If a video and an events file have the same stem, only the events file is
    used (it is the same clip, already converted).
    """
    from dance.evaluate import clip_files

    clips_dir = Path(clips_dir)
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"clips folder not found: {clips_dir}")
    return clip_files(clips_dir)


def choose_held_out(clips: list, holdout: list[str] | None, no_holdout: bool) -> list:
    """The clean clips that are held out of the training.

    `holdout` lists file names or name prefixes. A prefix holds out a whole
    video that was cut into several clips. Without `holdout`, the last clean
    clip by name of each dancer count is held out, when that count has 2 or
    more clean clips. Bad clips are never candidates: they are never trained on.
    """
    clean = [c for c in clips if c.role != "bad"]
    if no_holdout:
        return []
    if holdout:
        held = [c for c in clean if any(c.path.name == h or c.path.name.startswith(h) for h in holdout)]
        missing = [h for h in holdout if not any(c.path.name == h or c.path.name.startswith(h) for c in clean)]
        if missing:
            raise ValueError(f"no clean clip matches --holdout {', '.join(missing)}")
        return held
    held = []
    for group in sorted({c.group for c in clean}):
        members = [c for c in clean if c.group == group]
        if len(members) >= 2:
            held.append(members[-1])
    return held


def leave_one_out(train_clips: list, bad_clips: list, scorers: tuple[str, ...], *, epochs: int,
                  seed: int, encoding: str) -> tuple[str, dict[str, dict]]:
    """Leave-one-out test and the display remap that follows from it.

    For each clean clip: fit on all other clean clips, then score that clip and
    all bad clips. Returns (report text, {scorer: display remap}).
    """
    import tempfile

    from dance import baseline, display
    from dance.spiky import load_model

    clean = {name: [] for name in scorers}                          # (mean, fraction) of each unseen clean clip
    bad = {name: {str(c.path): [] for c in bad_clips} for name in scorers}
    for i, held in enumerate(train_clips):
        rest = [c.features for j, c in enumerate(train_clips) if j != i]
        with tempfile.TemporaryDirectory() as tmp:
            models = {}
            if "spiking" in scorers:
                train(rest, tmp, epochs=epochs, seed=seed, encoding=encoding, regions=None, verbose=False)
                models["spiking"] = load_model(tmp)
            if "baseline" in scorers:
                baseline.train(rest, tmp)
                models["baseline"] = baseline.load_model(tmp)
            for name, model in models.items():
                s = np.asarray(model.score(held.features))
                clean[name].append((float(s.mean()), float((s >= 0.90).mean())))
                for c in bad_clips:
                    sb = np.asarray(model.score(c.features))
                    bad[name][str(c.path)].append((float(sb.mean()), float((sb >= 0.90).mean())))
    lines = [f"\nLeave-one-out test: {len(train_clips)} clean clip(s), {len(bad_clips)} bad clip(s). "
             'Each cell is "mean score / fraction of bins above p90".']
    lines.append("| Clip | " + " | ".join(scorers) + " |")
    lines.append("|---|" + "---|" * len(scorers))
    for i, held in enumerate(train_clips):
        cells = [f"{clean[n][i][0]:.3f} / {clean[n][i][1]:.3f}" for n in scorers]
        lines.append(f"| clean, not seen: {held.path.stem} | " + " | ".join(cells) + " |")
    bad_mean = {n: {k: tuple(np.mean(v, axis=0)) for k, v in bad[n].items()} for n in scorers}
    for c in bad_clips:
        cells = [f"{bad_mean[n][str(c.path)][0]:.3f} / {bad_mean[n][str(c.path)][1]:.3f}" for n in scorers]
        lines.append(f"| bad: {c.path.stem} | " + " | ".join(cells) + " |")
    cells = [f"{np.mean([m for m, _ in clean[n]]):.3f} / {np.mean([f for _, f in clean[n]]):.3f}" for n in scorers]
    lines.append(f"| **clean, not seen, all {len(train_clips)}** | " + " | ".join(cells) + " |")
    if bad_clips:
        cells = [f"{np.mean([v[0] for v in bad_mean[n].values()]):.3f} / "
                 f"{np.mean([v[1] for v in bad_mean[n].values()]):.3f}" for n in scorers]
        lines.append(f"| **bad, all {len(bad_clips)}** | " + " | ".join(cells) + " |")
        for n in scorers:
            top = max(m for m, _ in clean[n])
            above = sum(1 for v in bad_mean[n].values() if v[0] > top)
            lines.append(f"{n}: {above} of {len(bad_clips)} bad clips have a mean score above the highest "
                         f"clean clip ({top:.3f}).")
    remaps = {n: display.fit_remap([m for m, _ in clean[n]], [v[0] for v in bad_mean[n].values()])
              for n in scorers}
    for n in scorers:
        r = remaps[n]
        lines.append(f"{n} display remap: mean {r['clean_ref']:.3f} -> {r['clean_display']:.0f}, "
                     f"mean {r['bad_ref']:.3f} -> {r['bad_display']:.0f}.")
        shown = [f"{display.display_score(m, r):.0f}" for m, _ in clean[n]]
        lines.append(f"{n} display scores, clean clips not seen: {', '.join(shown)}.")
        if bad_clips:
            shown = [f"{display.display_score(v[0], r):.0f}" for v in bad_mean[n].values()]
            lines.append(f"{n} display scores, bad clips: {', '.join(shown)}.")
    return "\n".join(lines), remaps


def main(argv: list[str] | None = None) -> None:
    """One command to train again on clean clips, with a held-out check.

    Folder layout:
        CLIPS_DIR/1p/, CLIPS_DIR/2p/, CLIPS_DIR/3p/   clips with 1, 2, 3 dancers
        CLIPS_DIR/*.mp4                               flat form, `--dancers N` dancers

    A clip whose name starts with `bad_` is never used for training. It is
    scored after the training and shown in the table. Some clean clips are
    held out, so the table shows how clips score that the training did not see.
    """
    import os
    import time

    p = argparse.ArgumentParser(
        description="Train the scorers on clean (normal) clips, with a held-out check and a good/bad table."
    )
    p.add_argument("clips_dir", nargs="+",
                   help="one or more folders, each with sub-folders 1p, 2p, 3p or with clips directly in it")
    p.add_argument("--out", default="models/dance", help="folder for the model artefacts")
    p.add_argument("--scorer", choices=("baseline", "spiking", "both"), default="both",
                   help="which scorer to fit")
    p.add_argument("--dancers", type=int, default=1,
                   help="flat form: number of vertical strips when a clip has no poses file")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="use only the first S seconds of each clip")
    p.add_argument("--holdout", action="append", default=None,
                   help="file name, or start of the file name, of clean clips to hold out; can be given "
                        "more than once; default: the last clean clip by name of each dancer count")
    p.add_argument("--no-holdout", action="store_true", help="train on all clean clips, no held-out check")
    p.add_argument("--leave-one-out", action="store_true",
                   help="for small sets: test each clean clip against a model fitted on the others, print the "
                        "table, fit the display remap (display.json), then train the saved models on all clean clips")
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                   help="worker processes for the event and feature work")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="spiking scorer: training epochs")
    p.add_argument("--seed", type=int, default=0, help="spiking scorer: random seed")
    p.add_argument("--encoding", choices=ENCODINGS, default="onoff",
                   help="spiking scorer. onoff (default): rectified, see encode.py; "
                        "shift: the input encoding of Mr. Spiky, opt-in")
    args = p.parse_args(argv)

    from dance import baseline, evaluate

    start = time.perf_counter()
    try:
        clips = evaluate.collect_clips(args.clips_dir, args.dancers)
        if not clips:
            raise SystemExit(
                f"error: no clips in {', '.join(args.clips_dir)}. Put video files or eventify .h5 files "
                "in that folder, or in its sub-folders 1p, 2p, 3p."
            )
        held = choose_held_out(clips, args.holdout, args.no_holdout or args.leave_one_out)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"error: {err}")
    held_paths = {str(c.path) for c in held}
    train_clips = [c for c in clips if c.role != "bad" and str(c.path) not in held_paths]
    bad_clips = [c for c in clips if c.role == "bad"]
    if not train_clips:
        raise SystemExit("error: no clean clip is left for training. Give more clean clips or use --no-holdout.")
    if args.leave_one_out and len(train_clips) < 3:
        raise SystemExit("error: --leave-one-out needs 3 or more clean clips.")
    if not held and not args.no_holdout and not args.leave_one_out:
        print("No held-out check: a dancer count needs 2 or more clean clips for it.")

    evaluate.add_features(clips, args.max_seconds, args.jobs)
    roles = {}
    for c in clips:
        role = "bad" if c.role == "bad" else ("held-out clean" if str(c.path) in held_paths else "train")
        roles[str(c.path)] = role
        print(f"{c.group}/{c.path.name}: {c.features.shape[0]} bins, {c.features.shape[1]} dancer(s), {role}")
    features_s = time.perf_counter() - start

    feature_clips = [c.features for c in train_clips]
    models: dict[str, object] = {}
    try:
        if args.scorer in ("spiking", "both"):
            from dance.spiky import load_model

            train(feature_clips, args.out, epochs=args.epochs, seed=args.seed, encoding=args.encoding,
                  regions=None, verbose=False)
            models["spiking"] = load_model(args.out)
        if args.scorer in ("baseline", "both"):
            baseline.train(feature_clips, args.out)
            models["baseline"] = baseline.load_model(args.out)
    except (ValueError, DegenerateCalibrationError) as err:
        raise SystemExit(f"error: {err}")

    loo_text = None
    if args.leave_one_out:
        from dance import display

        loo_text, remaps = leave_one_out(train_clips, bad_clips, tuple(models), epochs=args.epochs,
                                         seed=args.seed, encoding=args.encoding)
        display.save_remaps(args.out, remaps)

    total_s = time.perf_counter() - start
    counts = ", ".join(f"{sum(1 for c in train_clips if c.group == g)} with {g[0]} dancer(s)"
                       for g in sorted({c.group for c in train_clips}))
    print(f"\nModels saved in {args.out}. Trained on {len(train_clips)} clean clip(s): {counts}.")
    if "spiking" in models:
        meta = json.loads((Path(args.out) / THRESHOLD_FILE).read_text())
        act = meta["activity"]
        print(f"Spiking scorer: encoding {meta['encoding']}, neurons that fire per bin: "
              f"hidden {act['hidden']:.1%}, output {act['output']:.1%} (target 5% to 15%).")
    if loo_text is not None:
        print(loo_text)
        print(f"Display remap saved in {Path(args.out) / 'display.json'}.")
    shown = [] if args.leave_one_out else held + bad_clips
    if shown:
        print('\nClips that the training did not use. Each cell is "mean score / fraction of bins above p90".')
        print(evaluate.report(shown, models, roles))
        print("\nExpected for a held-out clean clip: a mean near 0.4 to 0.5 and a fraction near 0.10.")
        for clip in held:
            for name, model in models.items():
                s = np.asarray(model.score(clip.features))
                if not (0.30 <= float(s.mean()) <= 0.60) or float((s >= 0.90).mean()) > 0.25:
                    print(f"NOTE: {clip.group}/{clip.path.name} is outside that range for the {name} scorer "
                          f"({s.mean():.3f} / {(s >= 0.90).mean():.3f}). It differs from the training "
                          "clips, or there are too few training clips.")
    print(f"Time: {total_s:.1f} s (events and features {features_s:.1f} s).")


if __name__ == "__main__":
    main()

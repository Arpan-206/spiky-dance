"""Step 4: features -> unusualness scores, offline and live.

A score is a percentile against the training clips, 0 to 1. So the p90
threshold is the constant 0.90. All scores come from the trained model in
`dance.spiky`. When the trained artefacts are missing, the program stops with
an error. It never invents a score.

Offline:
    python -m dance.score SOURCE [--out scores.npy] [--models models/dance]
        [--poses poses.npy] [--dancers N] [--max-seconds S]
    SOURCE is a video clip or an eventify .h5 events file.
    Writes scores.npy (float32, shape (bins, dancers)) and scores.json.

Live:
    python -m dance.score --live [--device 0] [--models models/dance]
        [--dancers N] [--duration S] [--warmup-frames 30] [--json]
    Reads the webcam through eventify. Gives one update about 10 times a second.

Dancer numbers are column numbers of the scores array. They start at 0. With
vertical strips, dancer 0 is the left strip.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from . import stream
from .contracts import BIN_MS, FEATURE_DIM
from .features import Regions, clip_features, extract_features, find_poses

DEFAULT_MODELS = "models/dance"
P90 = 0.90              # scores are percentiles, so the p90 threshold is constant
EMIT_EVERY = 3          # windows per live update: 3 * 33 ms = about 10 updates a second
BUFFER_WINDOWS = 90     # live context given to the model: about 3 seconds
ROLL_WINDOWS = 30       # the live rolling score is the mean of the last second
TRAIN_HINT = "Run `python -m dance.train` first."


# --------------------------------------------------------------------- model

def load(model_dir: str | Path = DEFAULT_MODELS) -> Any:
    """Load the trained model. Raise FileNotFoundError if artefacts are missing.

    The import is inside the function, so this module imports without the model.
    """
    from dance.spiky import load_model

    try:
        return load_model(model_dir)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"trained model artefacts are missing in '{model_dir}' ({err}). {TRAIN_HINT}"
        ) from err


def model_scores(model: Any, features: np.ndarray) -> np.ndarray:
    """Call `model.score` and check the result against the contract.

    A result with a wrong shape, or with values outside 0..1, raises ValueError.
    The values are never changed.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3 or features.shape[2] != FEATURE_DIM:
        raise ValueError(f"features must have shape (bins, dancers, {FEATURE_DIM}), got {features.shape}")
    scores = np.asarray(model.score(features))
    if scores.shape != features.shape[:2]:
        raise ValueError(f"model.score gave shape {scores.shape}, expected {features.shape[:2]}")
    if scores.size and not np.isfinite(scores).all():
        raise ValueError("model.score gave NaN or infinite scores")
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
        raise ValueError(
            f"model.score gave values outside 0..1: min {scores.min():.4f}, max {scores.max():.4f}"
        )
    return scores.astype(np.float32, copy=False)


# ------------------------------------------------------------------- summary

def summarise(scores: np.ndarray, threshold: float = P90) -> dict:
    """(bins, dancers) scores -> final numbers per dancer and a ranking.

    Per dancer: the mean score, and the fraction of bins with score >= threshold.
    Rank 1 is the most unusual dancer (the highest mean score).
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (bins, dancers), got {scores.shape}")
    bins, dancers = scores.shape
    if bins == 0:
        raise ValueError("there are no time bins to score")
    means = scores.mean(axis=0, dtype=np.float64)
    above = (scores >= np.float32(threshold)).mean(axis=0, dtype=np.float64)
    order = np.argsort(-means, kind="stable")
    rank = np.empty(dancers, dtype=np.int64)
    rank[order] = np.arange(1, dancers + 1)
    return {
        "bins": int(bins),
        "bin_ms": BIN_MS,
        "threshold": float(threshold),
        "ranking": [int(d) for d in order],
        "dancers": [
            {
                "dancer": d,
                "rank": int(rank[d]),
                "mean_score": float(means[d]),
                "fraction_above_p90": float(above[d]),
                "bins": int(bins),
            }
            for d in range(dancers)
        ],
    }


def score_features(model: Any, features: np.ndarray) -> tuple[np.ndarray, dict]:
    """(bins, dancers, 13) features -> ((bins, dancers) float32 scores, summary)."""
    scores = model_scores(model, features)
    return scores, summarise(scores)


def format_table(summary: dict) -> str:
    """The ranked table of dancers, most unusual first."""
    head = ("rank", "dancer", "mean score", "fraction above p90", "bins")
    by_dancer = {row["dancer"]: row for row in summary["dancers"]}
    rows = [
        (
            str(by_dancer[d]["rank"]),
            str(d),
            f"{by_dancer[d]['mean_score']:.3f}",
            f"{by_dancer[d]['fraction_above_p90']:.3f}",
            str(by_dancer[d]["bins"]),
        )
        for d in summary["ranking"]
    ]
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
    lines = ["  ".join(h.rjust(w) for h, w in zip(head, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(lines)


def save_scores(scores: np.ndarray, summary: dict, out: str | Path) -> tuple[Path, Path]:
    """Write the scores as .npy and the summary as .json with the same stem."""
    out = Path(out)
    if out.suffix != ".npy":
        out = out.with_name(out.name + ".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(scores, dtype=np.float32))
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps({**summary, "scores_file": out.name}, indent=2) + "\n")
    return out, json_path


# ------------------------------------------------------------------- offline

def load_poses(source: str | Path, poses: str | Path | np.ndarray | None = None) -> np.ndarray | None:
    """Poses from an array, from a file, or from a file next to the source."""
    if isinstance(poses, np.ndarray):
        return poses
    path = Path(poses) if poses is not None else find_poses(source)
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"poses file not found: {path}")
    return np.load(path)


def score_clip(
    source: str | Path,
    model_dir: str | Path = DEFAULT_MODELS,
    *,
    poses: str | Path | np.ndarray | None = None,
    dancers: int = 1,
    max_seconds: float | None = None,
    events_out: str | Path | None = None,
    model: Any = None,
) -> tuple[np.ndarray, dict]:
    """Clip or eventify .h5 file -> ((bins, dancers) scores, summary).

    The model is loaded before the event work. So missing artefacts stop the
    run at once. With `events_out`, the events are also saved as eventify HDF5.
    """
    if not Path(source).exists():
        raise FileNotFoundError(f"source not found: {source}")
    if model is None:
        model = load(model_dir)
    pose_array = load_poses(source, poses)
    features, info = clip_features(source, pose_array, dancers, max_seconds, events_out)
    scores, summary = score_features(model, features)
    summary.update(
        source=str(source),
        models=str(model_dir),
        regions=info["regions"],
        sensor_shape=info["sensor_shape"],
    )
    if events_out is not None:
        summary.update(events_file=str(events_out), events=info["events"])
    return scores, summary


# ---------------------------------------------------------------------- live

def live_loop(
    windows: Iterable[np.ndarray],
    model: Any,
    regions: Regions,
    emit: Callable[[dict], None],
    *,
    emit_every: int = EMIT_EVERY,
    buffer_windows: int = BUFFER_WINDOWS,
    roll_windows: int = ROLL_WINDOWS,
    threshold: float = P90,
) -> np.ndarray:
    """Score a stream of 33 ms windows and call `emit` every `emit_every` windows.

    windows: (1, 2, H, W) event-count arrays, as `stream.window_frames` yields.
    emit: called with {"t": seconds, "scores": [...], "latest": [...], "above": [...]}.
        "scores" is the rolling score per dancer: the mean of the last
        `roll_windows` scores. "latest" is the score of the newest window.
        "above" is true when the rolling score is >= threshold.

    `model.score` starts from zero membrane state at each call. So each update
    scores the last `buffer_windows` feature rows again.

    Ctrl-C and a closed output pipe stop the loop without an error. Returns the
    scores of the session, shape (bins, dancers), for the final table.
    """
    buffer: deque[np.ndarray] = deque(maxlen=buffer_windows)
    session: list[np.ndarray] = []
    pending = 0
    try:
        for k, frame in enumerate(windows):
            buffer.append(extract_features(frame, regions.boxes(k)))
            pending += 1
            if pending < emit_every:
                continue
            scores = model_scores(model, np.concatenate(buffer))
            session.append(scores[-pending:])
            pending = 0
            rolling = scores[-roll_windows:].mean(axis=0)
            emit(
                {
                    "t": round((k + 1) * BIN_MS / 1000, 3),
                    "scores": [round(float(v), 4) for v in rolling],
                    "latest": [round(float(v), 4) for v in scores[-1]],
                    "above": [bool(v >= threshold) for v in rolling],
                }
            )
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        close = getattr(windows, "close", None)
        if close is not None:
            close()
    if session:
        return np.concatenate(session)
    return np.zeros((0, regions.dancers), dtype=np.float32)


def json_emitter(out: TextIO) -> Callable[[dict], None]:
    """One JSON object per line, flushed, for a pipe."""
    def emit(update: dict) -> None:
        out.write(json.dumps(update) + "\n")
        out.flush()
    return emit


def line_emitter(out: TextIO) -> Callable[[dict], None]:
    """One compact text line per update. `*` marks a dancer at or above p90."""
    def emit(update: dict) -> None:
        cells = [
            f"d{d} {score:.2f}{'*' if above else ' '} (now {latest:.2f})"
            for d, (score, latest, above) in enumerate(
                zip(update["scores"], update["latest"], update["above"])
            )
        ]
        out.write(f"t={update['t']:7.2f}s  " + "  ".join(cells) + "\n")
        out.flush()
    return emit


def source_windows(
    source: str | int,
    info: stream.SourceInfo,
    *,
    duration_s: float | None = None,
    warmup_frames: int = 0,
    live: bool = False,
) -> Iterable[np.ndarray]:
    """eventify event chunks of a file or a webcam -> 33 ms windows.

    With `live`, time follows the wall clock, also when chunks are empty.
    """
    chunks = stream.event_chunks(source, info, duration_s=duration_s, warmup_frames=warmup_frames)
    clock = (lambda: int(time.monotonic() * 1e6)) if live else None
    chunk_dt_us = int(1_000_000 / info.fps / (stream.INTERP + 1))
    return stream.window_frames(chunks, info.sensor_shape, chunk_dt_us=chunk_dt_us, clock=clock)


def score_live(
    device: int = 0,
    model_dir: str | Path = DEFAULT_MODELS,
    *,
    dancers: int = 1,
    duration_s: float | None = None,
    warmup_frames: int = 30,
    as_json: bool = False,
    model: Any = None,
) -> np.ndarray:
    """Webcam -> rolling scores on stdout. Returns the scores of the session."""
    if model is None:
        model = load(model_dir)
    info = stream.probe_source(device)
    regions = Regions(info.sensor_shape, None, dancers)
    windows = source_windows(device, info, duration_s=duration_s, warmup_frames=warmup_frames, live=True)
    emit = json_emitter(sys.stdout) if as_json else line_emitter(sys.stdout)
    width, height = info.sensor_size
    print(
        f"live: device {device}, sensor {width}x{height}, {dancers} dancer(s). Press Ctrl-C to stop.",
        file=sys.stderr,
    )
    session = live_loop(windows, model, regions, emit)
    # With --json, stdout has only the JSON lines. The table goes to stderr.
    table_out = sys.stderr if as_json else sys.stdout
    try:
        if len(session):
            print(format_table(summarise(session)), file=table_out)
        else:
            print("No time bins were scored.", file=table_out)
    except BrokenPipeError:
        pass
    return session


# ----------------------------------------------------------------------- CLI

def stop(message: str) -> None:
    """Stop the program with a non-zero exit code and a message on stderr."""
    raise SystemExit(f"error: {message}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Unusualness scores per dancer, offline or live")
    p.add_argument("source", nargs="?", help="video clip or eventify .h5 events file (offline mode)")
    p.add_argument("--out", default="scores.npy", help="scores file; the summary gets the same stem with .json")
    p.add_argument("--models", default=DEFAULT_MODELS, help="folder with the trained artefacts")
    p.add_argument("--poses", default=None, help="poses.npy; default: look next to the source")
    p.add_argument("--dancers", type=int, default=1, help="number of vertical strips when there are no poses")
    p.add_argument("--max-seconds", type=float, default=None, help="offline: use only the first S seconds")
    p.add_argument("--live", action="store_true", help="read the webcam and print rolling scores")
    p.add_argument("--device", type=int, default=0, help="live: webcam index")
    p.add_argument("--duration", type=float, default=None, help="live: stop after S seconds")
    p.add_argument("--warmup-frames", type=int, default=30, help="live: frames to drop while exposure settles")
    p.add_argument("--json", action="store_true", help="live: one JSON object per line on stdout")
    args = p.parse_args(argv)

    if args.live == (args.source is not None):
        p.error("give a SOURCE for offline mode, or --live for the webcam, not both")

    start = time.perf_counter()
    try:
        if args.live:
            score_live(
                args.device,
                args.models,
                dancers=args.dancers,
                duration_s=args.duration,
                warmup_frames=args.warmup_frames,
                as_json=args.json,
            )
            return
        scores, summary = score_clip(
            args.source,
            args.models,
            poses=args.poses,
            dancers=args.dancers,
            max_seconds=args.max_seconds,
        )
    except FileNotFoundError as err:
        stop(str(err))
    summary["elapsed_s"] = round(time.perf_counter() - start, 2)
    npy_path, json_path = save_scores(scores, summary, args.out)
    print(format_table(summary))
    print(f"scores: {npy_path}, shape {scores.shape}, regions from {summary['regions']}")
    print(f"summary: {json_path}")
    print(f"elapsed: {summary['elapsed_s']:.1f} s")


if __name__ == "__main__":
    main()

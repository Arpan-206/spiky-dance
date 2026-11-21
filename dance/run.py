"""One command: clip -> events.h5, scores.npy, scores.json and a ranked table.

    python -m dance.run clip.mp4 [--out-dir out] [--models models/dance]
        [--scorer spiky|baseline] [--poses poses.npy] [--dancers N] [--max-seconds S]

The events file is eventify's HDF5 export. The trained model is loaded before
the event work. When the trained artefacts are missing, the program stops with
an error and writes no file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import score

EVENTS_NAME = "events.h5"
SCORES_NAME = "scores.npy"


def run_clip(
    source: str | Path,
    out_dir: str | Path = ".",
    model_dir: str | Path = score.DEFAULT_MODELS,
    *,
    poses: str | Path | np.ndarray | None = None,
    dancers: int = 1,
    max_seconds: float | None = None,
    model: Any = None,
    scorer: str = score.DEFAULT_SCORER,
) -> dict:
    """Clip -> `<out_dir>/events.h5`, `scores.npy`, `scores.json`. Returns the summary."""
    start = time.perf_counter()
    if not Path(source).exists():
        raise FileNotFoundError(f"source not found: {source}")
    if model is None:
        model = score.load(model_dir, scorer)      # before any event work and any output
    out_dir = Path(out_dir)
    scores, summary = score.score_clip(
        source,
        model_dir,
        poses=poses,
        dancers=dancers,
        max_seconds=max_seconds,
        events_out=out_dir / EVENTS_NAME,
        model=model,
        scorer=scorer,
    )
    summary["elapsed_s"] = round(time.perf_counter() - start, 2)
    npy_path, json_path = score.save_scores(scores, summary, out_dir / SCORES_NAME)
    summary.update(scores_file=str(npy_path), summary_file=str(json_path))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Clip -> events, scores and a ranked table of dancers")
    p.add_argument("source", help="video clip")
    p.add_argument("--out-dir", default=".", help="folder for events.h5, scores.npy and scores.json")
    p.add_argument("--models", default=score.DEFAULT_MODELS, help="folder with the trained artefacts")
    p.add_argument("--scorer", choices=score.SCORERS, default=score.DEFAULT_SCORER, help="which trained scorer to use")
    p.add_argument("--poses", default=None, help="poses.npy; default: look next to the source")
    p.add_argument("--dancers", type=int, default=1, help="number of vertical strips when there are no poses")
    p.add_argument("--max-seconds", type=float, default=None, help="use only the first S seconds")
    args = p.parse_args(argv)

    try:
        summary = run_clip(
            args.source,
            args.out_dir,
            args.models,
            poses=args.poses,
            dancers=args.dancers,
            max_seconds=args.max_seconds,
            scorer=args.scorer,
        )
    except FileNotFoundError as err:
        score.stop(str(err))
    print(score.format_table(summary))
    print(f"events: {summary['events_file']}, {summary['events']} events")
    print(f"scores: {summary['scores_file']}, {summary['bins']} bins, regions from {summary['regions']}")
    print(f"summary: {summary['summary_file']}")
    print(f"elapsed: {summary['elapsed_s']:.1f} s")


if __name__ == "__main__":
    main()

"""One command: clip -> episode folder with events.h5, scores.npy, scores_meta.json.

    python -m dance.run CLIP [--episodes-dir episodes] [--scorer spiking|baseline]
        [--models models/dance] [--dancers N] [--max-seconds S] [--preview]
    python -m dance.run EPISODE_DIR [same options]

All output for a clip goes in `<episodes-dir>/<clip name>/`. The clip name is
the file name without its suffix.

With an episode folder as the target, the clip is `clip.mp4` in that folder.
Other videos in the folder (skeleton.mp4, robot.mp4, panel.mp4, preview.mp4)
are then not used. If there is no `clip.mp4`, the one other video is the
clip; more than one other video is an error. If the folder has no video but
has an `events.h5`, the scores come from that file.

Read from the episode folder, when present (the pose build writes them):
    poses.npy   (frames, dancers, 17, 3), pixels of the ORIGINAL video,
                leftmost dancer = 0, missing joints have confidence 0
    meta.json   fps, width, height, n_frames, dancers, clip_id of the original
                video. `dancers` is the number of vertical strips when
                --dancers is not given and there is no poses.npy.

Written to the episode folder:
    events.h5          eventify HDF5 export, coordinates in sensor pixels
    scores.npy         float32 (bins, dancers), values 0 to 1
    sync.npy           float32 (bins, dancers), values 0 to 2; only for 2 or more dancers
    preview.mp4        with --preview: the event view with boxes and score traces
    scores_meta.json   bin_ms, bins, sensor size, scorer, threshold, ranking

`meta.json` and `pose3d.npz` are never written or changed here.

The trained scorer is loaded before the event work. When its artefacts are
missing, the program stops with an error and writes no file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import score, stream
from .features import read_pose_meta

EVENTS_NAME = "events.h5"
SCORES_NAME = "scores.npy"
SYNC_NAME = "sync.npy"
SCORES_META_NAME = "scores_meta.json"
POSES_NAME = "poses.npy"
POSE_META_NAME = "meta.json"
CLIP_NAME = "clip.mp4"
PREVIEW_NAME = "preview.mp4"
DEFAULT_EPISODES = "episodes"


def resolve_episode(target: str | Path, episodes_dir: str | Path = DEFAULT_EPISODES) -> tuple[Path, Path, bool]:
    """Target -> (episode folder, source to read, True if the source is a clip).

    A clip gives the folder `<episodes_dir>/<clip name>`. A folder is used as it is.
    """
    target = Path(target)
    if target.is_dir():
        if (target / CLIP_NAME).is_file():
            return target, target / CLIP_NAME, True
        clips = sorted(
            f for f in target.iterdir()
            if f.suffix.lower() in stream.VIDEO_SUFFIXES and f.name != PREVIEW_NAME
        )
        if len(clips) > 1:
            names = ", ".join(c.name for c in clips)
            raise FileNotFoundError(
                f"episode folder {target} has no {CLIP_NAME} and more than one other video: {names}. "
                f"Name the source clip {CLIP_NAME}."
            )
        if clips:
            return target, clips[0], True
        events = target / EVENTS_NAME
        if events.exists():
            return target, events, False
        raise FileNotFoundError(
            f"episode folder {target} has no video and no {EVENTS_NAME}. "
            "Give the clip: python -m dance.run CLIP --episodes-dir DIR"
        )
    if not target.exists():
        raise FileNotFoundError(f"clip or episode folder not found: {target}")
    return Path(episodes_dir) / target.stem, target, True


def run_clip(
    target: str | Path,
    episodes_dir: str | Path = DEFAULT_EPISODES,
    model_dir: str | Path = score.DEFAULT_MODELS,
    *,
    dancers: int | None = None,
    max_seconds: float | None = None,
    model: Any = None,
    scorer: str = score.DEFAULT_SCORER,
) -> dict:
    """Clip or episode folder -> events.h5, scores.npy, scores_meta.json. Returns the summary.

    `dancers` is the number of vertical strips when the episode has no
    poses.npy. Without it, the `dancers` value of meta.json is used, else 1.
    """
    start = time.perf_counter()
    episode, source, is_clip = resolve_episode(target, episodes_dir)
    if model is None:
        model = score.load(model_dir, scorer)      # before any event work and any output

    # Poses and their meta file come only from the episode folder.
    poses_path = episode / POSES_NAME
    meta_path = episode / POSE_META_NAME
    has_poses = poses_path.exists()
    pose_meta = read_pose_meta(meta_path) if meta_path.exists() else None
    dancers_from = "--dancers"
    if dancers is None:
        from_meta = pose_meta["dancers"] if pose_meta else None
        dancers, dancers_from = (from_meta, POSE_META_NAME) if from_meta else (1, "the default")
    episode.mkdir(parents=True, exist_ok=True)
    events_path = episode / EVENTS_NAME
    scores, sync, summary = score.score_clip_full(
        source,
        model_dir,
        poses=poses_path if has_poses else None,
        auto_poses=False,
        pose_meta=pose_meta if has_poses else None,
        dancers=dancers,
        max_seconds=max_seconds,
        events_out=events_path if is_clip else None,
        model=model,
        scorer=scorer,
    )
    np.save(episode / SCORES_NAME, scores)
    sync_meta = summary["sync"]
    if sync is not None:
        np.save(episode / SYNC_NAME, sync)
        sync_meta = {"file": SYNC_NAME, **sync_meta}
    elif (episode / SYNC_NAME).exists():
        (episode / SYNC_NAME).unlink()         # an old file of a run with more dancers

    height, width = summary["sensor_shape"]
    scores_meta = {
        "bin_ms": summary["bin_ms"],
        "bins": summary["bins"],
        "sensor_width": int(width),
        "sensor_height": int(height),
        "scorer": scorer,
        "threshold": summary["threshold"],
        "dancers": len(summary["dancers"]),
        "regions": summary["regions"],
        "pose_scale": summary.get("pose_scale"),
        "ranking": summary["ranking"],
        "per_dancer": summary["dancers"],
        "source": Path(source).name,
        "events_file": EVENTS_NAME,
        "scores_file": SCORES_NAME,
        "sync": sync_meta,
    }
    (episode / SCORES_META_NAME).write_text(json.dumps(scores_meta, indent=2) + "\n")

    summary.update(
        dancers_from=dancers_from,
        episode=str(episode),
        events_file=str(events_path),
        scores_file=str(episode / SCORES_NAME),
        scores_meta_file=str(episode / SCORES_META_NAME),
        elapsed_s=round(time.perf_counter() - start, 2),
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Clip -> episode folder with events, scores and a ranked table")
    p.add_argument("target", help="video clip, or an episode folder")
    p.add_argument("--episodes-dir", default=DEFAULT_EPISODES, help="for a clip: output goes in EPISODES_DIR/<clip name>/")
    p.add_argument("--models", default=score.DEFAULT_MODELS, help="folder with the trained artefacts")
    p.add_argument("--scorer", choices=score.SCORERS, default=score.DEFAULT_SCORER, help="which trained scorer to use")
    p.add_argument("--dancers", type=int, default=None,
                   help="number of vertical strips when the episode has no poses.npy; "
                        "default: `dancers` in meta.json, else 1")
    p.add_argument("--max-seconds", type=float, default=None, help="use only the first S seconds")
    p.add_argument("--preview", action="store_true", help="also write preview.mp4 in the episode folder")
    args = p.parse_args(argv)

    try:
        summary = run_clip(
            args.target,
            args.episodes_dir,
            args.models,
            dancers=args.dancers,
            max_seconds=args.max_seconds,
            scorer=args.scorer,
        )
    except FileNotFoundError as err:
        score.stop(str(err))
    print(score.format_table(summary))
    print(f"episode: {summary['episode']}")
    if "events" in summary:
        print(f"events: {summary['events_file']}, {summary['events']} events")
    print(f"scores: {summary['scores_file']}, {summary['bins']} bins, {args.scorer} scorer")
    if summary["sync"] is not None:
        print(f"sync: {Path(summary['episode']) / SYNC_NAME}")
    else:
        print("sync: not written. The sync score needs 2 or more dancers.")
    print(score.regions_line(summary))
    if summary["regions"] == "strips":
        print(f"dancer count: {len(summary['dancers'])}, from {summary['dancers_from']}")
    print(f"scores meta: {summary['scores_meta_file']}")
    print(f"elapsed: {summary['elapsed_s']:.1f} s")
    if args.preview:
        from .preview import write_preview

        print(f"preview: {write_preview(summary['episode'])}")


if __name__ == "__main__":
    main()

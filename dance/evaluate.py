"""Score a set of clips and print the good/bad table, by dancer count.

    python -m dance.evaluate CLIPS_DIR [CLIPS_DIR ...] [--models models/dance]
        [--scorer spiking|baseline|both] [--dancers N]

Folder layout, the same as for `dance.train`:
    CLIPS_DIR/1p/*.mp4   clips with 1 dancer
    CLIPS_DIR/2p/*.mp4   clips with 2 dancers
    CLIPS_DIR/3p/*.mp4   clips with 3 dancers
    CLIPS_DIR/*.mp4      flat form: all clips have `--dancers N` dancers

The role of a clip comes from its file name:
    bad_*      a clip of bad dancing
    clean_*    a clean clip
    other      any other name

Each cell of the table is "mean score / fraction of bins above p90".
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .sync import sync_score

P90 = 0.90
SUBFOLDER = re.compile(r"^([1-9])p$")


@dataclass
class Clip:
    path: Path
    dancers: int
    role: str                          # "clean", "bad" or "other"
    features: np.ndarray | None = None

    @property
    def group(self) -> str:
        return f"{self.dancers}p"


def role_of(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("bad_"):
        return "bad"
    return "clean" if name.startswith("clean_") else "other"


def clip_files(folder: Path) -> list[Path]:
    """Video files and eventify HDF5 files in one folder, sorted by name.

    If a video and an events file have the same stem, only the events file is used.
    """
    from . import stream

    suffixes = set(stream.VIDEO_SUFFIXES) | set(stream.HDF5_SUFFIXES)
    found = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith(".")
    )
    events_stems = {p.stem for p in found if p.suffix.lower() in stream.HDF5_SUFFIXES}
    return [p for p in found if p.suffix.lower() in stream.HDF5_SUFFIXES or p.stem not in events_stems]


def collect_clips(roots: list[str | Path], dancers: int = 1) -> list[Clip]:
    """All clips of the given folders, with their dancer count and role.

    Sub-folders `1p`, `2p`, `3p` give the dancer count. Clips directly in a
    folder have `dancers` dancers (the flat form).
    """
    clips: list[Clip] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"clips folder not found: {root}")
        clips += [Clip(p, dancers, role_of(p)) for p in clip_files(root)]
        for sub in sorted(d for d in root.iterdir() if d.is_dir() and SUBFOLDER.match(d.name)):
            count = int(SUBFOLDER.match(sub.name).group(1))
            clips += [Clip(p, count, role_of(p)) for p in clip_files(sub)]
    return clips


def add_features(clips: list[Clip], max_seconds: float | None = None, jobs: int = 4) -> None:
    """Compute the features of all clips, in parallel worker processes."""
    from concurrent.futures import ProcessPoolExecutor

    from .features import clip_features_job

    todo = [c for c in clips if c.features is None]
    work = [(str(c.path), c.dancers, max_seconds) for c in todo]
    if jobs <= 1 or len(work) <= 1:
        results = [clip_features_job(job) for job in work]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(work))) as pool:
            results = list(pool.map(clip_features_job, work))
    for clip, (features, info) in zip(todo, results):
        clip.features = features
        clip.dancers = features.shape[1]      # poses can give another count than the folder name


def _cell(scores: np.ndarray) -> str:
    return f"{scores.mean():.3f} / {(scores >= P90).mean():.3f}"


def report(clips: list[Clip], models: dict[str, object], roles: dict[str, str] | None = None) -> str:
    """The table by dancer count. `models` maps a scorer name to a loaded model.

    `roles` gives a display name for each clip path (for example "held-out
    clean"). Without it, the role of the clip is used.
    """
    names = list(models)
    lines: list[str] = []
    groups = sorted({c.group for c in clips})
    for group in groups:
        members = [c for c in clips if c.group == group]
        lines.append(f"\n### {group[0]} dancer(s): {len(members)} clip(s)")
        head = ["Role", "Clip", "Dancer", "Bins"] + names + ["Mean sync"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        pooled: dict[str, dict[str, list[np.ndarray]]] = {}
        for clip in members:
            role = (roles or {}).get(str(clip.path), clip.role)
            scores = {n: np.asarray(m.score(clip.features)) for n, m in models.items()}
            sync = sync_score(clip.features)
            for d in range(clip.features.shape[1]):
                cells = [_cell(scores[n][:, d]) for n in names]
                sync_cell = "-" if sync is None else f"{sync[:, d].mean():.3f}"
                lines.append(f"| {role} | {clip.path.stem} | {d} | {len(clip.features)} | "
                             + " | ".join(cells) + f" | {sync_cell} |")
            for n in names:
                pooled.setdefault(role, {}).setdefault(n, []).append(scores[n].ravel())
        for role, by_scorer in pooled.items():
            cells = [_cell(np.concatenate(by_scorer[n])) for n in names]
            count = sum(1 for c in members if (roles or {}).get(str(c.path), c.role) == role)
            lines.append(f"| **{role}, all {count} clip(s)** | | all | | " + " | ".join(cells) + " | |")
    return "\n".join(lines)


def load_models(model_dir: str | Path, scorer: str) -> dict[str, object]:
    from . import score

    wanted = ("spiking", "baseline") if scorer == "both" else (scorer,)
    return {name: score.load(model_dir, name) for name in wanted}


def main(argv: list[str] | None = None) -> None:
    import os

    p = argparse.ArgumentParser(description="Score clips with the trained scorers and print the table by dancer count")
    p.add_argument("clips_dir", nargs="+", help="folder with sub-folders 1p, 2p, 3p, or with clips directly in it")
    p.add_argument("--models", default="models/dance", help="folder with the trained artefacts")
    p.add_argument("--scorer", choices=("spiking", "baseline", "both"), default="both")
    p.add_argument("--dancers", type=int, default=1, help="flat form: number of vertical strips")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = p.parse_args(argv)

    try:
        models = load_models(args.models, args.scorer)
        clips = collect_clips(args.clips_dir, args.dancers)
    except FileNotFoundError as err:
        raise SystemExit(f"error: {err}")
    if not clips:
        raise SystemExit(f"error: no clips in {', '.join(args.clips_dir)}")
    add_features(clips, args.max_seconds, args.jobs)
    print('Each cell is "mean score / fraction of bins above p90".')
    print(report(clips, models))


if __name__ == "__main__":
    main()

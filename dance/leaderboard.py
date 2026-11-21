"""Leaderboard for single-dancer episodes: a ranked table of display scores.

    python -m dance.leaderboard EPISODE_DIR [EPISODE_DIR ...] [--names NAME ...]
        [--out episodes/leaderboard.png] [--seconds 5]

Each episode folder must hold `scores.npy` and `scores_meta.json` of ONE
dancer (run `python -m dance.run` on the clip first). Rank 1 is the best
dance: the highest display score. See `dance/display.py` for the display score.

Written next to `--out`:
    leaderboard.png    1280x720, the ranked table
    leaderboard.mp4    the same picture as a short video: the final screen
    leaderboard.json   the ranking, and `lowest`: the worst episode (for the robot replay)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .display import display_score, load_remap

WIDTH, HEIGHT, FPS = 1280, 720, 25
BACKGROUND = (14, 12, 10)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
GOLD = (40, 200, 255)       # the warm yellow of the ON events
BLUE = (255, 120, 40)       # the blue of the OFF events
BAR_BACK = (40, 36, 32)


def read_episode(episode: str | Path) -> dict:
    """Scores of one single-dancer episode folder."""
    episode = Path(episode)
    scores_path, meta_path = episode / "scores.npy", episode / "scores_meta.json"
    missing = [p.name for p in (scores_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{episode} has no {', '.join(missing)}. Run `python -m dance.run {episode}` first."
        )
    scores = np.load(scores_path)
    if scores.ndim != 2 or scores.shape[1] != 1:
        raise ValueError(
            f"{episode}: the leaderboard is for one dancer per clip, but scores.npy has shape {scores.shape}"
        )
    if scores.shape[0] == 0:
        raise ValueError(f"{episode}: scores.npy has no time bins")
    meta = json.loads(meta_path.read_text())
    return {
        "episode": str(episode),
        "mean_score": float(scores.mean()),
        "fraction_above_p90": float((scores >= float(meta.get("threshold", 0.90))).mean()),
        "scorer": meta.get("scorer"),
        "models": meta.get("models"),
    }


def rank(episodes: list[str | Path], names: list[str] | None = None, models: str | Path | None = None) -> dict:
    """Read the episodes and rank them. Rank 1 has the highest display score."""
    if not episodes:
        raise ValueError("give at least 1 episode folder")
    if names is not None and len(names) != len(episodes):
        raise ValueError(f"{len(names)} names for {len(episodes)} episode folders")
    rows = [read_episode(e) for e in episodes]
    scorers = {r["scorer"] for r in rows}
    folders = {r["models"] for r in rows} if models is None else {str(models)}
    if len(scorers) > 1 or len(folders) > 1:
        raise ValueError(
            "the episodes were scored with different scorers or model folders "
            f"(scorers {sorted(map(str, scorers))}, models {sorted(map(str, folders))}). Score them the same way."
        )
    scorer, model_dir = scorers.pop(), folders.pop()
    remap = load_remap(model_dir, scorer) if model_dir else load_remap(".", "none")
    for i, row in enumerate(rows):
        row["name"] = names[i] if names else Path(row["episode"]).name
        row["display_score"] = round(display_score(row["mean_score"], remap), 1)
    ranking = sorted(rows, key=lambda r: (-r["display_score"], r["mean_score"]))
    for i, row in enumerate(ranking):
        row["rank"] = i + 1
    keys = ("rank", "name", "episode", "display_score", "mean_score", "fraction_above_p90")
    return {
        "ranking": [{k: r[k] for k in keys} for r in ranking],
        "lowest": {"name": ranking[-1]["name"], "episode": ranking[-1]["episode"]},
        "scorer": scorer,
        "models": model_dir,
        "remap": remap,
    }


def format_table(board: dict) -> str:
    head = ("rank", "name", "display score", "mean unusualness", "fraction above p90")
    rows = [
        (str(r["rank"]), r["name"], f"{r['display_score']:.1f}", f"{r['mean_score']:.3f}",
         f"{r['fraction_above_p90']:.3f}")
        for r in board["ranking"]
    ]
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
    lines = ["  ".join(h.ljust(w) if i == 1 else h.rjust(w) for i, (h, w) in enumerate(zip(head, widths)))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) if i == 1 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
              for r in rows]
    return "\n".join(lines)


def render(board: dict) -> np.ndarray:
    """The leaderboard as one 1280x720 BGR picture."""
    img = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "MACARENA LEADERBOARD", (70, 90), font, 1.5, TEXT, 3, cv2.LINE_AA)
    cv2.putText(img, "display score 0 to 100: higher is a better dance", (72, 130), font, 0.6, DIM, 1, cv2.LINE_AA)
    rows = board["ranking"][:8]
    top, step = 190, min(70, (HEIGHT - 290) // max(len(rows), 1))
    bar_x0, bar_x1 = 520, 1040
    for i, row in enumerate(rows):
        y = top + i * step
        best, worst = i == 0, i == len(rows) - 1 and len(rows) > 1
        colour = GOLD if best else (BLUE if worst else TEXT)
        cv2.putText(img, f"{row['rank']}", (80, y + 12), font, 1.1, colour, 2, cv2.LINE_AA)
        cv2.putText(img, str(row["name"])[:22], (150, y + 12), font, 1.0, colour, 2, cv2.LINE_AA)
        cv2.rectangle(img, (bar_x0, y - 14), (bar_x1, y + 14), BAR_BACK, -1)
        end = bar_x0 + int(round((bar_x1 - bar_x0) * row["display_score"] / 100.0))
        if end > bar_x0:
            cv2.rectangle(img, (bar_x0, y - 14), (end, y + 14), colour, -1)
        cv2.putText(img, f"{row['display_score']:.0f}", (bar_x1 + 30, y + 14), font, 1.2, colour, 2, cv2.LINE_AA)
    if len(board["ranking"]) > len(rows):
        cv2.putText(img, f"+ {len(board['ranking']) - len(rows)} more", (150, top + len(rows) * step + 10),
                    font, 0.6, DIM, 1, cv2.LINE_AA)
    remap = board["remap"]
    fit = (f"fitted on {remap['n_clean']} clean and {remap['n_bad']} bad takes" if remap.get("fitted")
           else "default values, not fitted")
    foot = [
        f"Scorer: {board['scorer']}. Models: {board['models']}. Display remap: {fit}.",
        "The score comes from a simulated event-camera stream. It measures how unusual the motion is.",
    ]
    for i, line in enumerate(foot):
        cv2.putText(img, line, (72, HEIGHT - 60 + 26 * i), font, 0.5, DIM, 1, cv2.LINE_AA)
    return img


def write(board: dict, out: str | Path, seconds: float = 5.0) -> dict[str, Path]:
    """Write leaderboard.png, .mp4 and .json next to `out`."""
    out = Path(out)
    if out.suffix.lower() != ".png":
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    picture = render(board)
    if not cv2.imwrite(str(out), picture):
        raise RuntimeError(f"OpenCV cannot write the picture: {out}")
    video = out.with_suffix(".mp4")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write the video: {video}")
    for _ in range(max(1, int(round(seconds * FPS)))):
        writer.write(picture)
    writer.release()
    data = out.with_suffix(".json")
    data.write_text(json.dumps(board, indent=2) + "\n")
    return {"png": out, "mp4": video, "json": data}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ranked table of display scores for single-dancer episodes")
    p.add_argument("episodes", nargs="+", help="episode folders, each with scores.npy and scores_meta.json")
    p.add_argument("--names", nargs="+", default=None, help="one display name for each episode, in the same order")
    p.add_argument("--out", default="episodes/leaderboard.png", help="picture path; .mp4 and .json get the same stem")
    p.add_argument("--models", default=None, help="model folder for the display remap; default: the one in scores_meta.json")
    p.add_argument("--seconds", type=float, default=5.0, help="length of the final-screen video")
    args = p.parse_args(argv)
    try:
        board = rank(args.episodes, args.names, args.models)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"error: {err}")
    files = write(board, args.out, args.seconds)
    print(format_table(board))
    if not board["remap"].get("fitted"):
        print("NOTE: the display remap is not fitted for this model folder. Default values are used.")
    print(f"lowest ranked: {board['lowest']['name']} ({board['lowest']['episode']})")
    print(f"picture: {files['png']}")
    print(f"final screen video: {files['mp4']}")
    print(f"data: {files['json']}")


if __name__ == "__main__":
    main()

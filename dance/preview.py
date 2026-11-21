"""Fallback preview: the event view with dancer boxes and score traces.

    python -m dance.preview EPISODE_DIR [--out preview.mp4]

Reads from the episode folder: events.h5, scores.npy, scores_meta.json, and
sync.npy, poses.npy, meta.json when present. Writes preview.mp4 there.

The picture:
  - top: the events on a near-black background. ON events are warm yellow, OFF
    events are blue. Events fade quickly. Each dancer has a box and a label.
  - bottom: the score of each dancer over time, with the threshold line. With
    2 or more dancers, a second strip shows the sync score.

Plain OpenCV, no polish. The same renderer shows a window in live mode
(`python -m dance.score --live --preview`).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np

from . import stream
from .contracts import BIN_MS
from .features import Regions, read_pose_meta

BACKGROUND = (14, 12, 10)                  # BGR, near black
ON_BGR = np.array([40, 200, 255], dtype=np.float32)    # warm yellow
OFF_BGR = np.array([255, 120, 40], dtype=np.float32)   # blue
DANCER_BGR = [(90, 230, 90), (255, 90, 255), (240, 240, 240)]   # green, magenta, white
DECAY = 0.45            # part of the old events that stays after one 33 ms window
FULL_BRIGHT = 3.0       # events in one pixel that give full colour
VIEW_SCALE = 2          # the event view is drawn at 2 times the sensor size
SCORE_PANEL_H = 110
SYNC_PANEL_H = 70
LIVE_SPAN_BINS = 300    # live: the traces show the last 10 seconds
FPS = 1000.0 / BIN_MS


class PreviewRenderer:
    """Draws one preview picture for each 33 ms window."""

    def __init__(
        self,
        sensor_shape: tuple[int, int],
        dancers: int,
        threshold: float = 0.90,
        span_bins: int | None = None,
        with_sync: bool = False,
        scroll: bool = False,
    ):
        self.h, self.w = sensor_shape
        self.dancers = dancers
        self.threshold = threshold
        self.span = max(2, span_bins or LIVE_SPAN_BINS)
        self.scroll = scroll
        self.with_sync = with_sync and dancers >= 2
        self.view_w, self.view_h = self.w * VIEW_SCALE, self.h * VIEW_SCALE
        self.width = self.view_w
        self.height = self.view_h + SCORE_PANEL_H + (SYNC_PANEL_H if self.with_sync else 0)
        self._on = np.zeros((self.h, self.w), dtype=np.float32)
        self._off = np.zeros((self.h, self.w), dtype=np.float32)
        self._scores: list[np.ndarray] = []
        self._sync: list[np.ndarray] = []
        self._count = 0

    def set_scores(self, scores: np.ndarray | None, sync: np.ndarray | None = None) -> None:
        """Live mode: the newest score (and sync) values, used until the next call."""
        self._live_scores = None if scores is None else np.asarray(scores, dtype=np.float32)
        self._live_sync = None if sync is None else np.asarray(sync, dtype=np.float32)

    _live_scores: np.ndarray | None = None
    _live_sync: np.ndarray | None = None

    def frame(
        self,
        window: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray | None = None,
        sync: np.ndarray | None = None,
    ) -> np.ndarray:
        """One (1, 2, H, W) window + (dancers, 4) boxes -> one BGR uint8 picture.

        `scores` and `sync` are the values of this window, shape (dancers,).
        Without them, the values of the last `set_scores` call are used.
        """
        scores = self._live_scores if scores is None else scores
        sync = self._live_sync if sync is None else sync
        nan = np.full(self.dancers, np.nan, dtype=np.float32)
        self._scores.append(nan if scores is None else np.asarray(scores, dtype=np.float32))
        self._sync.append(nan if sync is None else np.asarray(sync, dtype=np.float32))
        self._count += 1

        self._on = self._on * DECAY + window[0, 1]
        self._off = self._off * DECAY + window[0, 0]
        on = np.clip(self._on / FULL_BRIGHT, 0.0, 1.0)[..., None]
        off = np.clip(self._off / FULL_BRIGHT, 0.0, 1.0)[..., None]
        view = np.array(BACKGROUND, dtype=np.float32) * (1.0 - np.maximum(on, off)) + on * ON_BGR + off * OFF_BGR
        view = np.clip(view, 0, 255).astype(np.uint8)
        view = cv2.resize(view, (self.view_w, self.view_h), interpolation=cv2.INTER_NEAREST)

        last = self._scores[-1]
        for d in range(min(self.dancers, len(boxes))):
            colour = DANCER_BGR[d % len(DANCER_BGR)]
            x0, y0, x1, y1 = (int(v) * VIEW_SCALE for v in boxes[d])
            cv2.rectangle(view, (x0, y0), (x1 - 1, y1 - 1), colour, 1)
            label = f"d{d}" if np.isnan(last[d]) else f"d{d} {last[d]:.2f}"
            if not np.isnan(last[d]) and last[d] >= self.threshold:
                label += " !"
            cv2.putText(view, label, (x0 + 4, max(y0 + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        seconds = self._count * BIN_MS / 1000
        cv2.putText(view, f"{seconds:6.2f} s", (self.view_w - 90, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)

        panels = [view, self._panel(self._scores, SCORE_PANEL_H, 1.0, self.threshold, "score", "p90")]
        if self.with_sync:
            panels.append(self._panel(self._sync, SYNC_PANEL_H, 2.0, 1.0, "sync", "1.0"))
        return np.vstack(panels)

    def _panel(self, rows: list[np.ndarray], height: int, top: float, line: float, name: str, line_name: str) -> np.ndarray:
        """One trace strip: a value from 0 to `top`, with a marked horizontal line."""
        panel = np.full((height, self.width, 3), (26, 24, 22), dtype=np.uint8)
        pad = 6

        def y_of(value: float) -> int:
            return int(round(height - 1 - pad - (height - 1 - 2 * pad) * min(max(value / top, 0.0), 1.0)))

        cv2.line(panel, (0, y_of(line)), (self.width - 1, y_of(line)), (70, 70, 200), 1)
        cv2.putText(panel, line_name, (self.width - 34, y_of(line) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (110, 110, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, name, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
        first = max(0, len(rows) - self.span) if self.scroll else 0
        data = np.asarray(rows[first:], dtype=np.float32)
        xs = np.arange(len(data)) * (self.width - 1) / (self.span - 1)
        for d in range(self.dancers):
            ok = ~np.isnan(data[:, d])
            if ok.sum() < 2:
                continue
            points = np.stack([xs[ok], [y_of(v) for v in data[ok, d]]], axis=1).astype(np.int32)
            cv2.polylines(panel, [points.reshape(-1, 1, 2)], False, DANCER_BGR[d % len(DANCER_BGR)], 1, cv2.LINE_AA)
        if not self.scroll and len(data):
            x = int(xs[-1])
            cv2.line(panel, (x, 0), (x, height - 1), (90, 90, 90), 1)
        return panel


def episode_regions(episode: Path, meta: dict, sensor_shape: tuple[int, int], dancers: int) -> Regions:
    """The same dancer regions as the scoring run of this episode."""
    poses_path = episode / "poses.npy"
    if meta.get("regions") == "poses" and poses_path.exists():
        poses = np.load(poses_path)
        scale: float | tuple[float, float] = tuple(meta.get("pose_scale") or (1.0, 1.0))
        fps = 30.0
        if (episode / "meta.json").exists():
            pose_meta = read_pose_meta(episode / "meta.json")
            scale = (sensor_shape[1] / pose_meta["width"], sensor_shape[0] / pose_meta["height"])
            fps = pose_meta["fps"] or fps
        return Regions(sensor_shape, poses, pose_fps=fps, scale=scale)
    return Regions(sensor_shape, None, dancers)


def write_preview(episode: str | Path, out: str | Path | None = None) -> Path:
    """Episode folder -> preview.mp4. Returns the path of the video."""
    episode = Path(episode)
    needed = [episode / "events.h5", episode / "scores.npy", episode / "scores_meta.json"]
    missing = [f.name for f in needed if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"{episode} has no {', '.join(missing)}. Run `python -m dance.run` on the clip first."
        )
    meta = json.loads((episode / "scores_meta.json").read_text())
    scores = np.load(episode / "scores.npy")
    sync = np.load(episode / "sync.npy") if (episode / "sync.npy").exists() else None
    events, sensor_shape = stream.read_hdf5(episode / "events.h5")
    dancers = scores.shape[1]
    regions = episode_regions(episode, meta, sensor_shape, dancers)
    renderer = PreviewRenderer(
        sensor_shape, dancers, float(meta.get("threshold", 0.90)), span_bins=len(scores), with_sync=sync is not None
    )
    out = Path(out) if out is not None else episode / "preview.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (renderer.width, renderer.height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write the video: {out}")
    try:
        windows = stream.window_frames(
            stream.split_by_window(events), sensor_shape, chunk_dt_us=stream.WINDOW_US, n_windows=len(scores)
        )
        for k, window in enumerate(windows):
            if k >= len(scores):
                break
            boxes = regions.boxes(k, window)
            writer.write(renderer.frame(window, boxes, scores[k], None if sync is None else sync[k]))
    finally:
        writer.release()
    return out


def live_tap(
    windows: Iterable[np.ndarray],
    regions: Regions,
    renderer: PreviewRenderer,
    show: Callable[[np.ndarray], bool],
) -> Iterator[np.ndarray]:
    """Pass the live windows through and show a preview picture for each one.

    `show` gets the picture and returns False to stop the stream (the q key).
    """
    for k, window in enumerate(windows):
        if not show(renderer.frame(window, regions.boxes(k))):
            return
        yield window


def window_show(title: str = "dance preview: press q to stop") -> Callable[[np.ndarray], bool]:
    """Show pictures in an OpenCV window. The q key stops."""
    def show(picture: np.ndarray) -> bool:
        cv2.imshow(title, picture)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
    return show


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Episode folder -> preview.mp4 (events, boxes, score traces)")
    p.add_argument("episode", help="episode folder with events.h5, scores.npy and scores_meta.json")
    p.add_argument("--out", default=None, help="video path; default: EPISODE/preview.mp4")
    args = p.parse_args(argv)
    try:
        out = write_preview(args.episode, args.out)
    except FileNotFoundError as err:
        raise SystemExit(f"error: {err}")
    print(f"preview: {out}")


if __name__ == "__main__":
    main()

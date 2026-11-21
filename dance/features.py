"""Step 2: binned events -> 13 features per time window per dancer.

Input: event-count tensors in eventify's `frame_stream` layout,
(n_bins, 2, H, W), channel 0 = OFF, channel 1 = ON. One bin is one 33 ms
window. The tensor is cropped to each dancer's box.

Dancer regions:
  - With poses (frames, dancers, 17, 3; COCO keypoints; x, y, confidence):
    the keypoint bounding box of each dancer, padded 15% on each side.
    Poses are the preferred source.
  - Without poses: equal vertical strips, one per dancer. Inside its strip,
    the box of a dancer centres itself on the dancer: its centre and size
    follow a slow running median (about 5 s) of the event positions.

There are 26 numbers per bin per dancer: 13 features, then their 13
bin-to-bin deltas (feature[k] - feature[k - 1], 0 for the first bin).

The 13 features, with coordinates normalised to the box:
   0      rate        event count against the dancer's own running mean (about
                      10 s): r / (1 + r) with r = count / mean. 0..1, and 0.5
                      is the dancer's own mean. 0 on an empty bin.
   1      polarity    (ON - OFF) / count, -1..1, 0 on an empty bin
   2      centroid_x  mean x of events, 0..1, 0.5 on an empty bin
   3      centroid_y  mean y of events, 0..1, 0.5 on an empty bin
   4..12  grid        event counts in a 3x3 grid (row-major, top-left first),
                      normalised to sum to 1, all 0 on an empty bin

Output shape: (bins, dancers, 26), float32.

A stream that comes window by window uses `base_features` on each window
`StreamFeatures`, which keeps the running values, and `add_deltas` on the
stacked rows. `extract_features` does all of it on one tensor with given boxes.

CLI: python -m dance.features clip.mp4|events.h5 features.npy [--poses poses.npy] [--dancers N]
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from . import stream
from .contracts import FEATURE_DIM

BASE_DIM = 13
NUM_FEATURES = FEATURE_DIM
BASE_NAMES: tuple[str, ...] = (
    "rate", "polarity", "centroid_x", "centroid_y",
    "grid_00", "grid_01", "grid_02",
    "grid_10", "grid_11", "grid_12",
    "grid_20", "grid_21", "grid_22",
)
FEATURE_NAMES: tuple[str, ...] = BASE_NAMES + tuple(f"d_{name}" for name in BASE_NAMES)
assert FEATURE_DIM == 2 * BASE_DIM == len(FEATURE_NAMES)
BOX_PAD = 0.15
MIN_KEYPOINT_CONF = 0.3
MIN_BOX_SIZE = 6       # pixels; the 3x3 grid needs at least 3
RATE_WINDOWS = 300     # windows in the running mean of the event count: about 10 s
CENTRE_WINDOWS = 150   # windows in the running median of the event positions: about 5 s
BOX_STD = 2.5          # half size of a self-centred box, in standard deviations of the event positions
MIN_CENTRE_EVENTS = 20 # a window with fewer events does not move the box


def strip_boxes(width: int, height: int, dancers: int) -> np.ndarray:
    """Equal vertical strips. Returns (dancers, 4) ints: x0, y0, x1, y1."""
    if dancers < 1:
        raise ValueError("dancers must be >= 1")
    if width < dancers * MIN_BOX_SIZE:
        raise ValueError(f"frame width {width} is too small for {dancers} dancers")
    edges = np.rint(np.linspace(0, width, dancers + 1)).astype(np.int64)
    return np.stack(
        [edges[:-1], np.zeros(dancers, np.int64), edges[1:], np.full(dancers, height, np.int64)],
        axis=1,
    )


def pose_boxes(
    poses: np.ndarray,
    width: int,
    height: int,
    scale: float | tuple[float, float] = 1.0,
    pad: float = BOX_PAD,
    min_conf: float = MIN_KEYPOINT_CONF,
) -> np.ndarray:
    """Padded keypoint bounding box per pose frame per dancer.

    Returns (frames, dancers, 4) ints: x0, y0, x1, y1 in sensor pixels.
    Pose x, y are pixels of the ORIGINAL video. They are multiplied by `scale`
    to get sensor pixels. `scale` is one number, or (scale_x, scale_y).
    If every coordinate is <= 1.5, they are taken as 0..1 fractions instead.
    A joint with confidence 0 is a missing joint and is not used.
    A frame with fewer than 2 confident keypoints reuses the nearest good box.
    A dancer with no good frame at all gets a vertical strip.
    """
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 4 or poses.shape[2] != 17 or poses.shape[3] != 3:
        raise ValueError(f"poses must have shape (frames, dancers, 17, 3), got {poses.shape}")
    n_frames, dancers = poses.shape[:2]
    if n_frames == 0 or dancers == 0:
        raise ValueError(f"poses has no frames or no dancers: {poses.shape}")

    xy = poses[..., :2].copy()
    valid = (poses[..., 2] > min_conf) & np.isfinite(xy).all(axis=-1)
    if valid.any() and float(xy[valid].max()) <= 1.5:
        xy *= np.array([width, height], dtype=np.float32)
    else:
        sx, sy = scale if isinstance(scale, tuple) else (scale, scale)
        xy *= np.array([sx, sy], dtype=np.float32)

    x = np.where(valid, xy[..., 0], np.nan)
    y = np.where(valid, xy[..., 1], np.nan)
    enough = valid.sum(axis=2) >= 2
    boxes = np.zeros((n_frames, dancers, 4), dtype=np.float64)
    strips = strip_boxes(width, height, dancers)
    for d in range(dancers):
        good = np.nonzero(enough[:, d])[0]
        if good.size == 0:
            boxes[:, d] = strips[d]
            continue
        x0, x1 = np.nanmin(x[good, d], axis=1), np.nanmax(x[good, d], axis=1)
        y0, y1 = np.nanmin(y[good, d], axis=1), np.nanmax(y[good, d], axis=1)
        bw, bh = x1 - x0, y1 - y0
        padded = np.stack([x0 - pad * bw, y0 - pad * bh, x1 + pad * bw, y1 + pad * bh], axis=1)
        # Nearest earlier good frame, or the first good frame at the start.
        idx = np.searchsorted(good, np.arange(n_frames), side="right") - 1
        boxes[:, d] = padded[np.clip(idx, 0, None)]

    out = np.empty(boxes.shape, dtype=np.int64)
    out[..., 0] = np.clip(np.floor(boxes[..., 0]), 0, width - MIN_BOX_SIZE)
    out[..., 1] = np.clip(np.floor(boxes[..., 1]), 0, height - MIN_BOX_SIZE)
    out[..., 2] = np.clip(np.ceil(boxes[..., 2]), out[..., 0] + MIN_BOX_SIZE, width)
    out[..., 3] = np.clip(np.ceil(boxes[..., 3]), out[..., 1] + MIN_BOX_SIZE, height)
    return out


class _StripTracker:
    """The self-centred box of one dancer inside its vertical strip.

    The centre is the running median of the event centroid over the last
    CENTRE_WINDOWS windows. The half size is BOX_STD times the spread of the
    event positions about that centre, pooled over the same windows. The box
    never leaves the strip. Before the first events, the box is the whole strip.
    """

    def __init__(self, strip: np.ndarray):
        self.strip = tuple(int(v) for v in strip)
        self.box = np.array(self.strip, dtype=np.int64)
        self._history: deque[tuple[float, float, float, float]] = deque(maxlen=CENTRE_WINDOWS)
        x0, y0, x1, y1 = self.strip
        self._xs = np.arange(x1 - x0) + 0.5
        self._ys = np.arange(y1 - y0) + 0.5

    def update(self, total: np.ndarray) -> np.ndarray:
        """`total` is the (H, W) event count of one window. Returns the box."""
        x0, y0, x1, y1 = self.strip
        crop = total[y0:y1, x0:x1]
        count = float(crop.sum())
        if count < MIN_CENTRE_EVENTS:
            return self.box
        col, row = crop.sum(axis=0), crop.sum(axis=1)
        cx, cy = float(col @ self._xs) / count, float(row @ self._ys) / count
        sx = float(np.sqrt(col @ (self._xs - cx) ** 2 / count))
        sy = float(np.sqrt(row @ (self._ys - cy) ** 2 / count))
        self._history.append((cx, cy, sx, sy))
        hist = np.asarray(self._history)
        mx, my = np.median(hist[:, 0]), np.median(hist[:, 1])
        # Pooled spread: the spread inside each window plus the motion of the centroid.
        spread_x = np.sqrt(np.mean(hist[:, 2] ** 2 + (hist[:, 0] - mx) ** 2))
        spread_y = np.sqrt(np.mean(hist[:, 3] ** 2 + (hist[:, 1] - my) ** 2))
        hx = max(BOX_STD * spread_x, MIN_BOX_SIZE / 2)
        hy = max(BOX_STD * spread_y, MIN_BOX_SIZE / 2)
        bx0 = int(np.clip(np.floor(x0 + mx - hx), x0, x1 - MIN_BOX_SIZE))
        by0 = int(np.clip(np.floor(y0 + my - hy), y0, y1 - MIN_BOX_SIZE))
        bx1 = int(np.clip(np.ceil(x0 + mx + hx), bx0 + MIN_BOX_SIZE, x1))
        by1 = int(np.clip(np.ceil(y0 + my + hy), by0 + MIN_BOX_SIZE, y1))
        self.box = np.array([bx0, by0, bx1, by1], dtype=np.int64)
        return self.box


class Regions:
    """Dancer boxes for each time window."""

    def __init__(
        self,
        sensor_shape: tuple[int, int],
        poses: np.ndarray | None = None,
        dancers: int = 1,
        pose_fps: float = 30.0,
        scale: float | tuple[float, float] = 1.0,
        window_us: int = stream.WINDOW_US,
        self_centre: bool = True,
    ):
        height, width = sensor_shape
        if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
            raise ValueError(f"sensor {width}x{height} is too small")
        self.mode = "poses" if poses is not None else "strips"
        self._window_s = window_us / 1_000_000
        self._pose_fps = pose_fps
        if poses is not None:
            self._frames = pose_boxes(poses, width, height, scale)
            self.dancers = self._frames.shape[1]
        else:
            self._strips = strip_boxes(width, height, dancers)
            self._trackers = [_StripTracker(s) for s in self._strips] if self_centre else None
            self.dancers = dancers

    def boxes(self, window_index: int, frame: np.ndarray | None = None) -> np.ndarray:
        """(dancers, 4) int boxes for one window: x0, y0, x1, y1.

        In strips mode, `frame` (the (1, 2, H, W) event counts of this window)
        moves the self-centred boxes. Without `frame`, the boxes stay where they are.
        """
        if self.mode == "strips":
            if self._trackers is None:
                return self._strips
            if frame is None:
                return np.stack([t.box for t in self._trackers])
            total = frame[0, 0] + frame[0, 1]
            return np.stack([t.update(total) for t in self._trackers])
        mid_s = (window_index + 0.5) * self._window_s
        frame = int(round(mid_s * self._pose_fps))
        return self._frames[min(max(frame, 0), len(self._frames) - 1)]


def add_deltas(base: np.ndarray) -> np.ndarray:
    """(bins, dancers, 13) features -> (bins, dancers, 26) with bin-to-bin deltas.

    Delta k is base[k] - base[k - 1]. The delta of the first bin is 0.
    """
    base = np.asarray(base, dtype=np.float32)
    if base.ndim != 3 or base.shape[2] != BASE_DIM:
        raise ValueError(f"base features must have shape (bins, dancers, {BASE_DIM}), got {base.shape}")
    out = np.zeros((*base.shape[:2], FEATURE_DIM), dtype=np.float32)
    out[..., :BASE_DIM] = base
    out[1:, :, BASE_DIM:] = base[1:] - base[:-1]
    return out


class RateNormaliser:
    """Event count -> rate against the dancer's own running mean.

    The mean covers the last RATE_WINDOWS windows, with the current one.
    rate = r / (1 + r), r = count / mean. So 0.5 is the dancer's own mean.
    An empty bin gives 0.
    """

    def __init__(self, dancers: int):
        self._history: deque[np.ndarray] = deque()
        self._sum = np.zeros(dancers, dtype=np.float64)

    def __call__(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=np.float64)
        self._history.append(counts)
        self._sum += counts
        if len(self._history) > RATE_WINDOWS:
            self._sum -= self._history.popleft()
        mean = np.maximum(self._sum, 0.0) / len(self._history)
        return np.where(counts > 0, counts / (counts + np.maximum(mean, 1e-9)), 0.0)


class StreamFeatures:
    """Features for a stream that comes one window at a time.

    It keeps the running values: the self-centred boxes (in `regions`) and the
    running mean of the event count. Use one object for one clip or one live run.
    """

    def __init__(self, regions: Regions):
        self.regions = regions
        self._rate = RateNormaliser(regions.dancers)
        self._index = 0

    def push(self, frame: np.ndarray) -> np.ndarray:
        """One (1, 2, H, W) window -> (1, dancers, 13) features, without deltas."""
        boxes = self.regions.boxes(self._index, frame)
        self._index += 1
        feats, counts = _window_features(frame, boxes)
        feats[0, :, 0] = self._rate(counts[0])
        return feats


def extract_features(frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(n_bins, 2, H, W) event counts + boxes -> (n_bins, dancers, 26).

    The contract function: 13 features and their 13 bin-to-bin deltas.
    `boxes` is (dancers, 4) for all bins, or (n_bins, dancers, 4).
    The running mean for the rate starts at the first bin of `frames`.
    """
    return add_deltas(base_features(frames, boxes))


def base_features(frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(n_bins, 2, H, W) event counts + boxes -> (n_bins, dancers, 13), no deltas."""
    feats, counts = _window_features(frames, boxes)
    rate = RateNormaliser(feats.shape[1])
    for k in range(len(feats)):
        feats[k, :, 0] = rate(counts[k])
    return feats


def _window_features(frames: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features with the rate not yet set, and the event counts (n_bins, dancers).

    `boxes` is (dancers, 4) for all bins, or (n_bins, dancers, 4).
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1] != 2:
        raise ValueError(f"frames must have shape (n_bins, 2, H, W), got {frames.shape}")
    n_bins = frames.shape[0]
    boxes = np.asarray(boxes, dtype=np.int64)
    if boxes.ndim == 2:
        boxes = np.broadcast_to(boxes, (n_bins, *boxes.shape))
    dancers = boxes.shape[1]

    out = np.zeros((n_bins, dancers, BASE_DIM), dtype=np.float32)
    out[..., 2:4] = 0.5
    counts = np.zeros((n_bins, dancers), dtype=np.float64)
    for k in range(n_bins):
        off, on = frames[k, 0], frames[k, 1]
        if not (off.any() or on.any()):
            continue
        for d in range(dancers):
            x0, y0, x1, y1 = (int(v) for v in boxes[k, d])
            w, h = x1 - x0, y1 - y0
            crop_on = on[y0:y1, x0:x1]
            total = crop_on + off[y0:y1, x0:x1]
            count = float(total.sum())
            if count <= 0:
                continue
            ye = [0, h // 3, (2 * h) // 3]
            xe = [0, w // 3, (2 * w) // 3]
            bands = np.add.reduceat(total, ye, axis=0)           # (3, w)
            grid = np.add.reduceat(bands, xe, axis=1)            # (3, 3)
            col = bands.sum(axis=0)
            row = total.sum(axis=1)
            f = out[k, d]
            counts[k, d] = count
            f[1] = (2.0 * float(crop_on.sum()) - count) / count
            f[2] = float(col @ (np.arange(w) + 0.5)) / (count * w)
            f[3] = float(row @ (np.arange(h) + 0.5)) / (count * h)
            f[4:] = grid.ravel() / count
    return out, counts


def read_pose_meta(path: str | Path) -> dict:
    """Read the `meta.json` that the pose build writes. This file is never written here.

    Returns {"width", "height", "fps", "frames"} of the ORIGINAL video. `fps`
    and `frames` are None when the file does not give them.
    """
    raw = json.loads(Path(path).read_text())

    def pick(*names: str):
        for name in names:
            if raw.get(name) is not None:
                return raw[name]
        return None

    width = pick("width", "w", "frame_width", "video_width")
    height = pick("height", "h", "frame_height", "video_height")
    if width is None or height is None:
        raise ValueError(f"{path}: no width and height found. Keys in the file: {sorted(raw)}")
    fps = pick("fps", "frame_rate", "framerate", "frameRate")
    frames = pick("frames", "frame_count", "n_frames", "num_frames", "frameCount")
    return {
        "width": int(width),
        "height": int(height),
        "fps": float(fps) if fps is not None else None,
        "frames": int(frames) if frames is not None else None,
    }


def find_poses(near: str | Path) -> Path | None:
    """Look for `<stem>.poses.npy`, then `poses.npy`, next to a clip or events file."""
    near = Path(near)
    stem = near.name.split(".")[0]
    for cand in (near.with_name(f"{stem}.poses.npy"), near.with_name("poses.npy")):
        if cand.exists():
            return cand
    return None


def _open_source(
    source: str | Path, max_seconds: float | None, keep: list[np.ndarray] | None
) -> tuple[Iterator[np.ndarray], tuple[int, int], float, float]:
    """Return (window iterator, sensor_shape, fps, scale) for a clip or an HDF5 file."""
    source = Path(source)
    if source.suffix.lower() in stream.HDF5_SUFFIXES:
        events, shape = stream.read_hdf5(source)
        if max_seconds is not None:
            events = events[events["t"] < int(max_seconds * 1_000_000)]
        if keep is not None:
            keep.append(events)
        chunks = stream.split_by_window(events)
        return stream.window_frames(chunks, shape, chunk_dt_us=stream.WINDOW_US), shape, 30.0, 1.0

    info = stream.probe_source(str(source))
    chunks = stream.event_chunks(str(source), info, duration_s=max_seconds)
    if keep is not None:
        chunks = _tee(chunks, keep)
    duration_us = info.duration_us
    if max_seconds is not None and duration_us is not None:
        duration_us = min(duration_us, int(max_seconds * 1_000_000))
    n_windows = None if duration_us is None else duration_us // stream.WINDOW_US + 1
    dt = int(1_000_000 / info.fps / (stream.INTERP + 1))
    frames = stream.window_frames(chunks, info.sensor_shape, chunk_dt_us=dt, n_windows=n_windows)
    return frames, info.sensor_shape, info.fps, info.scale


def _tee(chunks: Iterator[np.ndarray], keep: list[np.ndarray]) -> Iterator[np.ndarray]:
    for chunk in chunks:
        keep.append(chunk)
        yield chunk


def clip_features(
    source: str | Path,
    poses: np.ndarray | None = None,
    dancers: int = 1,
    max_seconds: float | None = None,
    events_out: str | Path | None = None,
    pose_meta: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Clip (video) or eventify HDF5 file -> ((bins, dancers, 26) features, info).

    With `events_out`, the events are also saved as eventify HDF5.

    Events are in sensor pixels (at most 320 wide). Poses are in pixels of the
    original video. `pose_meta` (from `read_pose_meta`) gives the original
    width and height, so the pose boxes are scaled to sensor pixels before the
    crop. Without `pose_meta`, the size of the clip itself is used.
    """
    keep: list[np.ndarray] | None = [] if events_out is not None else None
    frames, shape, fps, scale = _open_source(source, max_seconds, keep)
    if poses is not None and pose_meta is not None:
        scale = (shape[1] / pose_meta["width"], shape[0] / pose_meta["height"])
        fps = pose_meta.get("fps") or fps
    regions = Regions(shape, poses, dancers, pose_fps=fps, scale=scale)
    tracker = StreamFeatures(regions)
    rows = [tracker.push(frame) for frame in frames]
    if rows:
        feats = add_deltas(np.concatenate(rows))
    else:
        feats = np.zeros((0, regions.dancers, FEATURE_DIM), dtype=np.float32)
    info = {
        "source": str(source),
        "sensor_shape": [int(shape[0]), int(shape[1])],
        "windows": int(len(feats)),
        "window_ms": stream.WINDOW_MS,
        "dancers": int(regions.dancers),
        "regions": regions.mode,
    }
    if poses is not None:
        sx, sy = scale if isinstance(scale, tuple) else (scale, scale)
        info["pose_scale"] = [float(sx), float(sy)]
    if events_out is not None:
        info["events"] = stream.save_hdf5(events_out, keep or [], shape)
        info["events_file"] = str(events_out)
    return feats, info


def clip_features_job(job: tuple[str, int, float | None]) -> tuple[np.ndarray, dict]:
    """One clip for a worker process: (path, dancers, max_seconds) -> (features, info).

    Poses come from a `<clip name>.poses.npy` or `poses.npy` next to the clip, with
    `meta.json` next to the poses for the size of the original video.
    """
    path, dancers, max_seconds = job
    poses_path = find_poses(path)
    poses = np.load(poses_path) if poses_path else None
    pose_meta = None
    if poses_path is not None and poses_path.with_name("meta.json").exists():
        pose_meta = read_pose_meta(poses_path.with_name("meta.json"))
    return clip_features(path, poses, dancers, max_seconds, pose_meta=pose_meta)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Clip or eventify HDF5 -> (bins, dancers, 26) features")
    p.add_argument("source", help="video clip or eventify .h5 events file")
    p.add_argument("out", help="output .npy path")
    p.add_argument("--poses", default=None, help="poses.npy; default: look next to the source")
    p.add_argument("--dancers", type=int, default=1, help="number of vertical strips when there are no poses")
    p.add_argument("--max-seconds", type=float, default=None)
    args = p.parse_args(argv)

    poses_path = Path(args.poses) if args.poses else find_poses(args.source)
    poses = np.load(poses_path) if poses_path else None
    feats, info = clip_features(args.source, poses, args.dancers, args.max_seconds)
    np.save(args.out, feats)
    print(f"{args.out}: shape {feats.shape}, regions from {info['regions']}")


if __name__ == "__main__":
    main()

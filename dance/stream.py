"""Event sources. All events come from eventify-dvs.

This module does not convert video to events. It calls eventify and puts
eventify's output into time windows.

Event format: eventify's EVENT_DTYPE (x int16, y int16, t int64 microseconds,
p int8 with 0 = OFF and 1 = ON). Event files: eventify's HDF5 export.

Window layout: the same as eventify's `frame_stream` with n_bins=1. Each
window is a float32 array of shape (1, 2, H, W). Channel 0 counts OFF events.
Channel 1 counts ON events. Window k covers [k * 33 ms, (k + 1) * 33 ms),
counted from the first frame of the source.

`frame_stream` itself has no `interp` or `warmup_frames` argument. So this
module puts `video_to_event_stream` chunks into the same layout.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from eventify import EVENT_DTYPE, video_to_event_stream, write_hdf5

from .contracts import BIN_MS

WINDOW_MS = BIN_MS
WINDOW_US = WINDOW_MS * 1000
INTERP = 3          # sub-frames between real frames, for 30 fps sources
MAX_WIDTH = 320     # sensor width cap, in pixels
C_THRESH = 0.05     # eventify's default log-intensity threshold
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
HDF5_SUFFIXES = {".h5", ".hdf5"}


@dataclass
class SourceInfo:
    sensor_size: tuple[int, int]   # (W, H), eventify's convention
    fps: float
    frames: int | None             # None for a webcam or an HDF5 file
    scale: float                   # source pixels * scale = sensor pixels

    @property
    def sensor_shape(self) -> tuple[int, int]:
        """(H, W), the convention of eventify's HDF5 export."""
        return (self.sensor_size[1], self.sensor_size[0])

    @property
    def duration_us(self) -> int | None:
        if not self.frames or self.frames < 2:
            return None
        return int((self.frames - 1) * 1_000_000 / self.fps)


def capped_sensor_size(width: int, height: int, max_width: int = MAX_WIDTH) -> tuple[int, int]:
    """Sensor size (W, H) with the width capped and the aspect ratio kept."""
    if width <= max_width:
        return (int(width), int(height))
    return (max_width, max(2, round(height * max_width / width)))


def probe_source(source: str | int, max_width: int = MAX_WIDTH) -> SourceInfo:
    """Read size, frame rate and frame count of a video file or a webcam."""
    if isinstance(source, str) and not Path(source).exists():
        raise FileNotFoundError(f"video not found: {source}")
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise IOError(f"OpenCV cannot open video source: {source!r}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else 0
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise IOError(f"video source reports no frame size: {source!r}")
    if not (fps > 0):
        fps = 30.0
    size = capped_sensor_size(width, height, max_width)
    return SourceInfo(size, float(fps), frames if frames > 0 else None, size[0] / width)


def event_chunks(
    source: str | int,
    info: SourceInfo,
    *,
    c_thresh: float = C_THRESH,
    interp: int = INTERP,
    duration_s: float | None = None,
    warmup_frames: int = 0,
) -> Iterator[np.ndarray]:
    """eventify's `video_to_event_stream` with this project's settings."""
    return video_to_event_stream(
        source,
        c_thresh=c_thresh,
        sensor_size=info.sensor_size,
        interp=interp,
        duration_s=duration_s,
        warmup_frames=warmup_frames,
    )


def read_hdf5(path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Read eventify's HDF5 export. Returns (events, sensor_shape (H, W))."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"events file not found: {path}")
    with h5py.File(path, "r") as f:
        grp = f["events"]
        events = np.zeros(grp["xs"].shape[0], dtype=EVENT_DTYPE)
        events["x"], events["y"] = grp["xs"][:], grp["ys"][:]
        events["t"], events["p"] = grp["ts"][:], grp["ps"][:]
        h, w = (int(v) for v in grp.attrs["sensor_shape"])
    return events, (h, w)


def save_hdf5(
    path: str | Path,
    chunks: list[np.ndarray],
    sensor_shape: tuple[int, int],
    duration_us: int | None = None,
) -> int:
    """Write event chunks with eventify's `write_hdf5`. Returns the event count.

    `duration_us` is the length of the clip. It is saved as one more attribute
    of the `/events` group, `duration_us`. The layout of eventify is unchanged.
    The file needs it because a clip can end without events (a freeze).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = np.concatenate(chunks) if chunks else np.zeros(0, dtype=EVENT_DTYPE)
    write_hdf5(path, events, sensor_shape=sensor_shape)
    if duration_us is not None:
        with h5py.File(path, "a") as f:
            f["events"].attrs["duration_us"] = np.int64(duration_us)
    return int(events.size)


def read_hdf5_duration(path: str | Path) -> int | None:
    """The clip length in microseconds, if the file has it. Plain eventify exports do not."""
    with h5py.File(path, "r") as f:
        value = f["events"].attrs.get("duration_us")
    return None if value is None else int(value)


class WindowBinner:
    """Put event chunks into fixed time windows, in the `frame_stream` layout.

    An empty chunk has no timestamps. For an empty chunk, time moves forward by
    `chunk_dt_us`, or to the value of `clock` (microseconds) when one is given.
    """

    def __init__(
        self,
        sensor_shape: tuple[int, int],
        window_us: int = WINDOW_US,
        chunk_dt_us: int = 8_333,
        clock: Callable[[], int] | None = None,
    ):
        self.h, self.w = sensor_shape
        self.window_us = window_us
        self._size = 2 * self.h * self.w
        self._pending: dict[int, np.ndarray] = {}
        self._next = 0
        self._t_now = 0
        self._dt = chunk_dt_us
        self._clock = clock
        self._anchor: int | None = None

    def _blank(self) -> np.ndarray:
        return np.zeros((1, 2, self.h, self.w), dtype=np.float32)

    def _add(self, chunk: np.ndarray) -> None:
        # Events older than the next window to emit go into that window.
        idx = np.maximum(chunk["t"] // self.window_us, self._next)
        flat = (chunk["p"].astype(np.int64) * self.h + chunk["y"]) * self.w + chunk["x"]
        lo, hi = int(idx.min()), int(idx.max())
        for k in range(lo, hi + 1):
            sel = flat if lo == hi else flat[idx == k]
            if sel.size == 0:
                continue
            frame = self._pending.get(k)
            if frame is None:
                frame = self._pending[k] = self._blank()
            frame += np.bincount(sel, minlength=self._size).reshape(1, 2, self.h, self.w)

    def _emit(self, upto: int) -> list[np.ndarray]:
        out = []
        while self._next < upto:
            frame = self._pending.pop(self._next, None)
            out.append(frame if frame is not None else self._blank())
            self._next += 1
        return out

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Add one chunk. Return the windows that are now complete, in order."""
        if chunk.size:
            self._add(chunk)
            self._t_now = max(self._t_now, int(chunk["t"].max()))
            if self._clock is not None:
                self._anchor = self._clock() - self._t_now
        elif self._clock is not None and self._anchor is not None:
            self._t_now = max(self._t_now, self._clock() - self._anchor)
        else:
            self._t_now += self._dt
        return self._emit(self._t_now // self.window_us)

    def flush(self, n_windows: int | None = None) -> list[np.ndarray]:
        """Emit the windows that are left, up to `n_windows` in total."""
        upto = max(self._pending, default=self._next - 1) + 1
        if n_windows is not None:
            upto = max(upto, n_windows)
        return self._emit(upto)


def window_frames(
    chunks: Iterable[np.ndarray],
    sensor_shape: tuple[int, int],
    *,
    window_us: int = WINDOW_US,
    chunk_dt_us: int = 8_333,
    n_windows: int | None = None,
    clock: Callable[[], int] | None = None,
) -> Iterator[np.ndarray]:
    """Event chunks -> one (1, 2, H, W) array per time window."""
    binner = WindowBinner(sensor_shape, window_us, chunk_dt_us, clock)
    for chunk in chunks:
        yield from binner.push(chunk)
    yield from binner.flush(n_windows)


def split_by_window(events: np.ndarray, window_us: int = WINDOW_US) -> Iterator[np.ndarray]:
    """Split one large event array (an HDF5 file) into chunks, one per window."""
    if events.size == 0:
        return
    idx = events["t"] // window_us
    order = np.argsort(idx, kind="stable")
    events, idx = events[order], idx[order]
    n = int(idx[-1]) + 1
    bounds = np.searchsorted(idx, np.arange(n + 1))
    for k in range(n):
        yield events[bounds[k]:bounds[k + 1]]

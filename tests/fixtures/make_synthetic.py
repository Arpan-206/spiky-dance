"""Synthetic "dancer" clips with ground truth. No real video is used.

One dancer is a crude figure: a head, a body and two arms with hand blobs.
The arms swing with a fixed period, like a repeated dance move. The body
sways a little. Each dancer has its own vertical strip of STRIP_W pixels, so
`dance.features.strip_boxes` cuts the frame at the same places.

The frame is STRIP_W * dancers wide. So a dancer has the same box size in a
clip with 1 dancer and in a clip with 3 dancers. Features are normalised to
the box, so they stay comparable.

Kinds of motion, one per dancer:
    "normal"  smooth periodic motion.
    "jump"    the same motion, but in several segments the figure moves
              suddenly to a different place every few frames, and the arms
              move suddenly to a different point of the swing.
    "freeze"  the same motion, but all motion stops in several long segments.
    "still"   no motion in the whole clip (for noise measurements).

Ground truth: a bool array (frames, dancers). True = the frame is abnormal.
For "jump": the frames of a jump segment, and the frame of the jump back.
For "freeze" and "still": the frames that are equal to the frame before.

All choices come from `seed`. The same arguments give the same frames.

Command line (for manual tries; writes clips and `<name>.truth.json` files):
    uv run python -m tests.fixtures.make_synthetic OUT_DIR [--seed N] [--seconds S]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

STRIP_W, HEIGHT, FPS = 96, 120, 30.0
BACKGROUND = 35            # dark grey, not black: keeps compression noise below the event threshold
KINDS = ("normal", "jump", "freeze", "still")
MAX_DANCERS = 3

PERIOD_FRAMES = 36.0       # one arm swing: 1.2 s
PERIOD_SPREAD = 0.04       # +-4 % between dancers
AMP_SPREAD = 0.10          # +-10 % between dancers
ARM_SWING_RAD = 1.0        # arm angle goes this far to each side of horizontal
ARM_LEN = 18.0
SWAY_PX = 3.0

WARMUP_FRAMES = 30         # no abnormal segment in the first second
JUMP_SEGMENTS = 3
JUMP_FRACTION = 0.36       # part of the clip inside jump segments
JUMP_EVERY = (3, 6)        # frames between two sudden moves, low..high inclusive
JUMP_DX, JUMP_DY = 12.0, 10.0
JUMP_MIN_DIST = 10.0       # a sudden move is at least this far, in pixels
FREEZE_SEGMENTS = 2
FREEZE_FRACTION = 0.44     # part of the clip inside freeze segments

_SHIFT = 4                 # OpenCV sub-pixel bits: coordinates in 1/16 pixel


@dataclass(frozen=True)
class SyntheticClip:
    """A written clip and its ground truth."""

    path: Path
    kinds: tuple[str, ...]     # one kind per dancer, left to right
    seed: int
    n_frames: int
    fps: float
    width: int
    height: int
    abnormal: np.ndarray       # bool, (n_frames, dancers)

    @property
    def dancers(self) -> int:
        return len(self.kinds)

    def abnormal_bins(self, n_bins: int) -> np.ndarray:
        """Ground truth per feature time bin, bool (n_bins, dancers)."""
        return abnormal_bins(self.abnormal, n_bins, self.fps)


def _segments(n_frames: int, count: int, fraction: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    """`count` frame ranges [start, end) that cover about `fraction` of the clip.

    The clip is cut into `count` equal slots. Each slot holds one segment at a
    random place. No segment starts in the first WARMUP_FRAMES frames.
    """
    seg_len = max(2, int(round(n_frames * fraction / count)))
    slot = n_frames / count
    out = []
    for i in range(count):
        lo = max(int(np.ceil(i * slot)) + 4, WARMUP_FRAMES)
        hi = int(np.floor((i + 1) * slot)) - seg_len - 4
        start = lo if hi <= lo else int(rng.integers(lo, hi + 1))
        out.append((start, min(start + seg_len, n_frames - 1)))
    return out


def _dancer_motion(
    kind: str, n_frames: int, seed: int, dancer: int, in_step: bool = False
) -> dict[str, np.ndarray]:
    """Per-frame motion of one dancer, and its ground truth.

    Returns arrays of length n_frames: `u` (motion time, in frames), `dx`,
    `dy` (sudden offset of the whole figure), `arm_phase` (sudden phase offset
    of the arms), `abnormal` (bool). Also scalars `period`, `phase`, `amp`.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    # The normal parameters depend only on (seed, dancer), not on the kind.
    # With `in_step`, all dancers share the period, phase and size of dancer 0.
    base = np.random.default_rng([seed, 0 if in_step else dancer, 0])
    period = PERIOD_FRAMES * (1.0 + base.uniform(-PERIOD_SPREAD, PERIOD_SPREAD))
    phase = base.uniform(0.0, 2.0 * np.pi)
    amp = 1.0 + base.uniform(-AMP_SPREAD, AMP_SPREAD)

    rng = np.random.default_rng([seed, dancer, 1])
    moving = np.ones(n_frames, dtype=bool)      # False = frame equals the frame before
    dx = np.zeros(n_frames)
    dy = np.zeros(n_frames)
    arm_phase = np.zeros(n_frames)
    abnormal = np.zeros(n_frames, dtype=bool)

    if kind == "still":
        moving[1:] = False
        abnormal[1:] = True
    elif kind == "freeze":
        for start, end in _segments(n_frames, FREEZE_SEGMENTS, FREEZE_FRACTION, rng):
            moving[start:end] = False
            abnormal[start:end] = True
    elif kind == "jump":
        for start, end in _segments(n_frames, JUMP_SEGMENTS, JUMP_FRACTION, rng):
            k, here = start, np.zeros(2)
            while k < end:
                while True:
                    there = rng.uniform([-JUMP_DX, -JUMP_DY], [JUMP_DX, JUMP_DY])
                    if np.hypot(*(there - here)) >= JUMP_MIN_DIST:
                        break
                nxt = min(end, k + int(rng.integers(JUMP_EVERY[0], JUMP_EVERY[1] + 1)))
                dx[k:nxt], dy[k:nxt] = there
                arm_phase[k:nxt] = rng.uniform(0.5 * np.pi, 1.5 * np.pi)
                k, here = nxt, there
            abnormal[start:end + 1] = True      # frame `end` holds the jump back

    u = np.cumsum(moving) - 1.0
    return {
        "u": u, "dx": dx, "dy": dy, "arm_phase": arm_phase, "abnormal": abnormal,
        "period": np.float64(period), "phase": np.float64(phase), "amp": np.float64(amp),
    }


def _pt(x: float, y: float) -> tuple[int, int]:
    scale = 1 << _SHIFT
    return (int(round(x * scale)), int(round(y * scale)))


def _draw_dancer(frame: np.ndarray, x_off: float, m: dict[str, np.ndarray], k: int) -> None:
    """Draw one dancer of frame `k` into `frame` (grey, uint8). `x_off` = left edge of the strip."""
    a = 2.0 * np.pi * m["u"][k] / m["period"] + m["phase"]
    amp = float(m["amp"])
    cx = x_off + STRIP_W / 2 + m["dx"][k] + SWAY_PX * amp * np.cos(a)
    cy = 66.0 + m["dy"][k] + 1.5 * amp * np.sin(2.0 * a)
    aa, sh = cv2.LINE_AA, _SHIFT
    r = 1 << sh

    cv2.ellipse(frame, _pt(cx, cy + 6), (9 * r, 22 * r), 0, 0, 360, 170, -1, aa, sh)   # body
    cv2.circle(frame, _pt(cx, cy - 26), 7 * r, 200, -1, aa, sh)                         # head
    arm_a = a + m["arm_phase"][k]
    for side, lag in ((-1.0, 0.0), (1.0, 0.5 * np.pi)):
        # theta: 0 = arm points down, pi/2 = horizontal, pi = up.
        theta = 0.5 * np.pi + ARM_SWING_RAD * amp * np.sin(arm_a + lag)
        sx, sy = cx + side * 8.0, cy - 14.0
        hx = sx + side * ARM_LEN * np.sin(theta)
        hy = sy + ARM_LEN * np.cos(theta)
        cv2.line(frame, _pt(sx, sy), _pt(hx, hy), 150, 3, aa, sh)                       # arm
        cv2.circle(frame, _pt(hx, hy), 4 * r, 235, -1, aa, sh)                          # hand


def render_clip(
    kinds: Sequence[str] = ("normal",), seed: int = 0, seconds: float = 8.0, in_step: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Render a clip in memory.

    kinds: one kind per dancer, left to right. 1 to 3 dancers.
    in_step: all dancers do the same move at the same time, like a group dance.
    Returns (frames, abnormal):
        frames    uint8, (n_frames, HEIGHT, STRIP_W * dancers), grey
        abnormal  bool, (n_frames, dancers), the ground truth
    """
    kinds = tuple(kinds)
    if not 1 <= len(kinds) <= MAX_DANCERS:
        raise ValueError(f"1 to {MAX_DANCERS} dancers, got {len(kinds)}")
    n_frames = int(round(seconds * FPS))
    if n_frames < 2 * WARMUP_FRAMES:
        raise ValueError(f"the clip is too short: {seconds} s")
    motions = [_dancer_motion(kind, n_frames, seed, d, in_step) for d, kind in enumerate(kinds)]
    frames = np.full((n_frames, HEIGHT, STRIP_W * len(kinds)), BACKGROUND, dtype=np.uint8)
    for k in range(n_frames):
        for d, m in enumerate(motions):
            _draw_dancer(frames[k], d * STRIP_W, m, k)
    return frames, np.stack([m["abnormal"] for m in motions], axis=1)


def write_clip(path: str | Path, frames: np.ndarray) -> Path:
    """Write grey frames (n, H, W) as an mp4v video at FPS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        raise IOError(f"OpenCV cannot write an mp4v clip to {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    return path


def read_clip(path: str | Path) -> np.ndarray:
    """Decode a clip with OpenCV. Returns grey frames, uint8, (n, H, W)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open {path}")
    out = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    return np.stack(out) if out else np.zeros((0, 0, 0), dtype=np.uint8)


def make_clip(
    path: str | Path, kinds: Sequence[str] = ("normal",), seed: int = 0, seconds: float = 8.0,
    in_step: bool = False,
) -> SyntheticClip:
    """Render a clip, write it to `path`, and return it with its ground truth."""
    kinds = tuple(kinds)
    frames, abnormal = render_clip(kinds, seed, seconds, in_step)
    write_clip(path, frames)
    return SyntheticClip(
        path=Path(path), kinds=kinds, seed=seed, n_frames=len(frames), fps=FPS,
        width=frames.shape[2], height=frames.shape[1], abnormal=abnormal,
    )


def abnormal_bins(abnormal: np.ndarray, n_bins: int, fps: float = FPS, bin_ms: int = 33) -> np.ndarray:
    """Ground truth per frame -> ground truth per feature time bin.

    Bin k covers [k * bin_ms, (k + 1) * bin_ms). The events of the change from
    frame j - 1 to frame j have times in ((j - 1) / fps, j / fps]. A bin is
    abnormal if it overlaps the time span of one abnormal frame or more.
    Returns bool, (n_bins, dancers).
    """
    abnormal = np.asarray(abnormal, dtype=bool)
    n_frames = len(abnormal)
    # cum[j] = number of abnormal frames with index < j.
    cum = np.concatenate([np.zeros((1, *abnormal.shape[1:]), int), np.cumsum(abnormal, axis=0)])
    start_s = np.arange(n_bins) * bin_ms / 1000.0
    end_s = start_s + bin_ms / 1000.0
    lo = np.clip(np.floor(start_s * fps + 1e-9).astype(int) + 1, 0, n_frames - 1)
    hi = np.clip(np.ceil(end_s * fps - 1e-9).astype(int), lo, n_frames - 1)
    return (cum[hi + 1] - cum[lo]) > 0


def frame_ranges(mask: np.ndarray) -> list[list[int]]:
    """A bool array per frame -> [[start, end), ...] ranges of True."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [[int(s), int(e)] for s, e in zip(edges[::2], edges[1::2])]


def write_truth(clip: SyntheticClip, path: str | Path | None = None) -> Path:
    """Write the ground truth of a clip as JSON. Default path: `<clip stem>.truth.json`."""
    path = Path(path) if path is not None else clip.path.with_suffix(".truth.json")
    data = {
        "clip": clip.path.name,
        "kinds": list(clip.kinds),
        "seed": clip.seed,
        "frames": clip.n_frames,
        "fps": clip.fps,
        "size": [clip.width, clip.height],
        "abnormal_frames": [frame_ranges(clip.abnormal[:, d]) for d in range(clip.dancers)],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def make_set(out_dir: str | Path, seed: int = 0, seconds: float = 8.0) -> list[SyntheticClip]:
    """Write a small set of clips for manual tries. Returns the clips."""
    out_dir = Path(out_dir)
    plan: list[tuple[str, tuple[str, ...], int]] = [
        (f"normal_{i}", ("normal",), seed + i) for i in range(4)
    ]
    plan += [
        ("heldout_normal", ("normal",), seed + 100),
        ("jump", ("jump",), seed + 101),
        ("freeze", ("freeze",), seed + 102),
        ("trio_normal", ("normal", "normal", "normal"), seed + 103),
        ("trio_one_jump", ("normal", "jump", "normal"), seed + 104),
        ("trio_one_freeze", ("normal", "normal", "freeze"), seed + 105),
    ]
    clips = []
    for name, kinds, clip_seed in plan:
        clip = make_clip(out_dir / f"{name}.mp4", kinds, clip_seed, seconds)
        write_truth(clip)
        clips.append(clip)
    return clips


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write a small set of synthetic dancer clips")
    p.add_argument("out_dir", help="folder for the clips; do not use a folder that git tracks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=8.0)
    args = p.parse_args(argv)
    for clip in make_set(args.out_dir, args.seed, args.seconds):
        part = clip.abnormal.mean(axis=0)
        print(f"{clip.path}: {clip.width}x{clip.height}, {clip.n_frames} frames, "
              f"kinds {'/'.join(clip.kinds)}, abnormal part per dancer {np.round(part, 2).tolist()}")


if __name__ == "__main__":
    main()

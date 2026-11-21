"""Synthetic clips for the tests. No real video files are used."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

WIDTH, HEIGHT, FPS = 160, 120, 30.0
SQUARE = 24


def square_positions(
    n_frames: int,
    period: float = 30.0,
    phase: float = 0.0,
    jumps: tuple[int, ...] = (),
    freeze: tuple[int, int] | None = None,
) -> np.ndarray:
    """Top-left corner of the square per frame, shape (n_frames, 2).

    Normal motion is a smooth loop (a figure of eight). `jumps` lists frames at
    which the square moves suddenly to the opposite side for 3 frames.
    `freeze` gives a frame range in which the square does not move.
    """
    k = np.arange(n_frames, dtype=np.float64)
    if freeze is not None:
        k = np.where((k >= freeze[0]) & (k < freeze[1]), freeze[0], k)
    a = 2 * np.pi * k / period + phase
    x = (WIDTH - SQUARE) / 2 + 45 * np.sin(a)
    y = (HEIGHT - SQUARE) / 2 + 30 * np.sin(2 * a)
    for j in jumps:
        x[j:j + 3] = (WIDTH - SQUARE) - x[j:j + 3]
        y[j:j + 3] = (HEIGHT - SQUARE) - y[j:j + 3]
    return np.stack([x, y], axis=1)


def write_square_clip(path: Path, positions: np.ndarray | None, n_frames: int = 60) -> Path:
    """Write a clip of a white square on black. `positions=None` gives a static clip."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "OpenCV cannot write mp4v test clips"
    for k in range(n_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        x, y = (60, 40) if positions is None else np.rint(positions[k]).astype(int)
        frame[y:y + SQUARE, x:x + SQUARE] = 255
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def moving_clip(tmp_path: Path) -> Path:
    return write_square_clip(tmp_path / "moving.mp4", square_positions(60), 60)


@pytest.fixture
def static_clip(tmp_path: Path) -> Path:
    return write_square_clip(tmp_path / "static.mp4", None, 60)

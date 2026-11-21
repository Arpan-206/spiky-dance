"""The fallback preview video and the live preview tap."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from dance import baseline, preview, run, score, stream
from dance.features import Regions, clip_features
from tests.conftest import HEIGHT, WIDTH


@pytest.fixture
def episode(moving_clip, tmp_path):
    """A scored episode: baseline fitted on the clip itself, 2 strips."""
    feats, _ = clip_features(moving_clip, dancers=2)
    models = tmp_path / "models"
    baseline.train([feats], models)
    summary = run.run_clip(moving_clip, tmp_path / "episodes", models, dancers=2, scorer="baseline")
    return tmp_path / "episodes" / "moving", summary


def test_renderer_colours_boxes_and_size():
    r = preview.PreviewRenderer((HEIGHT, WIDTH), dancers=2, with_sync=True)
    window = np.zeros((1, 2, HEIGHT, WIDTH), dtype=np.float32)
    window[0, 1, 20:30, 20:30] = 5          # ON events
    window[0, 0, 60:70, 100:110] = 5        # OFF events
    boxes = np.array([[0, 0, 80, 120], [80, 0, 160, 120]])
    picture = r.frame(window, boxes, np.array([0.2, 0.95]), np.array([0.1, 1.5]))
    assert picture.dtype == np.uint8
    assert picture.shape == (r.height, r.width, 3)
    assert r.width == WIDTH * 2 and r.height == HEIGHT * 2 + preview.SCORE_PANEL_H + preview.SYNC_PANEL_H
    b, g, red = (int(v) for v in picture[50, 50])           # inside the ON patch, at 2x scale
    assert red > 200 and g > 150 and b < 100                # warm yellow
    b, g, red = (int(v) for v in picture[130, 210])         # inside the OFF patch
    assert b > 200 and red < 100                            # blue
    assert picture[200, 120].max() < 40                     # background: near black
    assert tuple(picture[100, 0]) == preview.DANCER_BGR[0]  # left edge of the box of dancer 0


def test_events_fade_quickly():
    r = preview.PreviewRenderer((HEIGHT, WIDTH), dancers=1)
    window = np.zeros((1, 2, HEIGHT, WIDTH), dtype=np.float32)
    window[0, 1, 20:30, 20:30] = 3
    boxes = np.array([[0, 0, WIDTH, HEIGHT]])
    bright = r.frame(window, boxes)[50, 50].astype(int).sum()
    empty = np.zeros_like(window)
    for _ in range(8):
        last = r.frame(empty, boxes)
    assert last[50, 50].astype(int).sum() < 0.2 * bright


def test_write_preview_makes_a_video_of_the_episode(episode):
    folder, summary = episode
    out = preview.write_preview(folder)
    assert out == folder / "preview.mp4" and out.stat().st_size > 1000
    cap = cv2.VideoCapture(str(out))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read()
    cap.release()
    assert ok and first.shape == (height, width, 3)
    assert abs(frames - summary["bins"]) <= 1
    assert width == WIDTH * 2
    assert height == HEIGHT * 2 + preview.SCORE_PANEL_H + preview.SYNC_PANEL_H      # 2 dancers: sync strip
    assert json.loads((folder / "scores_meta.json").read_text())["dancers"] == 2


def test_run_with_preview_flag(moving_clip, tmp_path, episode, capsys):
    folder, _ = episode
    run.main([str(folder), "--models", str(tmp_path / "models"), "--scorer", "baseline", "--dancers", "2", "--preview"])
    assert (folder / "preview.mp4").exists()
    assert "preview:" in capsys.readouterr().out


def test_preview_needs_a_scored_episode(tmp_path):
    empty = tmp_path / "episodes" / "x"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="dance.run"):
        preview.write_preview(empty)


def test_live_tap_shows_each_window_and_stops_on_request(moving_clip):
    info = stream.probe_source(str(moving_clip))
    regions = Regions(info.sensor_shape, None, 2)
    renderer = preview.PreviewRenderer(info.sensor_shape, 2, with_sync=True, scroll=True)
    shown = []

    def show(picture):
        shown.append(picture.shape)
        return len(shown) < 20                     # like the q key at picture 20

    class Model:
        def score(self, features):
            return np.full(features.shape[:2], 0.5, dtype=np.float32)

    updates = []

    def emit(update):
        renderer.set_scores(update["latest"], update["sync"])
        updates.append(update)

    windows = preview.live_tap(score.source_windows(str(moving_clip), info), regions, renderer, show)
    session = score.live_loop(windows, Model(), regions, emit)
    assert len(shown) == 20 and set(shown) == {(renderer.height, renderer.width, 3)}
    assert len(updates) == 6 and session.shape == (18, 2)      # 19 windows passed: 6 updates of 3

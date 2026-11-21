"""Steps 1 and 2: eventify output feeds through the feature code."""

from __future__ import annotations

import numpy as np
from eventify import EVENT_DTYPE, frame_stream

from dance import stream
from dance.features import (
    NUM_FEATURES,
    add_deltas,
    base_features,
    Regions,
    clip_features,
    extract_features,
    pose_boxes,
    read_pose_meta,
    strip_boxes,
)
from tests.conftest import FPS, HEIGHT, WIDTH


def _check(feats: np.ndarray, dancers: int) -> None:
    assert feats.ndim == 3 and feats.shape[1:] == (dancers, NUM_FEATURES)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()
    assert feats[..., 0].min() >= 0 and feats[..., 0].max() <= 1
    assert np.abs(feats[..., 1]).max() <= 1
    sums = feats[..., 4:13].sum(axis=-1)
    assert np.all(np.isclose(sums, 1, atol=1e-4) | np.isclose(sums, 0, atol=1e-6))


def test_eventify_stream_feeds_features(moving_clip):
    """Acceptance test 1: video_to_event_stream output -> features, no error."""
    feats, info = clip_features(moving_clip, dancers=2)
    _check(feats, dancers=2)
    # 60 frames at 30 fps = 1.97 s = 60 windows of 33 ms.
    assert 58 <= len(feats) <= 61
    assert info["regions"] == "strips"
    assert (feats[..., 0] > 0).any(), "a moving square must give events"


def test_eventify_frame_stream_feeds_features(moving_clip):
    """Acceptance test 1: frame_stream's (n_bins, 2, H, W) tensor -> features."""
    boxes = strip_boxes(WIDTH, HEIGHT, 1)
    n = 0
    for frames in frame_stream(str(moving_clip), (WIDTH, HEIGHT), n_bins=1, window_ms=stream.WINDOW_MS):
        assert frames.shape == (1, 2, HEIGHT, WIDTH)
        _check(extract_features(frames, boxes), dancers=1)
        n += 1
    assert n > 10


def test_static_clip_gives_empty_bins_without_nan(static_clip):
    feats, _ = clip_features(static_clip, dancers=3)
    _check(feats, dancers=3)
    assert len(feats) > 0
    assert np.all(feats[..., 0] == 0)
    assert np.all(feats[..., 2:4] == 0.5)
    assert np.all(feats[..., 4:] == 0)


def test_empty_tensor_gives_right_shape_and_no_nan():
    frames = np.zeros((7, 2, HEIGHT, WIDTH), dtype=np.float32)
    feats = extract_features(frames, strip_boxes(WIDTH, HEIGHT, 3))
    assert feats.shape == (7, 3, NUM_FEATURES)
    _check(feats, dancers=3)


def test_feature_values_on_a_known_tensor():
    frames = np.zeros((1, 2, 90, 90), dtype=np.float32)
    frames[0, 1, 10, 80] = 3   # 3 ON events, top-right cell
    frames[0, 0, 80, 10] = 1   # 1 OFF event, bottom-left cell
    f = extract_features(frames, np.array([[0, 0, 90, 90]]))[0, 0]
    assert np.isclose(f[1], 0.5)                       # (3 - 1) / 4
    assert np.isclose(f[2], (3 * 80.5 + 10.5) / 4 / 90)
    assert np.isclose(f[3], (3 * 10.5 + 80.5) / 4 / 90)
    grid = f[4:13].reshape(3, 3)
    assert np.isclose(grid[0, 2], 0.75) and np.isclose(grid[2, 0], 0.25)


def test_window_binner_keeps_every_event():
    rng = np.random.default_rng(0)
    events = np.zeros(5000, dtype=EVENT_DTYPE)
    events["x"] = rng.integers(0, 32, 5000)
    events["y"] = rng.integers(0, 24, 5000)
    events["t"] = np.sort(rng.integers(0, 400_000, 5000))
    events["p"] = rng.integers(0, 2, 5000)
    chunks = np.array_split(events, 40)
    frames = list(stream.window_frames(chunks, (24, 32)))
    assert all(f.shape == (1, 2, 24, 32) for f in frames)
    assert sum(float(f.sum()) for f in frames) == 5000
    assert len(frames) == int(events["t"].max()) // stream.WINDOW_US + 1
    k = 5
    in_k = events[events["t"] // stream.WINDOW_US == k]
    assert frames[k].sum() == len(in_k)
    assert frames[k][0, 1].sum() == (in_k["p"] == 1).sum()


def test_hdf5_export_gives_the_same_features(moving_clip, tmp_path):
    h5 = tmp_path / "events.h5"
    direct, info = clip_features(moving_clip, dancers=2, events_out=h5)
    assert h5.exists() and info["events"] > 0
    events, shape = stream.read_hdf5(h5)
    assert events.dtype == EVENT_DTYPE and shape == (HEIGHT, WIDTH)
    from_file, _ = clip_features(h5, dancers=2)
    n = min(len(direct), len(from_file))
    assert abs(len(direct) - len(from_file)) <= 1
    np.testing.assert_allclose(direct[:n], from_file[:n], atol=1e-6)


def test_pose_boxes_follow_the_keypoints():
    poses = np.zeros((10, 2, 17, 3), dtype=np.float32)
    # Dancer 0: keypoints in x 20..60, y 30..90 (source pixels, scale 0.5).
    poses[:, 0, :, 0] = np.linspace(20, 60, 17)
    poses[:, 0, :, 1] = np.linspace(30, 90, 17)
    poses[:, 0, :, 2] = 0.9
    poses[4, 0, :, 2] = 0.0          # one frame without keypoints
    # Dancer 1: never seen -> vertical strip.
    boxes = pose_boxes(poses, width=160, height=120, scale=0.5)
    assert boxes.shape == (10, 2, 4)
    # 20..60 * 0.5 = 10..30, padded 15% of 20 = 3 -> 7..33.
    assert tuple(boxes[0, 0]) == (7, 10, 33, 50)
    assert tuple(boxes[4, 0]) == tuple(boxes[3, 0])
    assert tuple(boxes[0, 1]) == (80, 0, 160, 120)


def test_features_with_poses(moving_clip):
    poses = np.zeros((60, 2, 17, 3), dtype=np.float32)
    poses[..., 0] = np.linspace(0.2, 0.8, 17)      # 0..1 fractions
    poses[..., 1] = np.linspace(0.1, 0.9, 17)
    poses[..., 2] = 1.0
    feats, info = clip_features(moving_clip, poses=poses)
    _check(feats, dancers=2)
    assert info["regions"] == "poses"
    regions = Regions((HEIGHT, WIDTH), poses, pose_fps=FPS)
    assert regions.boxes(0).shape == (2, 4)
    assert regions.boxes(10_000).shape == (2, 4)   # past the last pose frame


def test_deltas_are_bin_to_bin_differences():
    assert NUM_FEATURES == 26
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 3, (6, 2, 30, 40)).astype(np.float32)
    frames[3] = 0                                   # one empty bin
    boxes = strip_boxes(40, 30, 2)
    base = base_features(frames, boxes)
    feats = extract_features(frames, boxes)
    assert base.shape == (6, 2, 13) and feats.shape == (6, 2, 26)
    np.testing.assert_array_equal(feats[..., :13], base)
    assert np.all(feats[0, :, 13:] == 0)            # first bin: no delta
    np.testing.assert_allclose(feats[1:, :, 13:], base[1:] - base[:-1], atol=1e-6)
    np.testing.assert_array_equal(feats, add_deltas(base))
    assert np.isfinite(feats).all()
    assert np.abs(feats[..., 14]).max() <= 2 and np.abs(np.delete(feats[..., 13:], 1, axis=-1)).max() <= 1


def test_pose_boxes_scale_original_pixels_to_sensor_pixels():
    """Poses are in pixels of the original video. Events are at most 320 px wide."""
    poses = np.zeros((3, 1, 17, 3), dtype=np.float32)
    poses[..., 0] = np.linspace(400, 800, 17)          # original 1280x720
    poses[..., 1] = np.linspace(200, 600, 17)
    poses[..., 2] = 0.9
    poses[:, :, 8, :] = (5000.0, 5000.0, 0.0)          # a missing joint: confidence 0, not used
    # Sensor 320x180: scale 0.25 -> x 100..200, y 50..150; pad 15% of 100 = 15.
    boxes = pose_boxes(poses, width=320, height=180, scale=(320 / 1280, 180 / 720))
    assert tuple(boxes[0, 0]) == (85, 35, 215, 165)
    # Different scale in x and y (a 4:3 original on a 320x180 sensor).
    boxes = pose_boxes(poses, width=320, height=180, scale=(0.25, 0.125))
    assert tuple(boxes[0, 0]) == (85, 17, 215, 83)


def test_clip_features_scale_poses_with_the_meta_file(tmp_path):
    from tests.conftest import square_positions, write_square_clip

    clip = write_square_clip(tmp_path / "big.mp4", square_positions(60), 60, upscale=4)   # 640x480
    meta_file = tmp_path / "meta.json"
    meta_file.write_text('{"frame_rate": 30, "width": 640, "height": 480, "frame_count": 60}')
    pose_meta = read_pose_meta(meta_file)
    assert pose_meta == {"width": 640, "height": 480, "fps": 30.0, "frames": 60}

    poses = np.zeros((60, 2, 17, 3), dtype=np.float32)
    poses[:, 0, :, 0] = np.linspace(92, 548, 17)       # around the moving square, original pixels
    poses[:, 0, :, 1] = np.linspace(72, 408, 17)
    poses[:, 1, :, 0] = np.linspace(10, 60, 17)        # a corner with no motion
    poses[:, 1, :, 1] = np.linspace(430, 470, 17)
    poses[..., 2] = 0.9
    feats, info = clip_features(clip, poses=poses, pose_meta=pose_meta)
    assert info["sensor_shape"] == [240, 320] and info["pose_scale"] == [0.5, 0.5]
    _check(feats, dancers=2)
    assert (feats[:, 0, 0] > 0).mean() > 0.9           # the square is inside box 0
    assert np.all(feats[:, 1, 0] == 0)                 # box 1 is empty
    whole, _ = clip_features(clip, dancers=1)
    assert (whole[:, 0, 0] > 0).sum() == (feats[:, 0, 0] > 0).sum()

"""Small real-model checks, with no more than five seconds per video run."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vlma.pose import run_pose


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample_episode(tmp_path_factory):
    if not (ROOT / "clips/sample_walk.mp4").exists():
        pytest.skip("sample clip missing, see README")
    source = ROOT / "clips" / "sample_walk.mp4"
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    output = tmp_path_factory.mktemp("pose_sample")
    summary = run_pose(source, output, max_frames=max(1, min(20, int(fps * 5))))
    return output, summary


def test_episode_contract(sample_episode):
    output, summary = sample_episode
    meta = json.loads((output / "meta.json").read_text())
    n, d = meta["n_frames"], meta["dancers"]
    poses = np.load(output / "poses.npy")
    assert poses.shape == (n, d, 17, 3)
    assert poses.dtype == np.float32
    assert np.isfinite(poses).all()
    assert np.all((poses[..., 2] >= 0) & (poses[..., 2] <= 1))
    with np.load(output / "pose3d.npz") as data:
        assert data["world"].shape == data["image"].shape == (n, d, 33, 3)
        assert data["vis"].shape == (n, d, 33)
        assert all(data[key].dtype == np.float32 for key in ("world", "image", "vis"))
        assert float(data["fps"]) == meta["fps"]
        np.testing.assert_allclose(poses[..., :2], data["image"][:, :, [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28], :2] * [meta["width"], meta["height"]], rtol=1e-6)
    for name in ("clip.mp4", "skeleton.mp4"):
        cap = cv2.VideoCapture(str(output / name))
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(meta["fps"])
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape[:2] == (meta["height"], meta["width"])
            decoded += 1
        cap.release()
        assert decoded == n
    assert len(summary["detected_fraction"]) == d
    assert len(summary["mean_confidence"]) == d


def test_left_to_right_anchor(sample_episode):
    output, summary = sample_episode
    if summary["dancers"] < 2:
        pytest.skip("Sample does not consistently contain two detected people")
    poses = np.load(output / "poses.npy")
    present = (poses[:, :, [11, 12], 2] > 0).all(axis=(1, 2))
    indices = np.flatnonzero(present)
    assert len(indices)
    hips = poses[indices[0], :, [11, 12], 0].mean(axis=0)
    assert hips[0] <= hips[1]


@pytest.mark.parametrize("dancers", [None, 2])
def test_black_video(tmp_path, dancers):
    source = tmp_path / "black.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (320, 240))
    assert writer.isOpened(), "Black-video VideoWriter failed to open"
    for _ in range(3):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    output = tmp_path / "episode"
    summary = run_pose(source, output, dancers=dancers)
    count = dancers or 1
    poses = np.load(output / "poses.npy")
    assert poses.shape == (3, count, 17, 3)
    assert not poses.any()
    assert summary["detected_fraction"] == [0.0] * count
    assert summary["mean_confidence"] == [0.0] * count
    with np.load(output / "pose3d.npz") as data:
        assert np.isnan(data["world"]).all()
        assert not data["vis"].any()
        assert not data["image"].any()
    assert (output / "clip.mp4").read_bytes() == source.read_bytes()

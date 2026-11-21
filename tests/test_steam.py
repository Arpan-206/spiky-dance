"""Small deterministic optical and video contract checks."""
import cv2
import numpy as np
import pytest

from vlma.steam import Steam, make, keypoint_drift


def frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    result = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result.append(frame)
    cap.release()
    return fps, np.asarray(result)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12, (96, 64))
    rng = np.random.default_rng(2)
    for _ in range(6):
        writer.write(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    writer.release()
    return path


def test_zero_and_video_contract(source, tmp_path):
    fps, original = frames(source)
    for level in (0, 40, 80):
        info = make(source, tmp_path / str(level), level)
        out_fps, output = frames(info['clip'])
        assert output.shape == original.shape
        assert out_fps == pytest.approx(fps)
        assert info['frames'] == len(original)
        if level == 0:
            np.testing.assert_array_equal(original, output)


def test_contrast_and_reproducibility():
    frame = np.random.default_rng(9).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    outputs = [Steam(96, 64, level, seed=3).apply(frame, 1.5)[0] for level in (0, 40, 80)]
    assert outputs[0].std() > outputs[1].std() > outputs[2].std()
    np.testing.assert_array_equal(outputs[1], Steam(96, 64, 40, seed=3).apply(frame, 1.5)[0])
    np.testing.assert_array_equal(outputs[0], frame)


def test_frame_limit(source, tmp_path):
    info = make(source, tmp_path / 'limited', 0, max_frames=3)
    assert len(frames(info['clip'])[1]) == 3


def test_drift_uses_only_valid_points(tmp_path):
    clean = np.array([[[[10, 20, 0.8], [np.nan, np.nan, 0]]]])
    veiled = np.array([[[[20, 20, 0.6], [40, 40, 0.5]]]])
    np.save(tmp_path / 'clean.npy', clean)
    np.save(tmp_path / 'veiled.npy', veiled)
    assert keypoint_drift(tmp_path / 'clean.npy', tmp_path / 'veiled.npy', 100, 50) == pytest.approx([10])


def test_optical_sweep_and_score_hook(source, tmp_path, monkeypatch):
    import types
    import vlma.steam as steam
    monkeypatch.setattr(steam, '__file__', str(tmp_path / 'vlma' / 'steam.py'))
    real_import = steam.importlib.import_module
    def unavailable(name):
        if name == 'vlma.pose':
            raise ImportError('pose intentionally absent in test')
        return real_import(name)
    monkeypatch.setattr(steam, 'importlib', types.SimpleNamespace(import_module=unavailable))
    rows = steam.sweep(source, [0], max_frames=2)
    assert rows == [{'level': 0, 'mean_transmittance': 1.0, 'unusualness_mean': None}]
    assert (tmp_path / 'episodes/source/steam_curve.png').exists()
    def score(command, check):
        assert check
        assert command[1] == str(tmp_path / 'episodes/source__steam00/clip.mp4')
        np.save(command[2] + '/scores.npy', np.array([[0.2], [0.6]]))
    monkeypatch.setattr(steam.subprocess, 'run', score)
    rows = steam.sweep(source, [0], max_frames=2, score_cmd='scorer "{clip}" "{out}"')
    assert rows[0]['unusualness_mean'] == pytest.approx(0.4)


def test_tracking_columns():
    from vlma.steam import _tracking_columns
    assert _tracking_columns({'dancers': 1, 'detected_fraction': [0.5],
                              'mean_confidence': [0.7]}) == {
        'dancer_0_detected_fraction': 0.5, 'dancer_0_mean_confidence': 0.7}


def test_pose_sweep_integration(source, tmp_path, monkeypatch):
    import types
    import vlma.steam as steam
    monkeypatch.setattr(steam, '__file__', str(tmp_path / 'vlma' / 'steam.py'))
    calls = []
    def run_pose(clip, out, dancers=None):
        from pathlib import Path
        out = Path(out)
        assert Path(clip).resolve() != (out / 'clip.mp4').resolve()
        calls.append(dancers)
        out.mkdir(parents=True, exist_ok=True)
        count = len(frames(clip)[1])
        np.save(out / 'poses.npy', np.tile([10., 20., 0.8], (count, 1, 17, 1)))
        return {'dancers': 1, 'detected_fraction': [1.0], 'mean_confidence': [0.8]}
    real_import = steam.importlib.import_module
    monkeypatch.setattr(steam, 'importlib', types.SimpleNamespace(import_module=lambda name:
                        types.SimpleNamespace(run_pose=run_pose) if name == 'vlma.pose'
                        else real_import(name)))
    rows = steam.sweep(source, [0, 80], max_frames=2)
    assert calls == [None, 1, 1]
    assert rows[1]['dancer_0_detected_fraction'] == 1
    assert rows[1]['dancer_0_mean_keypoint_drift_px'] == 0
    assert rows[1]['unusualness_mean'] is None

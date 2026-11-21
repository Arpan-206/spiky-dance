"""Short real-video and real-eventify integration checks for the pitch panel."""
from pathlib import Path
import shutil
import subprocess

import cv2
import h5py
import numpy as np
import pytest

from vlma.viewer import _Events, run_viewer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def real_inputs(tmp_path_factory):
    folder = tmp_path_factory.mktemp('viewer_real')
    if not (ROOT / 'clips/sample_walk.mp4').exists():
        pytest.skip('sample clip missing, see README')
    source = cv2.VideoCapture(str(ROOT / 'clips/sample_walk.mp4'))
    assert source.isOpened()
    fps = source.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(folder / 'clip.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 180))
    assert writer.isOpened()
    count = min(10, int(fps * 1.0))
    for _ in range(count):
        ok, frame = source.read()
        assert ok
        writer.write(cv2.resize(frame, (320, 180)))
    writer.release()
    source.release()
    result = subprocess.run([str(ROOT / '.venv/bin/eventify'), 'export', str(folder / 'clip.mp4'), str(folder / 'events.h5'), '--sensor-size', '320,180'], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return folder, fps, count


def read_panel(path, fps, count):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(fps, abs=0.01)
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == count
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    assert len(frames) == count
    return frames


def test_clip_only_renders_readable_panel(tmp_path, real_inputs):
    folder, fps, count = real_inputs
    shutil.copyfile(folder / 'clip.mp4', tmp_path / 'clip.mp4')
    result = run_viewer(tmp_path)
    assert result['n_frames'] == count
    frames = read_panel(tmp_path / 'panel.mp4', fps, count)
    # Missing event data remains a neutral dark tile below its honest label.
    region = frames[-1][40:120, 650:1260]
    assert np.max(region.mean(axis=(0, 1))) < 25
    assert not (tmp_path / 'events.h5').exists()


def test_real_events_are_visible_and_output_fully_decodes(tmp_path, real_inputs):
    folder, fps, count = real_inputs
    for name in ('clip.mp4', 'events.h5'):
        shutil.copyfile(folder / name, tmp_path / name)
    result = run_viewer(tmp_path, tmp_path / 'with_events.mp4')
    assert not result['warnings']
    frames = read_panel(tmp_path / 'with_events.mp4', fps, count)
    # Exclude labels and look for colored ON or OFF pixels inside event imagery.
    assert any(np.any(np.ptp(f[50:290, 650:1260].astype(int), axis=2) > 45) for f in frames)


def test_real_events_root_layout_matches_group_layout(tmp_path, real_inputs):
    folder, _, _ = real_inputs
    root_path = tmp_path / 'root.h5'
    with h5py.File(folder / 'events.h5') as source, h5py.File(root_path, 'w') as target:
        group = source['events']
        for key in ('xs', 'ys', 'ts', 'ps'):
            target.create_dataset(key, data=group[key][:])
        target.attrs['sensor_shape'] = group.attrs['sensor_shape']
    grouped, root = _Events(folder / 'events.h5'), _Events(root_path)
    assert np.all(np.diff(grouped.ts) >= 0)
    for seconds in (0.0, 0.1, 0.4, 0.8):
        np.testing.assert_array_equal(grouped.frame(seconds), root.frame(seconds))


def test_short_stream_holds_last_frame_and_warns(tmp_path, real_inputs):
    folder, fps, count = real_inputs
    shutil.copyfile(folder / 'clip.mp4', tmp_path / 'clip.mp4')
    source = cv2.VideoCapture(str(folder / 'clip.mp4'))
    ok, frame = source.read()
    source.release()
    assert ok
    writer = cv2.VideoWriter(str(tmp_path / 'skeleton.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 180))
    writer.write(frame)
    writer.release()
    np.save(tmp_path / 'scores.npy', np.array([[0.2, 0.95], [0.4, 0.8]]))
    with pytest.warns(RuntimeWarning):
        result = run_viewer(tmp_path)
    assert any('skeleton.mp4' in message for message in result['warnings'])
    frames = read_panel(tmp_path / 'panel.mp4', fps, count)
    np.testing.assert_allclose(frames[0][350:590, 20:620].mean(), frames[-1][350:590, 20:620].mean(), atol=2)

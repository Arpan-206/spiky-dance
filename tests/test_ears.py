"""Synthetic-only checks for the simulated audio sensor and scripted robot."""
import h5py
import numpy as np
import pytest

from vlma.ears import (CHANNEL_HZ, FPS, SAMPLE_RATE, breathing_analysis,
                       cochlea_events, decode_audio, robot_trajectory, write_events)
from vlma.robot import load_model


def synthetic_breaths(pause=False):
    rng = np.random.default_rng(42)
    time = np.arange(64 * SAMPLE_RATE) / SAMPLE_RATE
    bursts = np.zeros_like(time)
    for onset in np.arange(2, 64, 4):
        if pause and 22 <= onset < 38:
            continue
        bursts += np.exp(-.5 * ((time - onset - .5) / .22) ** 2)
    return (.08 * bursts * rng.standard_normal(len(time))).astype(np.float32)


@pytest.mark.parametrize('pause', [False, True])
def test_cochlea_layout_cadence_and_pause(tmp_path, pause):
    audio = synthetic_breaths(pause)
    events, theta = cochlea_events(audio)
    destination = tmp_path / 'audio_events.h5'
    write_events(destination, events, theta)
    with h5py.File(destination) as file:
        group = file['events']
        for key, dtype in [('xs', 'int16'), ('ys', 'int16'), ('ts', 'int64'), ('ps', 'int8')]:
            assert group[key].dtype == np.dtype(dtype)
            assert group[key].shape == events['ts'].shape
        assert tuple(group.attrs['sensor_shape']) == (1, 64)
        np.testing.assert_allclose(group.attrs['channel_hz'], CHANNEL_HZ)
        assert group.attrs['simulated']
        assert len(group['ts']) > 1000
        assert np.all(np.diff(group['ts'][:]) >= 0)
        assert np.all(group['ys'][:] == 0)
        assert group['xs'][:].min() >= 0 and group['xs'][:].max() < 64
        assert set(np.unique(group['ps'][:])) == {0, 1}
        for channel in range(64):
            timestamps = events['ts'][events['xs'] == channel]
            assert np.all(np.diff(timestamps) >= 1000)
    analysis, envelope = breathing_analysis(audio)
    assert len(envelope) == 64 * FPS
    assert analysis['breaths_per_min'] == pytest.approx(15, rel=.15)
    if pause:
        assert len(analysis['pauses']) == 1
        start, end = analysis['pauses'][0]
        assert 17 < start < 21 and 37 < end < 41
    else:
        assert analysis['pauses'] == []
    onsets = np.array(analysis['snore_onsets_s'])
    assert np.all(np.diff(onsets) >= 1.5)
    assert len(onsets) == (12 if pause else 16)


def test_silence_has_no_events_or_invented_breaths():
    audio = np.zeros(4 * SAMPLE_RATE, np.float32)
    events, _ = cochlea_events(audio)
    assert len(events['ts']) == 0
    analysis, _ = breathing_analysis(audio)
    assert analysis['breaths_per_min'] is None
    assert analysis['snore_onsets_s'] == []
    assert analysis['pauses'] == []


def test_decode_real_ffmpeg_path_on_synthetic_audio(tmp_path):
    from scipy.io import wavfile
    path = tmp_path / 'synthetic.wav'
    samples = np.sin(2 * np.pi * 300 * np.arange(48000) / 48000).astype(np.float32) * .1
    wavfile.write(path, 48000, np.column_stack([samples, samples]))
    decoded = decode_audio(path)
    assert decoded.shape == (SAMPLE_RATE,)
    assert np.isfinite(decoded).all()
    # ffmpeg's stereo-to-mono downmix uses equal-power coefficients.
    assert np.sqrt(np.mean(decoded ** 2)) == pytest.approx(.1, rel=.01)


def test_scripted_robot_limits_standing_and_alarm():
    audio = synthetic_breaths(True)
    analysis, envelope = breathing_analysis(audio)
    model = load_model()
    qpos, alarm = robot_trajectory(envelope, analysis, model)
    assert qpos.shape == (64 * FPS, model.nq)
    assert np.isfinite(qpos).all()
    np.testing.assert_allclose(qpos[:, :7], np.repeat(model.key_qpos[0, :7][None], len(qpos), axis=0))
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            value = qpos[:, model.jnt_qposadr[j]]
            low, high = model.jnt_range[j]
            assert np.all(value >= low - 1e-6) and np.all(value <= high + 1e-6)
    assert alarm[30 * FPS] > .99
    for side in ('left', 'right'):
        address = model.jnt_qposadr[model.joint(f'{side}_shoulder_pitch_joint').id]
        assert qpos[30 * FPS, address] < -2
        assert qpos[10 * FPS, address] > -.2
    for j in range(model.njnt):
        if 'hip' in model.joint(j).name or 'knee' in model.joint(j).name or 'ankle' in model.joint(j).name:
            address = model.jnt_qposadr[j]
            np.testing.assert_array_equal(qpos[:, address], np.float32(model.key_qpos[0, address]))

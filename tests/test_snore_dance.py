"""Synthetic onset checks through the production clock and trajectory paths."""
import numpy as np
import pytest

from vlma.snore_dance import dance_clock, dance_trajectory
from vlma.robot import MODEL_PATH, load_model

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason='download assets/mujoco_menagerie/unitree_g1, see README')


def test_clock_is_monotonic_and_one_slice_per_onset():
    onsets = np.array([1., 3., 7., 10.])
    times = np.unique(np.r_[np.arange(0, 14, .05), onsets])
    progress, frozen, alarm, moves = dance_clock(times, onsets, [])
    assert np.all(np.diff(progress) >= -1e-12)
    np.testing.assert_allclose(progress[np.searchsorted(times, onsets)], [0, 1, 2, 3])
    np.testing.assert_allclose(np.diff(progress[np.searchsorted(times, onsets)]), 1)
    assert not frozen.any() and not alarm.any()
    assert moves[-1] == 4
    assert progress[-1] == 4


@needs_model
def test_pause_freezes_then_uses_ears_alarm_and_resumes():
    model = load_model()
    source = np.repeat(model.key_qpos[0][None], 80, axis=0)
    address = model.jnt_qposadr[model.joint('left_shoulder_pitch_joint').id]
    source[:, address] = np.linspace(-.1, -1.5, len(source))
    times = np.arange(0, 18, .05)
    analysis = {'snore_onsets_s': [1, 4, 13, 16], 'pauses': [[5, 13]]}
    qpos, progress, frozen, alarm, moves = dance_trajectory(source, 25, times, analysis, model)
    paused = (times >= 5) & (times < 13)
    np.testing.assert_allclose(progress[paused], progress[times == 5][0])
    np.testing.assert_allclose(qpos[frozen], np.repeat(qpos[frozen][0][None], frozen.sum(), axis=0))
    assert frozen[times == 5][0] and frozen[times == 7.95][0]
    assert not alarm[times == 7.95][0] and alarm[times == 8][0]
    assert qpos[alarm, address].max() < -2
    assert not alarm[times == 13][0]
    assert qpos[times == 13, address][0] > -2
    assert moves[-1] == 4


@needs_model
def test_no_pauses_never_replaces_real_dance_with_alarm():
    model = load_model()
    source = np.repeat(model.key_qpos[0][None], 80, axis=0)
    source[:, -1] = np.linspace(0, .1, len(source))
    analysis = {'snore_onsets_s': [1, 4, 7], 'pauses': []}
    qpos, _, frozen, alarm, _ = dance_trajectory(source, 25, np.arange(0, 11, .05), analysis, model)
    assert not alarm.any() and not frozen.any()
    assert np.ptp(qpos[:, -1]) > .02


def test_fully_paused_onset_interval_holds_without_nan():
    times = np.arange(0, 12, .05)
    progress, _, alarm, _ = dance_clock(times, [1, 4, 10], [[4, 10]])
    assert np.isfinite(progress).all()
    np.testing.assert_array_equal(progress[(times >= 4) & (times < 10)], 1.)
    assert alarm[times == 7][0]

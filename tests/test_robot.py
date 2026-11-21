from pathlib import Path

import mujoco
import numpy as np
import pytest

from vlma.robot import MODEL_PATH
from vlma.robot import (_arm_indices, generate_fixture, load_model, retarget,
                        run_robot, torso_frame)


pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason='download assets/mujoco_menagerie/unitree_g1, see README')


@pytest.fixture(scope='module')
def replay(tmp_path_factory):
    directory = tmp_path_factory.mktemp('robot_development')
    fixture = generate_fixture(directory / 'pose3d.npz')
    result = run_robot(directory, render=False)
    with np.load(fixture) as source:
        world, vis = source['world'].copy(), source['vis'].copy()
    with np.load(result['npz']) as output:
        qpos = output['qpos'].copy()
        assert output['fps'] == 30
        assert output['limit_hits'].shape == (120,)
        assert output['limit_hits'].dtype.kind in 'iu'
        assert output['joint_names'].dtype.kind == 'U'
    return directory, world, vis, qpos


def test_frame_count_motion_limits_and_standing(replay):
    _, _, _, qpos = replay
    model = load_model()
    assert qpos.shape == (120, model.nq)
    driven = []
    for side in ('left', 'right'):
        addresses, _ = _arm_indices(model, side)
        driven.extend(addresses)
        assert np.all(np.ptp(qpos[:, addresses], axis=0) > 0.1)
    fixed = np.setdiff1d(np.arange(model.nq), driven)
    np.testing.assert_allclose(qpos[:, fixed], np.broadcast_to(model.key_qpos[0, fixed],
                                                              qpos[:, fixed].shape), atol=1e-6)
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            values = qpos[:, model.jnt_qposadr[j]]
            low, high = model.jnt_range[j]
            assert np.all(values >= low - 1e-6)
            assert np.all(values <= high + 1e-6)


def test_raised_human_arm_raises_robot_wrist(replay):
    _, world, _, qpos = replay
    model = load_model()
    data = mujoco.MjData(model)
    for side, wrist in (('left', 15), ('right', 16)):
        heights = []
        for angles in qpos:
            data.qpos[:] = angles
            mujoco.mj_forward(model, data)
            heights.append(data.body(f'{side}_wrist_yaw_link').xpos[2])
        # Behind-head landmark wrist is above shoulder; hips are below it.
        assert -world[89, 0, wrist, 1] > -world[119, 0, wrist, 1] + .4
        assert heights[89] > heights[119] + .20
        data.qpos[:] = qpos[89]
        mujoco.mj_forward(model, data)
        assert heights[89] > data.body(f'{side}_shoulder_roll_link').xpos[2]


def test_nan_gaps_hold_and_recover(replay):
    _, world, vis, _ = replay
    world = world[:, 0].copy()
    world[30:35, 13:16] = np.nan
    world[60:65, 11] = np.nan
    model = load_model()
    qpos, _, _ = retarget(world, vis[:, 0], 30, model)
    addresses, _ = _arm_indices(model, 'left')
    np.testing.assert_array_equal(qpos[30:35, addresses],
                                  np.broadcast_to(qpos[29, addresses], (5, 4)))
    np.testing.assert_array_equal(qpos[60:65], np.broadcast_to(qpos[59], (5, model.nq)))
    assert np.isfinite(qpos).all()
    assert np.linalg.norm(qpos[40] - qpos[29]) > .01


def test_torso_axes_derived_from_geometry(replay):
    _, world, _, _ = replay
    basis = torso_frame(world[0, 0])
    np.testing.assert_allclose(basis.T, [[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = world[0, 0] @ rotation.T + [3, 2, 1]
    np.testing.assert_allclose(torso_frame(rotated), rotation @ basis, atol=1e-6)


def test_explicit_dancer_names_and_validation(replay):
    directory, _, _, _ = replay
    result = run_robot(directory, dancer=0, explicit_dancer=True, render=False)
    assert Path(result['npz']).name == 'robot_d0.npz'
    with pytest.raises(ValueError, match='Dancer'):
        run_robot(directory, dancer=1, render=False)


def test_renderer_failure_preserves_npz(replay, monkeypatch):
    import vlma.robot as robot
    directory, _, _, _ = replay

    def fail(*args):
        raise RuntimeError('test GL failure')

    monkeypatch.setattr(robot, 'render_video', fail)
    result = run_robot(directory)
    assert Path(result['npz']).is_file()
    assert result['video'] is None
    assert result['rendering_error'] == 'RuntimeError: test GL failure'

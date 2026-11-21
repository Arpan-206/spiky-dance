"""Kinematic replay only: fixed standing base/legs, arm IK and mj_forward, no dynamics.

Torso axes are measured per frame: left from shoulder separation, up from hips
to shoulders, forward = left cross up. No source coordinate axis is hardcoded.
Forward sign assumes a right-handed landmark coordinate system. Palm orientation
is not recovered from shoulder/elbow/wrist directions.
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'assets/mujoco_menagerie/unitree_g1/scene.xml'
ARM_PARTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')


def load_model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def _unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError('Missing or degenerate landmark direction')
    return vector / length


def torso_frame(points):
    left = _unit(points[11] - points[12])
    up = (points[11] + points[12] - points[23] - points[24]) / 2
    up = _unit(up - left * np.dot(up, left))
    forward = _unit(np.cross(left, up))
    return np.stack((forward, left, up), axis=1)


def _arm_indices(model, side):
    ids = [model.joint(f'{side}_{part}_joint').id for part in ARM_PARTS]
    return np.array([model.jnt_qposadr[j] for j in ids]), model.jnt_range[ids]


def retarget(world, vis, fps, model=None):
    """Fit unit upper/forearm directions to actual G1 forward kinematics."""
    model = load_model() if model is None else model
    data = mujoco.MjData(model)
    standing = model.key_qpos[0].copy()
    previous = standing.copy()
    qpos = np.empty((len(world), model.nq), dtype=np.float32)
    hits = np.zeros(len(world), dtype=np.int32)
    frames = []
    # A 55 ms time constant preserves swing while suppressing landmark jitter.
    alpha = 1 - np.exp(-1 / (fps * 0.055))
    for frame, (points, visibility) in enumerate(zip(world, vis)):
        target = previous.copy()
        try:
            if not np.all(visibility[[11, 12, 23, 24]] >= 0.3):
                raise ValueError('Torso not visible')
            basis = torso_frame(points)
            frames.append(basis)
        except ValueError:
            basis = None
        for side, landmarks in (('left', (11, 13, 15)), ('right', (12, 14, 16))):
            addresses, ranges = _arm_indices(model, side)
            if basis is None or not np.all(visibility[list(landmarks)] >= 0.3):
                continue
            try:
                shoulder, elbow, wrist = points[list(landmarks)]
                desired = np.concatenate((_unit(elbow - shoulder) @ basis,
                                          _unit(wrist - elbow) @ basis))
            except ValueError:
                continue
            data.qpos[:] = standing
            anchor_id = model.body(f'{side}_shoulder_roll_link').id
            elbow_id = model.body(f'{side}_elbow_link').id
            wrist_id = model.body(f'{side}_wrist_yaw_link').id

            def residual(angles):
                data.qpos[addresses] = angles
                mujoco.mj_forward(model, data)
                upper = _unit(data.xpos[elbow_id] - data.xpos[anchor_id])
                fore = _unit(data.xpos[wrist_id] - data.xpos[elbow_id])
                return np.concatenate((upper, fore)) - desired

            solution = least_squares(residual, previous[addresses],
                                     bounds=(ranges[:, 0], ranges[:, 1]),
                                     max_nfev=35, diff_step=1e-4)
            angles = np.clip(solution.x, ranges[:, 0], ranges[:, 1])
            # Count saturated IK targets before the temporal filter.
            hits[frame] += np.count_nonzero(
                (angles - ranges[:, 0] < 1e-4) | (ranges[:, 1] - angles < 1e-4))
            target[addresses] = previous[addresses] + alpha * (angles - previous[addresses])
        previous = target
        qpos[frame] = target
    # Report the first valid basis; averaging turning frames destroys orthogonality.
    axes = frames[0].T.round(4).tolist() if frames else None
    return qpos, hits, axes


def render_video(model, qpos, fps, destination):
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0, 0, 0.85)
    camera.distance = 2.7
    camera.azimuth = 0
    camera.elevation = -8
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(str(destination), fps=fps, codec='libx264',
                                macro_block_size=1) as writer:
            for frame in qpos:
                data.qpos[:] = frame
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())


def run_robot(episode_dir, dancer=0, *, explicit_dancer=False, render=True):
    episode_dir = Path(episode_dir)
    with np.load(episode_dir / 'pose3d.npz', allow_pickle=False) as source:
        world = source['world']
        vis = source['vis']
        fps = float(np.asarray(source['fps']).item())
    if world.ndim != 4 or world.shape[2:] != (33, 3) or vis.shape != world.shape[:-1]:
        raise ValueError('Expected world (frames, dancers, 33, 3) and matching vis')
    if not np.isfinite(fps) or fps <= 0 or not len(world):
        raise ValueError('Expected positive finite fps and at least one frame')
    if dancer < 0 or dancer >= world.shape[1]:
        raise ValueError(f'Dancer {dancer} outside available range 0..{world.shape[1] - 1}')
    model = load_model()
    qpos, hits, axes = retarget(world[:, dancer], vis[:, dancer], fps, model)
    stem = f'robot_d{dancer}' if explicit_dancer or dancer != 0 else 'robot'
    npz_path = episode_dir / f'{stem}.npz'
    video_path = episode_dir / f'{stem}.mp4'
    names = np.array([model.joint(i).name for i in range(model.njnt)])
    np.savez_compressed(npz_path, qpos=qpos, joint_names=names,
                        limit_hits=hits, fps=fps,
                        joint_qposadr=model.jnt_qposadr,
                        torso_axes_forward_left_up=np.asarray(axes if axes is not None else [],
                                                              dtype=np.float32))
    result = dict(npz=str(npz_path), frames=len(qpos), fps=fps,
                  limit_hits=int(hits.sum()), torso_axes_forward_left_up=axes,
                  video=None, rendering_error=None)
    if render:
        try:
            render_video(model, qpos, fps, video_path)
            result['video'] = str(video_path)
        except Exception as error:
            result['rendering_error'] = f'{type(error).__name__}: {error}'
    return result


def generate_fixture(destination=None):
    """Development-only scripted 4 s sequence, coordinates derived explicitly here.

    Synthetic coordinates: +X anatomical left, -Y up, -Z forward. The palm flip
    changes pinky/index landmarks only, deliberately not the arm joint targets.
    """
    destination = Path(destination or ROOT / 'tests/fixtures/robot_development/pose3d.npz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Coordinates below are robot-like (forward, left, up), converted on save.
    key_elbows = np.array([[.28, .20, .45], [.28, .20, .45],
                           [.12, .26, .36], [0, .31, .67], [.02, .30, .20]])
    key_wrists = np.array([[.55, .20, .45], [.55, .20, .45],
                           [.02, -.20, .50], [-.10, .08, .68], [.02, .15, .04]])
    world = np.full((120, 1, 33, 3), np.nan, dtype=np.float32)
    vis = np.zeros((120, 1, 33), dtype=np.float32)
    for f in range(120):
        phase = f / 119 * 4
        k = min(int(phase), 3)
        blend = phase - k
        elbow = (1 - blend) * key_elbows[k] + blend * key_elbows[k + 1]
        wrist = (1 - blend) * key_wrists[k] + blend * key_wrists[k + 1]
        points = {11: [0, .20, .45], 12: [0, -.20, .45],
                  23: [0, .13, 0], 24: [0, -.13, 0]}
        for sign, ids in ((1, (13, 15, 17, 19)), (-1, (14, 16, 18, 20))):
            e, w = elbow.copy(), wrist.copy()
            e[1] *= sign
            w[1] *= sign
            points[ids[0]], points[ids[1]] = e, w
            flip = np.cos(min(phase, 1) * np.pi)
            points[ids[2]] = w + np.array([0, .03 * flip, 0])
            points[ids[3]] = w - np.array([0, .03 * flip, 0])
        for index, value in points.items():
            forward, left, up = value
            world[f, 0, index] = (left, -up, -forward)
            vis[f, 0, index] = 1
    np.savez_compressed(destination, world=world, vis=vis, image=world.copy(), fps=30.,
                        development_fixture=True,
                        sequence='front,palm flip,opposite shoulders,behind head,hips')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode_dir', type=Path, nargs='?')
    parser.add_argument('--dancer', type=int, default=None)
    parser.add_argument('--generate-fixture', action='store_true')
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    if args.generate_fixture:
        fixture = generate_fixture()
        print(f'DEVELOPMENT FIXTURE: {fixture}')
        args.episode_dir = args.episode_dir or fixture.parent
    if args.episode_dir is None:
        parser.error('episode_dir is required unless --generate-fixture is used')
    result = run_robot(args.episode_dir, dancer=args.dancer or 0,
                       explicit_dancer=args.dancer is not None, render=not args.no_render)
    print(json.dumps(result, indent=2))
    if result['rendering_error']:
        command = ['.venv/bin/python', '-m', 'vlma.robot', str(args.episode_dir)]
        if args.dancer is not None:
            command += ['--dancer', str(args.dancer)]
        print('Render outside sandbox: ' + shlex.join(command))


if __name__ == '__main__':
    main()

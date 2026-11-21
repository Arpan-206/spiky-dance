"""Real macarena replay driven by detected snores, with a scripted pause alarm."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

from vlma.ears import (FPS, SAMPLE_RATE, _finish_video, _text, _video_pipe,
                       breathing_analysis, decode_audio, robot_trajectory)
from vlma.robot import ARM_PARTS, ROOT, load_model, render_video

PROJECT = ROOT.parent
SLICES_PER_LOOP = 8


def select_episode():
    """Select the largest sum of peak-to-peak ranges over the eight arm joints."""
    scores = {}
    for path in sorted((ROOT / 'episodes').glob('team_0*/robot.npz')):
        with np.load(path, allow_pickle=False) as source:
            addresses = [int(a) for n, a in zip(source['joint_names'], source['joint_qposadr'])
                         if any(part in str(n) for part in ARM_PARTS)]
            scores[str(path.parent)] = float(np.ptp(source['qpos'][:, addresses], axis=0).sum())
    if not scores:
        raise ValueError('No team robot recordings found')
    return Path(max(scores, key=scores.get)), scores


def dance_clock(times, onsets, pauses):
    """Unwrapped slice progress, with one slice between consecutive onsets.

    Pause time is removed before interpolation. The last onset plays one final
    slice using the median onset interval, then holds instead of inventing snores.
    A completely paused interval holds until the next onset starts its next slice.
    """
    times = np.asarray(times, dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    if onsets.ndim != 1 or not np.isfinite(onsets).all() or np.any(np.diff(onsets) <= 0):
        raise ValueError('Onsets must be finite and strictly increasing')
    pauses = np.asarray(pauses, dtype=float).reshape(-1, 2)
    if (not np.isfinite(pauses).all() or np.any(pauses[:, 1] <= pauses[:, 0])
            or np.any(pauses[1:, 0] < pauses[:-1, 1])):
        raise ValueError('Pauses must be ordered, disjoint, finite intervals')

    def active_time(values):
        result = np.asarray(values, dtype=float).copy()
        for start, end in pauses:
            result -= np.clip(values - start, 0, end - start)
        return result

    progress = np.zeros_like(times)
    if len(onsets):
        anchors = active_time(onsets)
        index = np.searchsorted(onsets, times, side='right') - 1
        valid = index >= 0
        i = np.clip(index, 0, len(onsets) - 1)
        terminal = float(np.median(np.diff(onsets))) if len(onsets) > 1 else 3.7
        ends = np.r_[anchors[1:], anchors[-1] + terminal]
        lengths = ends[i] - anchors[i]
        fraction = np.divide(active_time(times) - anchors[i], lengths,
                             out=np.zeros_like(times), where=lengths > 1e-8)
        progress[valid] = i[valid] + np.clip(fraction[valid], 0, 1)
    frozen = np.zeros(times.shape, dtype=bool)
    alarm = np.zeros(times.shape, dtype=bool)
    for start, end in pauses:
        frozen |= (times >= start) & (times < min(start + 3, end))
        alarm |= (times >= start + 3) & (times < end)
    moves = np.searchsorted(onsets, times, side='right')
    return progress, frozen, alarm, moves


def dance_trajectory(source_qpos, source_fps, times, analysis, model):
    progress, frozen, alarm, moves = dance_clock(
        times, analysis['snore_onsets_s'], analysis['pauses'])
    # Eight equal slices span the complete recorded take, including its last pose.
    position = (progress % SLICES_PER_LOOP) / SLICES_PER_LOOP * (len(source_qpos) - 1)
    low = position.astype(int)
    high = np.minimum(low + 1, len(source_qpos) - 1)
    blend = (position - low)[:, None]
    qpos = ((1 - blend) * source_qpos[low] + blend * source_qpos[high]).astype(np.float32)
    # Use ears.py's actual full arms-up scripted pose, after its ramp has settled.
    if np.any(alarm):
        scripted, _ = robot_trajectory(np.zeros(8 * FPS), {'pauses': [[0, 10]]}, model)
        qpos[alarm] = scripted[6 * FPS]
    return qpos, progress, frozen, alarm, moves


def compose(destination, robot_path, audio_path, events, envelope, analysis,
            times, frozen, alarm, moves):
    duration = analysis['duration_s']
    demo = analysis.get('demo_pause')
    mute = (demo['start_s'], demo['duration_s']) if demo else None
    background = np.full((720, 1280, 3), (20, 17, 14), np.uint8)
    _text(background, 'SNORE-POWERED MACARENA', (24, 38), 1., (235, 239, 242))
    _text(background, 'simulated silicon cochlea (from a phone recording)', (24, 72), .73)
    if demo:
        _text(background, 'demo: pause inserted by us', (24, 101), .65, (80, 196, 255))
    _text(background, 'robot: real macarena replay, time-warped to the snoring', (20, 127), .51)
    _text(background, 'COCHLEA SPIKE RASTER', (686, 133), .62)
    _text(background, '4 kHz', (670, 160), .42)
    _text(background, '50 Hz', (670, 402), .42)
    _text(background, 'breathing envelope + snore onsets', (686, 456), .62)
    _text(background, 'macarena moves', (686, 625), .88)
    _text(background, 'not a medical device', (24, 705), .56, (150, 160, 174))
    _text(background, 'yellow: ON / snore onset    blue: OFF', (686, 705), .48)
    ts = events['ts'] / 1e6
    channels, polarities = events['xs'], events['ps']
    envelope_times = np.arange(len(envelope)) / FPS
    amp = envelope / max(float(np.percentile(envelope, 98)), 1e-9)
    onsets = np.asarray(analysis['snore_onsets_s'])
    capture = cv2.VideoCapture(str(robot_path))
    process = _video_pipe(destination, 1280, 720, audio_path=audio_path,
                          duration=duration, demo_pause=mute)
    left, right, top, bottom = 720, 1254, 160, 400
    try:
        for i, now in enumerate(times):
            frame = background.copy()
            ok, robot = capture.read()
            if not ok:
                raise RuntimeError(f'Robot render ended at frame {i}')
            frame[140:620, :640] = robot
            start = now - 5.7
            for second in range(math.ceil(start), math.floor(now + .3) + 1):
                x = int(left + (second - start) / 6 * (right - left))
                cv2.line(frame, (x, top), (x, bottom), (46, 42, 37), 1)
                if second >= 0:
                    _text(frame, f'{second}s', (x - 10, 424), .4)
            for channel in (0, 16, 32, 48, 63):
                y = int(bottom - channel / 63 * (bottom - top))
                cv2.line(frame, (left, y), (right, y), (38, 34, 30), 1)
            a, b = np.searchsorted(ts, [max(0, start), now], side='right')
            x = np.clip((left + (ts[a:b] - start) / 6 * (right - left)).astype(int), left, right)
            y = (bottom - channels[a:b] / 63 * (bottom - top)).astype(int)
            for polarity, color in ((0, (255, 145, 65)), (1, (55, 215, 255))):
                selected = polarities[a:b] == polarity
                frame[y[selected], x[selected]] = color
                frame[np.minimum(y[selected] + 1, bottom), x[selected]] = color
            for p0, p1 in analysis['pauses']:
                x0 = int(left + np.clip((p0 - start) / 6, 0, 1) * (right - left))
                x1 = int(left + np.clip((p1 - start) / 6, 0, 1) * (right - left))
                if x1 > x0:
                    cv2.rectangle(frame, (x0, 478), (x1, 575), (55, 34, 83), -1)
            selected = (envelope_times >= start) & (envelope_times <= now)
            points = np.column_stack((left + (envelope_times[selected] - start) / 6 * (right - left),
                                      571 - np.clip(amp[selected], 0, 1.2) * 72)).astype(np.int32)
            if len(points) > 1:
                cv2.polylines(frame, [points], False, (175, 215, 179), 2, cv2.LINE_AA)
            for onset in onsets[(onsets >= start) & (onsets <= now)]:
                x0 = int(left + (onset - start) / 6 * (right - left))
                cv2.line(frame, (x0, 577), (x0, 588), (55, 215, 255), 2)
            playhead = int(left + 5.7 / 6 * (right - left))
            cv2.line(frame, (playhead, top), (playhead, bottom), (225, 230, 236), 1)
            cv2.line(frame, (playhead, 478), (playhead, 588), (225, 230, 236), 1)
            status = ('breathing stopped: alarm' if alarm[i] else
                      'breathing stopped: holding pose' if frozen[i] else 'dancing')
            _text(frame, status, (24, 658), .8,
                  (75, 145, 255) if alarm[i] else (175, 230, 190))
            _text(frame, f'driven by snores: {moves[i]}', (686, 666), .88, (55, 215, 255))
            _text(frame, f'{now:04.1f} / {duration:.1f} s', (1010, 35), .62)
            process.stdin.write(frame.tobytes())
    finally:
        capture.release()
        _finish_video(process)


def run(episode=None, ears=None, audio=None, out=None):
    selected, scores = select_episode()
    episode = Path(episode) if episode else selected
    ears = Path(ears) if ears else PROJECT / 'episodes/snore'
    audio = Path(audio) if audio else ROOT / 'clips/snore.m4a'
    analysis = json.loads((ears / 'breathing.json').read_text())
    demo = analysis.get('demo_pause')
    out = Path(out) if out else PROJECT / ('episodes/snore_dance__demo_pause' if demo
                                          else 'episodes/snore_dance')
    out.mkdir(parents=True, exist_ok=True)
    decoded = decode_audio(audio)
    duration = len(decoded) / SAMPLE_RATE
    if demo:
        start, seconds = demo['start_s'], demo['duration_s']
        decoded[int(start * SAMPLE_RATE):int((start + seconds) * SAMPLE_RATE)] = 0
    _, envelope = breathing_analysis(decoded)
    analysis['duration_s'] = duration
    times = np.arange(math.ceil(duration * FPS)) / FPS
    with np.load(episode / 'robot.npz', allow_pickle=False) as source:
        qpos, progress, frozen, alarm, moves = dance_trajectory(
            source['qpos'], float(source['fps']), times, analysis, load_model())
    npz_path = out / 'snore_dance.npz'
    np.savez_compressed(npz_path, qpos=qpos, fps=FPS, time_s=times, slice_progress=progress,
                        frozen=frozen, alarm=alarm, moves=moves, source_episode=str(episode),
                        slices_per_loop=SLICES_PER_LOOP)
    result = dict(episode=str(episode), arm_range_scores=scores, moves=int(moves[-1]),
                  frames=len(times), duration_s=duration, alarm_frames=int(alarm.sum()),
                  qpos=str(npz_path), video=None)
    command = [sys.executable, '-m', 'vlma.snore_dance', '--render-only', str(npz_path)]
    rendered = subprocess.run(command, capture_output=True, text=True)
    if rendered.returncode:
        result['render_error'] = rendered.stderr.strip() or f'exit {rendered.returncode}'
        result['render_command'] = ('UV_CACHE_DIR=$TMPDIR/uv-cache uv run python -m '
                                    'vlma.snore_dance --render-only ' + shlex.quote(str(npz_path)))
        print(json.dumps(result, indent=2), flush=True)
        raise RuntimeError(result['render_error'])
    with h5py.File(ears / 'audio_events.h5') as file:
        events = {name: file['events'][name][:] for name in ('ts', 'xs', 'ps')}
    destination = out / 'snore_dance.mp4'
    compose(destination, out / 'robot_render.mp4', audio, events, envelope,
            analysis, times, frozen, alarm, moves)
    result['video'] = str(destination)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path)
    parser.add_argument('--ears', type=Path)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--render-only', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render_only:
        with np.load(args.render_only, allow_pickle=False) as source:
            render_video(load_model(), source['qpos'], FPS,
                         args.render_only.with_name('robot_render.mp4'))
    else:
        run(args.episode, args.ears, args.audio, args.out)


if __name__ == '__main__':
    main()

"""Simulated silicon cochlea from phone audio, not a medical device.

64 log-spaced second-order Butterworth bandpasses, half-wave rectification,
200 Hz low-pass envelopes and log compression feed DVS-style send-on-delta
comparators. This is a simulation inspired by silicon cochleae, not DAS hardware.
The standing robot is a scripted kinematic reaction, not measured human motion.
"""
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
from scipy import ndimage, signal

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = '/opt/homebrew/bin/ffmpeg'
SAMPLE_RATE = 16000
FPS = 20
CHANNEL_HZ = np.geomspace(50, 4000, 64)
METHOD = ('simulated 64-channel second-order Butterworth cochlea; half-wave, '
          '200 Hz low-pass, log1p, send-on-delta with 1 ms refractory; '
          '50-1000 Hz RMS at 20 Hz, 200 ms smoothing, autocorrelation cadence, '
          'prominent-peak snore onsets; pauses between onsets > max(10 s, 2.5 median gap)')


def decode_audio(audio_path):
    """Decode locally; never persist or log the decoded recording."""
    result = subprocess.run(
        [FFMPEG, '-v', 'error', '-i', str(audio_path), '-f', 'f32le',
         '-ac', '1', '-ar', str(SAMPLE_RATE), 'pipe:1'],
        capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Audio must contain finite samples')
    return audio


def cochlea_events(audio, sample_rate=SAMPLE_RATE, theta=1.4):
    """Return sorted simulated events and the actual comparator threshold.

    Envelopes are sampled at 4 kHz, four times the maximum per-channel event
    rate. Reference moves by exactly theta per event, retaining residual contrast.
    No events are fabricated or randomly added to meet the display rate target.
    """
    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Expected nonempty finite mono audio')
    if sample_rate != SAMPLE_RATE:
        raise ValueError('Cochlea expects decoded 16 kHz audio')
    step = 4
    lowpass = signal.butter(2, 200, fs=sample_rate, output='sos')
    logs = np.empty((math.ceil(len(audio) / step), len(CHANNEL_HZ)), np.float32)
    for channel, centre in enumerate(CHANNEL_HZ):
        edges = centre * np.array([2 ** (-1 / 6), 2 ** (1 / 6)])
        band = signal.sosfilt(signal.butter(2, edges, btype='bandpass',
                                          fs=sample_rate, output='sos'), audio)
        envelope = signal.sosfilt(lowpass, np.maximum(band, 0))[::step]
        envelope = np.maximum(envelope, 0)
        floor = max(float(np.percentile(envelope, 90)) * .005, 1e-8)
        logs[:, channel] = np.log1p(envelope / floor)
    # 1.4 natural-log units gives a readable real-snore raster without added noise.
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError('theta must be positive and finite')
    reference = logs[0].copy()
    last = np.full(len(CHANNEL_HZ), -4, dtype=np.int64)
    times, channels, polarities = [], [], []
    for i in range(1, len(logs)):
        delta = logs[i] - reference
        active = np.flatnonzero((np.abs(delta) >= theta) & (i - last >= 4))
        if len(active):
            positive = delta[active] > 0
            reference[active] += np.where(positive, theta, -theta)
            last[active] = i
            times.append(np.full(len(active), i * 250, dtype=np.int64))
            channels.append(active.astype(np.int16))
            polarities.append(positive.astype(np.int8))
    xs = np.concatenate(channels) if channels else np.empty(0, np.int16)
    events = dict(xs=xs, ys=np.zeros(len(xs), np.int16),
                  ts=np.concatenate(times) if times else np.empty(0, np.int64),
                  ps=np.concatenate(polarities) if polarities else np.empty(0, np.int8))
    return events, theta


def write_events(destination, events, theta):
    with h5py.File(destination, 'w') as file:
        group = file.create_group('events')
        for name, values in events.items():
            group.create_dataset(name, data=values, compression='gzip', shuffle=True)
        group.attrs['sensor_shape'] = (1, len(CHANNEL_HZ))
        group.attrs['channel_hz'] = CHANNEL_HZ
        group.attrs['simulated'] = True
        group.attrs['theta'] = theta
        group.attrs['refractory_us'] = 1000
        group.attrs['method'] = METHOD


def breathing_analysis(audio, sample_rate=SAMPLE_RATE):
    """Estimate rhythm and snore onsets from low-band energy, not clinical breaths."""
    filtered = signal.sosfilt(signal.butter(2, [50, 1000], btype='bandpass',
                                           fs=sample_rate, output='sos'), audio)
    block = sample_rate // FPS
    padded = np.pad(filtered, (0, (-len(filtered)) % block))
    envelope = np.sqrt(np.mean(padded.reshape(-1, block) ** 2, axis=1))
    envelope = ndimage.gaussian_filter1d(envelope, FPS * .2)
    baseline, top = np.percentile(envelope, [10, 95])
    dynamic = top - baseline
    if dynamic <= 1e-9:
        return dict(breaths_per_min=None, snore_onsets_s=[], pauses=[],
                    cadence_correlation=0.), envelope
    centered = envelope - envelope.mean()
    correlation = signal.correlate(centered, centered, mode='full', method='fft')
    correlation = correlation[len(envelope) - 1:] / max(np.dot(centered, centered), 1e-20)
    lo, hi = int(1.5 * FPS), min(int(8 * FPS), len(envelope) // 2)
    candidates, _ = signal.find_peaks(correlation[lo:hi + 1])
    period, strength = None, 0.
    if len(candidates):
        best = int(candidates[np.argmax(correlation[candidates + lo])] + lo)
        strength = float(correlation[best])
        if strength >= .1:
            period = best / FPS
    peaks, props = signal.find_peaks(
        envelope, distance=max(int(1.5 * FPS), int(.65 * (period or 0) * FPS)),
        prominence=.2 * dynamic, height=baseline + .15 * dynamic)
    if len(peaks):
        # Rising quarter-prominence crossing gives an onset, rather than a peak time.
        onsets = np.maximum(signal.peak_widths(envelope, peaks, rel_height=.75)[2],
                            peaks - 1.5 * FPS) / FPS
        for i in range(1, len(onsets)):
            onsets[i] = max(onsets[i], onsets[i - 1] + 1.5)
    else:
        onsets = np.empty(0)
    gaps = np.diff(onsets)
    median_gap = float(np.median(gaps)) if len(gaps) else None
    threshold = max(10., 2.5 * median_gap) if median_gap else math.inf
    pauses = [[float(a), float(b)] for a, b in zip(onsets[:-1], onsets[1:])
              if b - a > threshold]
    cadence = period or median_gap
    bpm = 60 / cadence if cadence and len(peaks) >= 3 else None
    return dict(breaths_per_min=None if bpm is None else round(bpm, 2),
                snore_onsets_s=onsets.round(3).tolist(), pauses=pauses,
                cadence_correlation=round(strength, 3)), envelope


def robot_trajectory(envelope, analysis, model):
    """Standing, clamped scripted motion; base and legs stay at the standing key."""
    qpos = np.repeat(model.key_qpos[0][None], len(envelope), axis=0)
    amplitude = ndimage.gaussian_filter1d(envelope, 4)
    amplitude = np.clip(amplitude / max(float(np.percentile(amplitude, 95)), 1e-9), 0, 1)
    times = np.arange(len(envelope)) / FPS
    alarm = np.zeros(len(times))
    for start, end in analysis['pauses']:
        alarm = np.maximum(alarm, np.clip((times - start - 3) / .7, 0, 1)
                           * np.clip((end + 1 - times) / .8, 0, 1))
    alarm = ndimage.gaussian_filter1d(alarm, 2)

    def set_joint(name, values):
        try:
            joint = model.joint(name)
        except KeyError:
            return
        low, high = model.jnt_range[joint.id]
        qpos[:, model.jnt_qposadr[joint.id]] = np.clip(values, low, high)

    set_joint('waist_pitch_joint', (.25 - .11 * amplitude) * (1 - alarm))
    set_joint('waist_roll_joint', .025 * np.sin(times * .6) * (1 - alarm))
    for side, sign in (('left', 1), ('right', -1)):
        set_joint(f'{side}_shoulder_pitch_joint', -.08 - 2.3 * alarm)
        set_joint(f'{side}_shoulder_roll_joint', sign * (.08 + .2 * alarm))
        set_joint(f'{side}_shoulder_yaw_joint', np.zeros(len(times)))
        set_joint(f'{side}_elbow_joint', .12 + .3 * alarm)
    return qpos.astype(np.float32), alarm


def _text(frame, message, xy, scale=.55, color=(210, 217, 227)):
    cv2.putText(frame, message, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _video_pipe(destination, width, height, *, audio_path=None, duration=None, demo_pause=None):
    command = [FFMPEG, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{width}x{height}', '-r', str(FPS), '-i', 'pipe:0']
    if audio_path is not None:
        command += ['-i', str(audio_path), '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac']
        if demo_pause:
            start, seconds = demo_pause
            command += ['-af', f"volume=0:enable='between(t,{start},{start + seconds})'"]
    command += ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart']
    if duration is not None:
        command += ['-t', str(duration)]
    return subprocess.Popen(command + [str(destination)], stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def _finish_video(process):
    process.stdin.close()
    error = process.stderr.read().decode(errors='replace')
    if process.wait() != 0:
        raise RuntimeError(f'ffmpeg video encoding failed: {error}')


def render_saved_robot(npz_path):
    """Run separately so a native GL crash cannot destroy the ears analysis."""
    from vlma.robot import load_model, render_video
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as source:
        qpos = source['qpos']
        demo = bool(source['demo_pause'])
        alarm = source['alarm']
    raw = npz_path.with_name('sleep_robot_unlabelled.mp4')
    destination = npz_path.with_suffix('.mp4')
    render_video(load_model(), qpos, FPS, raw)
    capture = cv2.VideoCapture(str(raw))
    process = _video_pipe(destination, 640, 480)
    try:
        for i in range(len(qpos)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Robot renderer stopped at frame {i}')
            cv2.rectangle(frame, (0, 0), (640, 74 if demo else 51), (14, 18, 25), -1)
            _text(frame, 'SIMULATED ROBOT | scripted reaction', (16, 23), .62)
            _text(frame, 'demo: pause inserted by us' if demo else 'Standing kinematic animation',
                  (16, 46), .5)
            _text(frame, 'SCRIPTED ALARM' if alarm[i] > .5 else 'SCRIPTED SLEEP',
                  (16, 468), .55, (70, 150, 255) if alarm[i] > .5 else (180, 220, 200))
            process.stdin.write(frame.tobytes())
    finally:
        capture.release()
        _finish_video(process)
    raw.unlink()
    return destination


def ears_video(destination, audio_path, events, envelope, analysis, duration,
               robot_video=None, demo_pause=None):
    """Compose a 1280x720 local video, muxing the real source audio with AAC."""
    width, height = 1280, 720
    background = np.full((height, width, 3), (17, 14, 12), np.uint8)
    cv2.rectangle(background, (865, 96), (1255, 635), (26, 23, 20), -1)
    _text(background, 'simulated silicon cochlea (from a phone recording)', (28, 36), .75)
    subtitle = ('demo: pause inserted by us' if demo_pause else
                '64 frequency channels | ON / OFF contrast events | original recording audio')
    _text(background, subtitle, (28, 65), .53, (80, 196, 255) if demo_pause else (160, 168, 181))
    _text(background, 'SPIKE RASTER', (78, 111), .54)
    _text(background, 'ON', (654, 111), .5, (55, 215, 255))
    _text(background, 'OFF', (710, 111), .5, (255, 145, 65))
    _text(background, '4 kHz', (22, 145), .45)
    _text(background, '50 Hz', (22, 408), .45)
    _text(background, 'breathing (estimated)', (78, 467), .62)
    _text(background, 'robot (scripted reaction)', (889, 126), .61)
    bpm = analysis['breaths_per_min']
    _text(background, f'breaths/min: {bpm:.1f}' if bpm else 'breaths/min: insufficient rhythm',
          (78, 665), .95, (218, 234, 242))
    _text(background, f"{analysis['events_per_s']:,.0f} simulated events/s  |  "
          f"{len(analysis['pauses'])} pauses flagged", (78, 698), .52)
    _text(background, 'Not a medical device', (889, 686), .5, (145, 155, 170))
    if demo_pause:
        _text(background, 'demo: pause inserted by us', (889, 153), .48, (80, 196, 255))
    capture = cv2.VideoCapture(str(robot_video)) if robot_video else None
    process = _video_pipe(destination, width, height, audio_path=audio_path,
                          duration=duration, demo_pause=demo_pause)
    ts = events['ts'] / 1e6
    xs, ps = events['xs'], events['ps']
    amp = envelope / max(float(np.percentile(envelope, 98)), 1e-9)
    envelope_times = np.arange(len(envelope)) / FPS
    onsets = np.asarray(analysis['snore_onsets_s'])
    left, right, top, bottom = 78, 823, 135, 408
    last_robot = None
    try:
        for frame_index in range(math.ceil(duration * FPS)):
            now = frame_index / FPS
            frame = background.copy()
            # Fixed six-second window, including pre-roll, with the playhead at now.
            start, end = now - 5.7, now + .3
            for second in range(math.ceil(start), math.floor(end) + 1):
                x = int(left + (second - start) / 6 * (right - left))
                cv2.line(frame, (x, top), (x, bottom), (39, 35, 31), 1)
                if second >= 0:
                    _text(frame, f'{second}s', (x - 9, 431), .4, (137, 148, 165))
            for channel in (0, 16, 32, 48, 63):
                y = int(bottom - channel / 63 * (bottom - top))
                cv2.line(frame, (left, y), (right, y), (33, 29, 26), 1)
            a, b = np.searchsorted(ts, [max(0, start), now], side='right')
            x = np.clip((left + (ts[a:b] - start) / 6 * (right - left)).astype(int), left, right)
            y = (bottom - xs[a:b] / 63 * (bottom - top)).astype(int)
            # Rasterize OFF first and ON second, preserving visible polarity colors.
            for polarity, color in ((0, (255, 145, 65)), (1, (55, 215, 255))):
                selected = ps[a:b] == polarity
                xx, yy = x[selected], y[selected]
                frame[yy, xx] = color
                frame[np.minimum(yy + 1, bottom), xx] = color
            for pause_start, pause_end in analysis['pauses']:
                x0 = int(left + max(0, pause_start - start) / 6 * (right - left))
                x1 = int(left + min(6, pause_end - start) / 6 * (right - left))
                if x1 > x0 and x1 >= left and x0 <= right:
                    cv2.rectangle(frame, (max(left, x0), 490), (min(right, x1), 602), (48, 35, 88), -1)
            selected = (envelope_times >= start) & (envelope_times <= now)
            points = np.column_stack((left + (envelope_times[selected] - start) / 6 * (right - left),
                                      600 - np.clip(amp[selected], 0, 1.2) * 88)).astype(np.int32)
            if len(points) >= 2:
                cv2.polylines(frame, [points], False, (175, 215, 179), 2, cv2.LINE_AA)
            for onset in onsets[(onsets >= start) & (onsets <= now)]:
                x0 = int(left + (onset - start) / 6 * (right - left))
                cv2.line(frame, (x0, 606), (x0, 619), (55, 215, 255), 2)
            playhead = int(left + 5.7 / 6 * (right - left))
            cv2.line(frame, (playhead, top), (playhead, bottom), (223, 226, 232), 1)
            cv2.line(frame, (playhead, 490), (playhead, 619), (223, 226, 232), 1)
            _text(frame, f'{now:05.1f} / {duration:.1f} s', (1040, 65), .55)
            if capture is not None:
                ok, robot_frame = capture.read()
                if ok:
                    last_robot = robot_frame
            if last_robot is not None:
                frame[205:497, 865:1255] = cv2.resize(last_robot, (390, 292))
            else:
                _text(frame, 'robot not rendered yet', (898, 341), .57)
                _text(frame, 'Saved joint motion: sleep_robot.npz', (889, 371), .43)
            paused = any(a0 + 3 <= now < b0 for a0, b0 in analysis['pauses'])
            _text(frame, 'SCRIPTED ALARM' if paused else 'SCRIPTED SLEEP', (889, 552), .68,
                  (80, 145, 255) if paused else (179, 215, 190))
            _text(frame, 'Torso + arms, standing G1', (889, 584), .48)
            _text(frame, 'Offline estimate, simulated sensors', (889, 611), .43)
            process.stdin.write(frame.tobytes())
    finally:
        if capture is not None:
            capture.release()
        _finish_video(process)


def run_ears(audio_path, out_dir=None, *, demo_pause=None):
    audio_path = Path(audio_path).resolve()
    directory = Path(out_dir) if out_dir is not None else ROOT / 'episodes' / audio_path.stem
    if demo_pause:
        start, seconds = demo_pause
        if not np.isfinite([start, seconds]).all() or start < 0 or seconds <= 0:
            raise ValueError('demo pause requires nonnegative START and positive SECONDS')
        if not directory.name.endswith('__demo_pause'):
            directory = directory.with_name(directory.name + '__demo_pause')
    directory.mkdir(parents=True, exist_ok=True)
    audio = decode_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    if demo_pause:
        start, seconds = demo_pause
        if start >= duration:
            raise ValueError('demo pause starts beyond the recording')
        audio[int(start * SAMPLE_RATE):int((start + seconds) * SAMPLE_RATE)] = 0
    events, theta = cochlea_events(audio)
    analysis, envelope = breathing_analysis(audio)
    analysis.update(events_per_s=round(len(events['ts']) / duration, 2), n_channels=64,
                    theta=theta, simulated=True, method=METHOD, duration_s=duration)
    if demo_pause:
        analysis['demo_pause'] = dict(start_s=demo_pause[0], duration_s=demo_pause[1],
                                      label='demo: pause inserted by us')
    write_events(directory / 'audio_events.h5', events, theta)
    if demo_pause:
        with h5py.File(directory / 'audio_events.h5', 'a') as file:
            file['events'].attrs['demo_label'] = 'demo: pause inserted by us'
    (directory / 'breathing.json').write_text(json.dumps(analysis, indent=2) + '\n')
    from vlma.robot import load_model
    model = load_model()
    qpos, alarm = robot_trajectory(envelope, analysis, model)
    npz_path = directory / 'sleep_robot.npz'
    np.savez_compressed(npz_path, qpos=qpos, fps=FPS, alarm=alarm,
                        simulated=True, scripted=True, demo_pause=bool(demo_pause))
    command = [sys.executable, '-m', 'vlma.ears', '--render-robot', str(npz_path.resolve())]
    rendered = subprocess.run(command, capture_output=True, text=True)
    robot_video = npz_path.with_suffix('.mp4') if rendered.returncode == 0 else None
    result = dict(analysis, out_dir=str(directory.resolve()),
                  ears_video=str((directory / 'ears.mp4').resolve()),
                  robot_video=str(robot_video.resolve()) if robot_video else None,
                  robot_npz=str(npz_path.resolve()), rendering_error=None)
    if rendered.returncode:
        result['rendering_error'] = (rendered.stderr.strip().splitlines()[-1]
                                     if rendered.stderr.strip() else f'process exit {rendered.returncode}')
        result['render_outside_command'] = shlex.join(command)
        print('Robot render failed: ' + result['rendering_error'], file=sys.stderr, flush=True)
        print('Render outside sandbox: ' + result['render_outside_command'], file=sys.stderr, flush=True)
    ears_video(directory / 'ears.mp4', audio_path, events, envelope, analysis, duration,
               robot_video=robot_video, demo_pause=demo_pause)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path, nargs='?')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--demo-pause', metavar='START,SECONDS')
    parser.add_argument('--render-robot', type=Path, help='Render saved scripted robot qpos locally')
    args = parser.parse_args()
    if args.render_robot:
        print(render_saved_robot(args.render_robot))
        return
    if args.audio is None:
        parser.error('audio is required')
    demo = None
    if args.demo_pause:
        try:
            demo = tuple(map(float, args.demo_pause.split(',')))
            if len(demo) != 2:
                raise ValueError
        except ValueError:
            parser.error('--demo-pause must be START,SECONDS')
    run_ears(args.audio, args.out, demo_pause=demo)


if __name__ == '__main__':
    main()

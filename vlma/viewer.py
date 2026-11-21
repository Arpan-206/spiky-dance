"""Render honestly labelled, time-aligned episode streams into a pitch panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import warnings

import cv2
import h5py
import numpy as np

WIDTH, HEIGHT = 1280, 720
SCORE_BIN_SECONDS = 0.033
COLORS = [(70, 205, 255), (255, 160, 80), (130, 230, 130), (210, 130, 240)]


def _text(image, text, xy, scale=0.45, color=(210, 215, 220)):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class _Stream:
    def __init__(self, path, fps, count, warn):
        self.cap = cv2.VideoCapture(str(path)) if path.exists() else None
        self.last = None
        self.ended = False
        self.warned_short = False
        self.warn = warn
        self.path = path
        if self.cap is not None and self.cap.isOpened():
            actual = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if actual != count:
                warn(f'{path.name}: {actual} frames, expected {count}; holding last frame after EOF')
                self.warned_short = actual < count
            if abs(actual_fps - fps) > 0.01:
                warn(f'{path.name}: fps {actual_fps:g}, expected {fps:g}; matching by frame index')

    def read(self):
        if self.cap is not None and not self.ended:
            ok, frame = self.cap.read()
            if ok:
                self.last = frame
            else:
                self.ended = True
                if self.last is None:
                    self.warn(f'{self.path.name}: no readable frames')
                elif not self.warned_short:
                    self.warn(f'{self.path.name}: reached EOF early; holding last frame')
        return self.last

    def close(self):
        if self.cap is not None:
            self.cap.release()


class _Events:
    def __init__(self, path):
        with h5py.File(path, 'r') as file:
            containers = [file]
            file.visititems(lambda name, obj: containers.append(obj) if isinstance(obj, h5py.Group) else None)
            group = next((g for g in containers if all(k in g for k in ('xs', 'ys', 'ts', 'ps'))), None)
            if group is None:
                raise ValueError('expected xs, ys, ts, ps datasets at root or in a group')
            xs, ys, ts, ps = (np.asarray(group[k]) for k in ('xs', 'ys', 'ts', 'ps'))
            if not (xs.ndim == ys.ndim == ts.ndim == ps.ndim == 1 and len(xs) == len(ys) == len(ts) == len(ps)):
                raise ValueError('event datasets must be equal-length vectors')
            shape = group.attrs.get('sensor_shape', file.attrs.get('sensor_shape'))
            if shape is None:
                raise ValueError('missing sensor_shape attribute')
            height, width = map(int, shape)
            if height <= 0 or width <= 0:
                raise ValueError('sensor_shape must be positive')
        order = np.argsort(ts, kind='stable')
        self.xs, self.ys, self.ts, self.ps = (a[order] for a in (xs, ys, ts, ps))
        self.latest = np.full((2, height, width), -np.inf, dtype=np.float64)
        self.cursor = 0

    def frame(self, seconds):
        now = seconds * 1_000_000
        end = int(np.searchsorted(self.ts, now, side='right'))
        sl = slice(self.cursor, end)
        x, y, t, p = self.xs[sl], self.ys[sl], self.ts[sl], self.ps[sl]
        valid = (x >= 0) & (x < self.latest.shape[2]) & (y >= 0) & (y < self.latest.shape[1])
        np.maximum.at(self.latest, ((p[valid] > 0).astype(int), y[valid], x[valid]), t[valid])
        self.cursor = end
        strength = np.exp((self.latest - now) / 45_000.0)
        # OpenCV uses BGR: OFF blue, ON warm yellow.
        colors = np.array([[255, 120, 45], [45, 210, 255]], dtype=float)
        pixels = 5 + np.einsum('phw,pc->hwc', strength, colors)
        return np.clip(pixels, 0, 255).astype(np.uint8)


def _tile(canvas, rect, frame, label):
    x, y, w, h = rect
    canvas[y:y+h, x:x+w] = (12, 14, 17)
    if frame is None:
        _text(canvas, 'not available yet', (x + 20, y + h // 2), 0.6, (115, 120, 130))
    else:
        fh, fw = frame.shape[:2]
        ratio = min(w / fw, h / fh)
        rw, rh = max(1, round(fw * ratio)), max(1, round(fh * ratio))
        ox, oy = x + (w-rw)//2, y + (h-rh)//2
        canvas[oy:oy+rh, ox:ox+rw] = cv2.resize(frame, (rw, rh))
    label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.rectangle(canvas, (x+7, y+5), (x+label_w+18, y+28), (12, 14, 17), -1)
    _text(canvas, label, (x+12, y+21))


def _scores(canvas, scores, seconds, duration, threshold=0.9, scorer=None, sync=None):
    _text(canvas, f'unusualness ({scorer} scorer)' if scorer else 'unusualness (network score)', (14, 628))
    if scores is None:
        _text(canvas, 'not available yet', (460, 666), 0.6, (115, 120, 130))
        return
    x0, x1, top, bottom = 42, 1040, 643, 706
    _text(canvas, '1', (20, top+4), 0.35)
    _text(canvas, '0', (20, bottom), 0.35)
    threshold_y = bottom - round(float(np.clip(threshold, 0, 1))*(bottom-top))
    cv2.line(canvas, (x0, threshold_y), (x1, threshold_y), (55, 55, 55))
    cv2.line(canvas, (x0, bottom), (x1, bottom), (65, 65, 65))
    n, dancers = scores.shape
    times = np.arange(n) * SCORE_BIN_SECONDS
    visible = times <= duration
    xs = x0 + np.minimum(times[visible] / max(duration, 0.001), 1) * (x1-x0)
    current = min(int(seconds / SCORE_BIN_SECONDS), n-1)
    row_height = min(19, 78 / max(dancers, 1))
    for dancer in range(dancers):
        color = COLORS[dancer % len(COLORS)]
        values = scores[visible, dancer]
        ys = bottom - np.clip(values, 0, 1) * (bottom-top)
        valid = np.isfinite(values)
        # Break at missing values instead of connecting across missing scores.
        starts = np.flatnonzero(valid & np.r_[True, ~valid[:-1]])
        ends = np.flatnonzero(valid & np.r_[~valid[1:], True])
        for start, end in zip(starts, ends):
            points = np.column_stack((xs[start:end+1], ys[start:end+1])).astype(np.int32)
            if len(points) > 1:
                cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
            elif len(points):
                cv2.circle(canvas, tuple(points[0]), 1, color, -1)
        value = scores[current, dancer]
        value_text = f'{value:.2f}' if np.isfinite(value) else 'n/a'
        if sync is not None and dancer < sync.shape[1]:
            value_text += f'  sync {sync[min(current, len(sync)-1), dancer]:.2f}'
        _text(canvas, f'D{dancer}: {value_text}', (1060, round(645+dancer*row_height)), min(0.43, row_height/28), color)
    play_x = round(x0 + min(seconds/max(duration, 0.001), 1)*(x1-x0))
    cv2.line(canvas, (play_x, top-3), (play_x, bottom), (235, 235, 235), 1)
    _text(canvas, f'{seconds:.1f}s / {duration:.1f}s', (880, 628), 0.4)


def run_viewer(episode_dir, out=None) -> dict:
    """Write a 1280x720 H.264 panel; missing streams stay explicitly unavailable."""
    episode = Path(episode_dir)
    output = Path(out) if out is not None else episode / 'panel.mp4'
    protected = [episode / name for name in ('clip.mp4', 'skeleton.mp4', 'robot.mp4', 'events.h5', 'scores.npy', 'meta.json')]
    if output.resolve() in [p.resolve() for p in protected]:
        raise ValueError('output must not overwrite episode inputs')
    clip = episode / 'clip.mp4'
    source = cv2.VideoCapture(str(clip))
    try:
        if not source.isOpened():
            raise FileNotFoundError(f'cannot open source video: {clip}')
        fps = source.get(cv2.CAP_PROP_FPS)
        count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        source.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f'invalid source fps or frame count: {fps}, {count}')
    ffmpeg = shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
    messages = []

    def warn(message):
        messages.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    events = None
    if (episode / 'events.h5').exists():
        try:
            events = _Events(episode / 'events.h5')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            warn(f'events.h5 unavailable: {exc}')
    scores = None
    if (episode / 'scores.npy').exists():
        try:
            scores = np.load(episode / 'scores.npy', allow_pickle=False)
            if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
                raise ValueError('expected nonempty (bins, dancers) array')
            scores = scores.astype(float)
            if len(scores)*SCORE_BIN_SECONDS < (count-1)/fps:
                warn('scores.npy ends before video; holding last score')
        except (OSError, ValueError, TypeError) as exc:
            scores = None
            warn(f'scores.npy unavailable: {exc}')
    threshold, scorer, sync = 0.9, None, None
    if (episode / 'scores_meta.json').exists():
        try:
            meta = json.loads((episode / 'scores_meta.json').read_text())
            threshold, scorer = float(meta.get('threshold', 0.9)), meta.get('scorer')
        except (OSError, ValueError, TypeError) as exc:
            warn(f'scores_meta.json unavailable: {exc}')
    if (episode / 'sync.npy').exists():
        try:
            sync = np.load(episode / 'sync.npy', allow_pickle=False).astype(float)
            if sync.ndim != 2 or not sync.shape[0]:
                raise ValueError('expected nonempty (bins, dancers) array')
        except (OSError, ValueError, TypeError) as exc:
            sync = None
            warn(f'sync.npy unavailable: {exc}')
    streams = []
    process = None
    try:
        streams = [_Stream(episode / name, fps, count, warn) for name in ('clip.mp4', 'skeleton.mp4', 'robot.mp4')]
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for index in range(count):
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            seconds = index / fps
            phone, pose, robot = [stream.read() for stream in streams]
            event_frame = events.frame(seconds) if events is not None else None
            _tile(canvas, (6, 6, 631, 296), phone, 'phone video (real)')
            _tile(canvas, (643, 6, 631, 296), event_frame, 'event camera (simulated from video)')
            _tile(canvas, (6, 308, 631, 296), pose, 'pose (estimated)')
            _tile(canvas, (643, 308, 631, 296), robot, 'robot (replay of pose)')
            _scores(canvas, scores, seconds, count/fps, threshold, scorer, sync)
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors='replace')
        if process.wait() != 0:
            raise RuntimeError(f'ffmpeg failed: {error}')
    except BrokenPipeError as exc:
        error = process.stderr.read().decode(errors='replace') if process else ''
        raise RuntimeError(f'ffmpeg stopped accepting frames: {error}') from exc
    finally:
        for stream in streams:
            stream.close()
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stderr.close()
    return {'out': str(output), 'width': WIDTH, 'height': HEIGHT, 'fps': fps, 'n_frames': count, 'warnings': messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode_dir', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    print(json.dumps(run_viewer(args.episode_dir, args.out), indent=2))


if __name__ == '__main__':
    main()

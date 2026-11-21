"""Reproducible optical steam and pose robustness sweeps."""
from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
import shlex
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np


class Steam:
    """Beer-Lambert veil, with rising multiscale density and mild scattering blur.

    Levels map linearly to beta in [0, 5]. Airlight is specified in RGB.
    """

    def __init__(self, width, height, level, seed=0, airlight=(235, 233, 228)):
        if not 0 <= level <= 99:
            raise ValueError("level must be in [0, 99]")
        self.width, self.height = width, height
        self.beta = 5.0 * level / 100.0
        self.airlight = np.asarray(airlight[::-1], dtype=np.float32)
        rng = np.random.default_rng(seed)
        self.fields = [rng.random((n, n), dtype=np.float32) for n in (8, 16, 32)]
        y, x = np.mgrid[:height, :width].astype(np.float32)
        self.x, self.y = x / width, y / height

    def apply(self, frame, seconds=0.0):
        if self.beta == 0:
            return frame.copy(), 1.0
        density = np.zeros((self.height, self.width), np.float32)
        for i, field in enumerate(self.fields):
            shift = seconds * (0.35 + i * 0.17)
            whole = int(np.floor(shift))
            fraction = shift - whole
            moving = ((1 - fraction) * np.roll(field, -whole, axis=0)
                      + fraction * np.roll(field, -whole - 1, axis=0))
            density += cv2.resize(moving, (self.width, self.height),
                                  interpolation=cv2.INTER_CUBIC) / (2 ** i)
        density = 0.35 + np.maximum(density, 0) * 0.85
        # A broad puff rises, swells and fades on a nine-second cycle.
        phase = (seconds % 9) / 9
        radius = 0.12 + 0.25 * phase
        puff = np.exp(-((self.x - 0.5 - 0.15 * np.sin(seconds * 0.3)) ** 2
                        + (self.y - (1.2 - 1.5 * phase)) ** 2) / (2 * radius ** 2))
        density += 0.8 * np.sin(np.pi * phase) ** 2 * puff
        transmittance = np.exp(-self.beta * density)[..., None]
        source = frame.astype(np.float32)
        blurred = cv2.GaussianBlur(source, (0, 0), sigmaX=0.3 + self.beta * 0.35)
        source = source * transmittance + blurred * (1 - transmittance)
        out = transmittance * source + (1 - transmittance) * self.airlight
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), float(transmittance.mean())


def make(clip_path, out_dir, level, max_frames=None, seed=0,
         airlight=(235, 233, 228)):
    """Write out_dir/clip.mp4, preserving decoded pixels exactly at level zero."""
    source = Path(clip_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = (out_dir / 'clip.mp4').resolve()
    if target == source:
        raise ValueError("output clip must differ from input clip")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0 or fps <= 0:
        cap.release()
        raise ValueError("video has invalid size or fps")
    steam = Steam(width, height, level, seed, airlight)
    # Lossless RGB encoding makes the level-zero decoded frames identical.
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
           '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264rgb',
           '-crf', '0', '-preset', 'ultrafast', '-threads', '1', str(target)]
    count, total = 0, 0.0
    try:
        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as encoder:
            try:
                while max_frames is None or count < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    output, transmittance = steam.apply(frame, count / fps)
                    encoder.stdin.write(output.tobytes())
                    total += transmittance
                    count += 1
            finally:
                encoder.stdin.close()
            error = encoder.stderr.read().decode()
            if encoder.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {error}")
    finally:
        cap.release()
    if not count:
        raise ValueError("video contains no decoded frames")
    return dict(clip=str(target), frames=count, width=width, height=height,
                fps=fps, mean_transmittance=total / count)


def keypoint_drift(clean_path, steam_path, width, height):
    """Per-dancer pixel drift for run_pose's (frames, dancers, joints, x/y/confidence)."""
    if not Path(clean_path).exists() or not Path(steam_path).exists():
        return []
    clean, veiled = np.load(clean_path), np.load(steam_path)
    if clean.shape != veiled.shape or clean.ndim != 4 or clean.shape[-1] != 3:
        print(f"Drift skipped: incompatible pose arrays {clean.shape}, {veiled.shape}")
        return []
    weight = np.minimum(clean[..., 2], veiled[..., 2])
    valid = (np.isfinite(clean[..., :2]).all(-1)
             & np.isfinite(veiled[..., :2]).all(-1)
             & np.isfinite(weight) & (weight > 0))
    distance = np.linalg.norm(clean[..., :2] - veiled[..., :2], axis=-1)
    results = []
    for dancer in range(clean.shape[1]):
        mask = valid[:, dancer]
        weights = weight[:, dancer][mask]
        results.append(float(np.average(distance[:, dancer][mask], weights=weights))
                       if weights.size else None)
    return results


def _tracking_columns(result):
    """Convert run_pose's per-dancer summary into CSV columns."""
    columns = {}
    for name in ('detected_fraction', 'mean_confidence'):
        for i, value in enumerate(result.get(name, [])):
            if value is not None:
                columns[f'dancer_{i}_{name}'] = value
    return columns


def sweep(clip_path, levels, max_frames=None, score_cmd=None, seed=0,
          airlight=(235, 233, 228)):
    clip_path = Path(clip_path).resolve()
    clip_id = clip_path.parent.name if clip_path.name == 'clip.mp4' else clip_path.stem
    episodes = Path(__file__).resolve().parents[1] / 'episodes'
    clean_dir = episodes / clip_id
    clean_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_pose = importlib.import_module('vlma.pose').run_pose
    except ImportError as exc:
        run_pose = None
        print(f'Tracking metrics skipped: vlma.pose is not importable ({exc})')
    clean_result = None
    with tempfile.TemporaryDirectory(prefix='vlma-steam-clean-') as staging:
        baseline = make(clip_path, staging, 0, max_frames, seed, airlight)
        if run_pose:
            clean_result = run_pose(baseline['clip'], str(clean_dir))
        Path(baseline['clip']).replace(clean_dir / 'clip.mp4')
    rows = []
    for level in levels:
        out_dir = episodes / f'{clip_id}__steam{level:02d}'
        # run_pose requires its source to be outside its output clip path.
        # Keep the lossless optical clip after pose writes its own episode copy.
        with tempfile.TemporaryDirectory(prefix='vlma-steam-') as staging:
            optical = make(clip_path, staging, level, max_frames, seed, airlight)
            if run_pose:
                result = run_pose(optical['clip'], str(out_dir),
                                  dancers=clean_result['dancers'])
            out_dir.mkdir(parents=True, exist_ok=True)
            Path(optical['clip']).replace(out_dir / 'clip.mp4')
        optical['clip'] = str(out_dir / 'clip.mp4')
        row = dict(level=level, mean_transmittance=optical['mean_transmittance'],
                   unusualness_mean=None)
        if run_pose:
            row.update(_tracking_columns(result))
            drift = keypoint_drift(clean_dir / 'poses.npy', out_dir / 'poses.npy',
                                   optical['width'], optical['height'])
            for i, value in enumerate(drift):
                row[f'dancer_{i}_mean_keypoint_drift_px'] = value
        if score_cmd:
            score_path = out_dir / 'scores.npy'
            if score_path.exists():
                raise FileExistsError(f'refusing to read potentially stale scores: {score_path}')
            command = [arg.format(clip=optical['clip'], out=str(out_dir))
                       for arg in shlex.split(score_cmd)]
            subprocess.run(command, check=True)
            scores = np.load(score_path)
            if (scores.ndim != 2 or not scores.size or not np.isfinite(scores).all()
                    or (scores < 0).any() or (scores > 1).any()):
                raise ValueError('scores.npy must be a finite (bins, dancers) array in [0, 1]')
            row['unusualness_mean'] = float(scores.mean())
        rows.append(row)
        print(row, flush=True)
    fields = ['level', 'mean_transmittance']
    fields += sorted({key for row in rows for key in row} - set(fields) - {'unusualness_mean'})
    fields += ['unusualness_mean']
    with (clean_dir / 'steam_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, clean_dir / 'steam_curve.png', clip_id)
    print(f"Wrote {clean_dir / 'steam_metrics.csv'} and steam_curve.png", flush=True)
    return rows


def _plot(rows, path, clip_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    levels = [row['level'] for row in rows]
    keys = sorted({key for row in rows for key in row
                   if key.endswith(('detected_fraction', 'mean_confidence'))})
    for key in keys:
        ax.plot(levels, [row.get(key, np.nan) for row in rows], marker='o',
                linestyle='--' if key.endswith('mean_confidence') else '-',
                label=key.replace('_', ' '))
    ax.set(xlabel='Simulated steam level (%)', ylabel='Tracked fraction / mean confidence',
           ylim=(-0.03, 1.03), title=f'{clip_id}: simulated steam')
    ax.grid(alpha=0.25)
    if not keys:
        ax.text(0.5, 0.5, 'Tracking metrics unavailable', transform=ax.transAxes, ha='center')
    if any(row['unusualness_mean'] is not None for row in rows):
        right = ax.twinx()
        right.plot(levels, [row['unusualness_mean'] for row in rows], 'k:s', label='Unusualness')
        right.set(ylabel='Mean unusualness', ylim=(-0.03, 1.03))
        right.legend(loc='lower right')
    if keys:
        ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('make', 'sweep'):
        p = sub.add_parser(name)
        p.add_argument('clip')
        p.add_argument('--max-frames', type=int)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--airlight', default='235,233,228', help='RGB values, comma separated')
        if name == 'make':
            p.add_argument('--level', required=True, type=int)
            p.add_argument('--out', required=True)
        else:
            p.add_argument('--levels', default='0,20,40,60,80')
            p.add_argument('--score-cmd')
    args = parser.parse_args()
    airlight = tuple(int(x) for x in args.airlight.split(','))
    if len(airlight) != 3 or any(x < 0 or x > 255 for x in airlight):
        parser.error('--airlight requires three RGB values in [0, 255]')
    if args.command == 'make':
        print(make(args.clip, args.out, args.level, args.max_frames, args.seed, airlight))
    else:
        levels = sorted(set(int(x) for x in args.levels.split(',')))
        if not levels or any(x < 0 or x > 99 for x in levels):
            parser.error('levels must be in [0, 99]')
        sweep(args.clip, levels, args.max_frames, args.score_cmd, args.seed, airlight)


if __name__ == '__main__':
    main()

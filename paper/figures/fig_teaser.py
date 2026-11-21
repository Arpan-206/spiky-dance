"""Teaser: one real cabin macarena take, four synchronized streams, six moves, and the score strip.
Data: spiky-dance/episodes/<TEASER_TAKE>/ (clip, events.h5, skeleton, robot, poses.npy, scores.npy)."""
import os, h5py, numpy as np, json
from style import *
take = os.environ.get('TEASER_TAKE', 'team_03')
ep = SEP / take
MOVES = {
    'team_03': [(42, 'arms out'), (96, 'hand to shoulder'), (120, 'hand behind head'), (138, 'both behind head'),
                (180, 'hands to hips'), (210, 'hands on hips')],
}
pick = MOVES[take]
idx = [k for k, _ in pick]
meta = json.load(open(ep / 'meta.json')); fps = float(meta['fps'])
video = read_frames(ep / 'clip.mp4', idx)
skel = read_frames(ep / 'skeleton.mp4', idx)
robot = read_frames(ep / 'robot.mp4', idx)
with h5py.File(ep / 'events.h5') as f:
    e = f['events']; xs, ys, ts, ps = e['xs'][:], e['ys'][:], e['ts'][:], e['ps'][:]
    H, W = f['events'].attrs['sensor_shape']
poses = np.load(ep / 'poses.npy')
v = poses[..., 2] > 0.3
x0, x1 = np.percentile(poses[..., 0][v], [0.5, 99.5]); y0 = np.percentile(poses[..., 1][v], 0.5)
hip = np.median(poses[:, 0, 11:13, 1][v[:, 0, 11:13]])
cx = 0.5 * (x0 + x1); top = y0 - 45; bot = hip + 0.55 * (hip - y0)
hh = bot - top; ww = hh * 1.12
box = [int(cx - ww / 2), int(top), int(cx + ww / 2), int(bot)]
def crop(im, sx=1.0, sy=1.0):
    a, b, c, d = box
    return im[int(b * sy):int(d * sy), int(a * sx):int(c * sx)]
def ev_frame(k):
    t1 = (k + 1) / fps * 1e6; t0 = t1 - 1.0e5
    m = (ts >= t0) & (ts < t1)
    on = np.zeros((H, W)); off = np.zeros((H, W))
    np.add.at(on, (ys[m & (ps == 1)], xs[m & (ps == 1)]), 1)
    np.add.at(off, (ys[m & (ps == 0)], xs[m & (ps == 0)]), 1)
    img = np.full((H, W, 3), 14, np.float32)
    c_on = np.array([242, 177, 52]); c_off = np.array([80, 145, 235])
    a_on = np.clip(on / 1.5, 0, 1)[..., None]; a_off = np.clip(off / 1.5, 0, 1)[..., None]
    img = img * (1 - a_off) + c_off * a_off
    img = img * (1 - a_on) + c_on * a_on
    return img.astype(np.uint8)
s = W / meta['width']
events = [crop(ev_frame(k), s, s) for k in idx]
video = [crop(im) for im in video]; skel = [crop(im) for im in skel]
rh, rw = robot[0].shape[:2]
RB = [int(rw * 0.5 - rh * 0.235), int(rh * 0.19), int(rw * 0.5 + rh * 0.235), int(rh * 0.61)]
robot = [im[RB[1]:RB[3], RB[0]:RB[2]] for im in robot]
rows = [(video, 'phone video', 'real'), (events, 'events', 'simulated'),
        (skel, 'pose', 'estimated'), (robot, 'Unitree G1', 'replay')]
tagc = {'real': TEAL, 'simulated': CORAL, 'estimated': AMBER, 'replay': NAVY}
n = len(idx)
fig = plt.figure(figsize=(FULL, 5.05))
gs = fig.add_gridspec(5, n, height_ratios=[1, 1, 1, 1, 0.62], wspace=0.025, hspace=0.035,
                      left=0.035, right=0.995, top=0.955, bottom=0.07)
for r, (ims, name, status) in enumerate(rows):
    for c, im in enumerate(ims):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(im, aspect='auto', interpolation='lanczos' if r != 1 else 'nearest')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)
        if r == 0:
            ax.set_title(pick[c][1], loc='center', fontsize=7.6, pad=3)
        if c == 0:
            ax.text(-0.07, 0.5, name, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7.6)
        if c == n - 1:
            ax.text(0.95, 0.05, status, transform=ax.transAxes, ha='right', va='bottom', fontsize=5.8,
                    color='white', bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.25', fc=tagc[status], ec='none', alpha=0.9))
ax = fig.add_subplot(gs[4, :])
sc = np.load(ep / 'scores.npy')[:, 0]; tb = (np.arange(len(sc)) + 0.5) * 0.033
thr = json.load(open(ep / 'scores_meta.json'))['threshold']
ax.fill_between(tb, 0, sc, color=NAVY, alpha=0.18, lw=0, step='mid')
ax.plot(tb, sc, color=NAVY, lw=0.6, drawstyle='steps-mid')
ax.axhline(thr, color=CORAL, lw=0.6, ls='--')
ax.text(0.003 * tb[-1], thr - 0.04, f'threshold: top {100 - int(round(thr * 100))}% of clean dancing', ha='left', va='top', fontsize=6, color=CORAL, bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))
for k, _ in pick:
    ax.axvline(k / fps, color=INK, lw=0.5, alpha=0.6, ls=':')
ax.set(xlim=(0, len(idx) and meta['n_frames'] / fps), ylim=(0, 1.02), yticks=[0, 0.5, 1])
ax.set_xlabel('time (s)', labelpad=1); ax.grid(False)
ax.text(-0.045, 0.5, 'score', transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7.6)
ax.yaxis.set_tick_params(labelsize=6); ax.xaxis.set_tick_params(labelsize=6)
save(fig, 'teaser')

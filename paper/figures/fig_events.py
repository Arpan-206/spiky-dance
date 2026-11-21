"""Simulated event stream statistics. Data: episodes/<episode>/events.h5 (eventify, simulated from video)."""
import h5py, numpy as np
from style import *
with h5py.File(SEP / 'team_03' / 'events.h5') as f:
    e = f['events']; xs, ys, ts, ps = e['xs'][:], e['ys'][:], e['ts'][:].astype(np.float64), e['ps'][:]
    H, W = f['events'].attrs['sensor_shape']
t = ts / 1e6
fig = plt.figure(figsize=(FULL, 3.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[1, 1], wspace=0.12, hspace=0.62)
ax = fig.add_subplot(gs[0, 0])
b = np.arange(0, t.max() + 0.0333, 0.0333)
on, _ = np.histogram(t[ps == 1], b); off, _ = np.histogram(t[ps == 0], b)
c = 0.5 * (b[1:] + b[:-1])
ax.fill_between(c, 0, on / 0.0333 / 1e3, step='mid', color=ON, alpha=0.9, lw=0, label='ON')
ax.fill_between(c, 0, -off / 0.0333 / 1e3, step='mid', color=OFF, alpha=0.9, lw=0, label='OFF')
ax.axhline(0, color=INK, lw=0.4)
ax.set(xlabel='time (s)', ylabel=r'k events s$^{-1}$ (OFF $<0$)', xlim=(0, t.max()))
ax.legend(loc='upper right', ncol=2)
tag(ax, 'a', x=-0.1, y=1.0)
ax = fig.add_subplot(gs[:, 1], projection='3d')
rng = np.random.default_rng(0); s = rng.choice(len(t), size=min(14000, len(t)), replace=False)
ax.scatter(t[s], xs[s], H - ys[s], c=np.where(ps[s] == 1, ON, OFF), s=0.3, alpha=0.55, lw=0, depthshade=False)
ax.set_xlabel('t (s)', labelpad=-6, fontsize=7); ax.set_ylabel('x (px)', labelpad=-6, fontsize=7); ax.set_zlabel('y (px)', labelpad=-7, fontsize=7)
ax.tick_params(labelsize=5.5, pad=-2)
ax.view_init(elev=16, azim=-58); ax.set_box_aspect((2.4, 1.6, 0.95), zoom=1.15)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_facecolor('white'); a.pane.set_edgecolor(MIST); a._axinfo['grid']['color'] = MIST; a._axinfo['grid']['linewidth'] = 0.4
ax.text2D(0.02, 0.93, 'b', transform=ax.transAxes, fontsize=9, fontweight='bold')
ax = fig.add_subplot(gs[1, 0])
cnt = np.zeros((H, W)); np.add.at(cnt, (ys, xs), 1)
ax.imshow(np.log1p(cnt), cmap='magma', aspect='equal', interpolation='nearest')
ax.set_xticks([0, W - 1]); ax.set_yticks([0, H - 1]); ax.grid(False)
ax.set_title('log(1 + events per pixel), whole clip', loc='left', fontsize=7)
tag(ax, 'c', x=-0.1, y=1.0)
save(fig, 'events')

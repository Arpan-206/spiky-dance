"""Simulated silicon cochlea on a real snoring recording. Data: episodes/snore/audio_events.h5, breathing.json.
The recording itself is not read or stored here; only the derived, simulated events."""
import h5py, json, numpy as np
from style import *
ep = EP / 'snore'
with h5py.File(ep / 'audio_events.h5') as f:
    e = f['events']; xs, ts, ps = e['xs'][:], e['ts'][:] / 1e6, e['ps'][:]; hz = e.attrs['channel_hz']
br = json.load(open(ep / 'breathing.json'))
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FULL, 2.45), sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1], hspace=0.08))
dt = 0.05; nb = int(np.ceil(ts.max() / dt)) + 1
img = np.zeros((len(hz), nb)); np.add.at(img, (xs, (ts / dt).astype(int)), 1)
ax.imshow(np.log1p(img), origin='lower', aspect='auto', cmap='magma', extent=(0, nb * dt, 0, len(hz)), interpolation='nearest')
yt = [0, 16, 32, 48, 63]; ax.set_yticks(yt); ax.set_yticklabels([f'{hz[i]:.0f}' for i in yt]); ax.grid(False)
ax.set_ylabel('channel centre (Hz)')
for o in br['snore_onsets_s']:
    ax.axvline(o, color='white', lw=0.5, ls=(0, (2, 2)), alpha=0.8)
tag(ax, 'a', x=-0.07, y=0.95)
rate = np.histogram(ts, bins=np.arange(0, nb * dt + dt, dt))[0] / dt / 1e3
ax2.plot(np.arange(len(rate)) * dt, rate, color=NAVY, lw=0.6)
ax2.plot(br['snore_onsets_s'], np.full(len(br['snore_onsets_s']), rate.max() * 1.08), 'v', color=CORAL, ms=3.5)
ax2.set(xlabel='time (s)', ylabel=r'k ev s$^{-1}$', xlim=(0, br['duration_s']), ylim=(0, rate.max() * 1.25))
tag(ax2, 'b', x=-0.07, y=0.85)
save(fig, 'snore')

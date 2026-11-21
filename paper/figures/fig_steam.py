"""Steam sweep on the real macarena clips: tracking, confidence, transmittance and keypoint drift
against simulated steam level. Data: spiky-dance/episodes/{macarena_solo,team_03}/steam_metrics.csv."""
import csv, numpy as np
from style import *
def load(name):
    p = SEP / name / 'steam_metrics.csv'
    if not p.exists(): return None
    rows = list(csv.DictReader(open(p)))
    g = lambda k: np.array([float(r[k]) if r.get(k) not in (None, '') else np.nan for r in rows])
    return dict(level=g('level'), T=g('mean_transmittance'), det=g('dancer_0_detected_fraction'),
                conf=g('dancer_0_mean_confidence'), drift=g('dancer_0_mean_keypoint_drift_px'))
a = load('macarena_solo'); b = load('team_03')
la, lb = 'public clip', 'cabin take 3'
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(FULL, 2.3), gridspec_kw=dict(wspace=0.42))
ax1.axvspan(25, 45, color=CORAL, alpha=0.07, lw=0)
ax1.plot(a['level'], a['det'], '-o', color=NAVY, label=f'tracked, {la}')
ax1.plot(a['level'], a['conf'], '--o', color=NAVY, mfc='white', lw=0.8, label=f'confidence, {la}')
if b is not None:
    ax1.plot(b['level'], b['det'], '-s', color=TEAL, label=f'tracked, {lb}')
    ax1.plot(b['level'], b['conf'], '--s', color=TEAL, mfc='white', lw=0.8, label=f'confidence, {lb}')
ax1.set(xlabel=r'simulated steam level $\ell$', ylabel='fraction / confidence', ylim=(-0.04, 1.08), xlim=(-3, 63))
ax1.legend(loc='lower left', fontsize=5.6, handlelength=2.0)
tag(ax1, 'a', x=-0.18, y=1.0)
ax2.plot(a['level'], a['T'], '-o', color=NAVY, label=la)
if b is not None: ax2.plot(b['level'], b['T'], 's', color=TEAL, mfc='none', label=lb)
lv = np.linspace(0, 60, 200); beta = 5 * lv / 100
m = a['level'] > 0; bm = 5 * a['level'][m] / 100
rho = float(np.exp(np.mean(np.log(-np.log(a['T'][m]) / bm))))
ax2.plot(lv, np.exp(-beta * rho), color=AMBER, lw=0.9, label=rf'$e^{{-\beta\bar\rho}}$, $\bar\rho={rho:.2f}$')
ax2.set_yscale('log'); ax2.set(xlabel=r'simulated steam level $\ell$', ylabel=r'mean transmittance $\bar T$', xlim=(-3, 63))
ax2.legend(loc='lower left', fontsize=5.6)
tag(ax2, 'b', x=-0.18, y=1.0)
ax3.plot(a['level'], a['drift'], '-o', color=NAVY, label=la)
if b is not None: ax3.plot(b['level'], b['drift'], '-s', color=TEAL, label=lb)
ax3.set(xlabel=r'simulated steam level $\ell$', ylabel='confidence-weighted drift (px)', xlim=(-3, 63))
ax3.set_yscale('symlog', linthresh=10); ax3.set_ylim(-0.5, 400)
ax3.legend(loc='upper left', fontsize=5.6)
tag(ax3, 'c', x=-0.18, y=1.0)
open(DATA / 'steam_rho.txt', 'w').write(f'{rho:.3f}\n')
save(fig, 'steam_curve')

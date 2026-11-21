"""Scorer comparison, numbers copied verbatim from spiky-dance/PROGRESS.md into data/scorer_real.csv and data/scorer_synth.csv."""
import csv, numpy as np
from style import *
def load(n): return list(csv.DictReader(open(DATA / n)))
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.2), gridspec_kw=dict(wspace=0.18, width_ratios=[3, 5.2]))
cols = [('onoff', 'spiking, on/off (default)', NAVY), ('baseline', 'Mahalanobis baseline', AMBER), ('shift', 'spiking, shift encoding', SLATE)]
for ax, name, title in zip(axes, ['scorer_real.csv', 'scorer_synth.csv'], ['real clips, 1 dancer', 'synthetic clips']):
    rows = load(name); x = np.arange(len(rows)); w = 0.26
    for k, (key, lab, col) in enumerate(cols):
        frac = [float(r[f'{key}_frac']) for r in rows]; mean = [float(r[f'{key}_mean']) for r in rows]
        ax.bar(x + (k - 1) * w, frac, w * 0.92, color=col, label=lab, lw=0)
        ax.scatter(x + (k - 1) * w, mean, marker='_', s=40, color=INK, lw=0.9, zorder=3, label='mean score' if k == 0 else None)
    ax.axhline(0.10, color=CORAL, lw=0.6, ls='--'); ax.text(len(rows) - 0.5, 0.115, 'nominal 0.10', color=CORAL, fontsize=6, ha='right')
    ax.set_xticks(x); ax.set_xticklabels([r['clip'].replace(': ', ':\n').replace(' (', '\n(').replace('held-out ', 'held-out\n').replace('sudden ', 'sudden\n') for r in rows], fontsize=6.2)
    ax.set_ylim(0, 1.05); ax.set_title(title, loc='left', fontsize=7.5)
axes[0].set_ylabel(r'fraction of bins above $p_{90}$')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.98), fontsize=6.5)
save(fig, 'scorer')

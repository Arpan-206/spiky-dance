"""Shared figure style: one palette, Times-matched serif text, thin spines, vector PDF."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
P = HERE.parents[1]
EP = P / 'episodes'
SEP = P / 'spiky-dance' / 'episodes'
import os
EPISODE = os.environ.get('VLMA_EPISODE', 'sample_person')
OUT = HERE / 'out'
DATA = HERE / 'data'
OUT.mkdir(exist_ok=True)

INK = '#1b2333'
NAVY = '#1f3a5f'
TEAL = '#2a9d8f'
AMBER = '#e9a23b'
CORAL = '#d1495b'
SLATE = '#8a94a6'
MIST = '#e8ecf2'
ON, OFF = '#f2b134', '#3b7dd8'
PALETTE = [NAVY, TEAL, AMBER, CORAL, SLATE]

COL = 3.25
FULL = 6.75

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': MIST, 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'axes.prop_cycle': matplotlib.cycler(color=PALETTE),
    'lines.linewidth': 1.2, 'lines.markersize': 3.5,
    'legend.frameon': False, 'legend.handlelength': 1.6,
    'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
    'figure.dpi': 200, 'savefig.dpi': 300,
})

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')

def tag(ax, text, x=-0.02, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right')

def read_frames(path, indices):
    import cv2
    cap = cv2.VideoCapture(str(path)); out = {}; i = 0
    want = set(indices)
    while want:
        ok, f = cap.read()
        if not ok: break
        if i in want:
            out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); want.discard(i)
        i += 1
    cap.release()
    return [out[i] for i in indices if i in out]

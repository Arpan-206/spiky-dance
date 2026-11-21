"""Steam strip: one frame of our own cabin take 3 under rising simulated steam (vlma.steam.make, seed 0).
Data: paper/figures/data/veil_team03/steamNN/clip.mp4. The public clip's frames are not shown."""
from style import *
levels = [0, 10, 20, 30, 40, 60]
k = 138
fig, axes = plt.subplots(1, len(levels), figsize=(FULL, 1.5), gridspec_kw=dict(wspace=0.03))
for ax, l in zip(axes, levels):
    im = read_frames(DATA / f'veil_team03/steam{l:02d}/clip.mp4', [k])[0]
    h, w = im.shape[:2]; im = im[int(h * 0.04):int(h * 0.86), int(w * 0.33):int(w * 0.65)]
    ax.imshow(im, aspect='auto'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(rf'$\ell={l}$', loc='center', fontsize=7.5, pad=2)
save(fig, 'steam_strip')

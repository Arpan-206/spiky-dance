"""Kinematic replay: G1 arm joint trajectories with joint limits. Data: episodes/sample_person/robot.npz."""
import numpy as np, xml.etree.ElementTree as ET
from style import *
r = np.load(SEP / 'team_03' / 'robot.npz', allow_pickle=True)
q, names, adr, fps = r['qpos'], [str(n) for n in r['joint_names']], r['joint_qposadr'], float(r['fps'])
model = P / 'assets/mujoco_menagerie/unitree_g1/g1.xml'
ranges = {}
if model.exists():
    for j in ET.parse(model).iter('joint'):
        if j.get('range'): ranges[j.get('name')] = [float(v) for v in j.get('range').split()]
t = np.arange(len(q)) / fps
parts = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow']
fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.85), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, side in zip(axes, ['left', 'right']):
    for p, col in zip(parts, PALETTE):
        n = f'{side}_{p}_joint'; i = adr[names.index(n)]
        ax.plot(t, q[:, i], color=col, label=p.replace('_', ' '))
    ax.set(xlabel='time (s)', xlim=(0, t[-1]))
    ax.set_title(f'{side} arm', loc='left', fontsize=7.5)
axes[0].set_ylabel('joint angle (rad)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
save(fig, 'robot_joints')

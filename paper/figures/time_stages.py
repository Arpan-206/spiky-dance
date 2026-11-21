"""Time each pipeline stage on the sample clip into a scratch folder; writes data/timings.csv."""
import csv, json, subprocess, sys, tempfile, time
from pathlib import Path
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P))
from vlma.pose import run_pose
from vlma.robot import run_robot
from vlma.steam import make
import os
clip = Path(os.environ.get('TIMING_CLIP', str(P / 'episodes/sample_person/clip.mp4')))
out = Path(tempfile.mkdtemp(prefix='vlma-timing-'))
rows = []
def stage(name, fn):
    t = time.perf_counter(); r = fn(); rows.append((name, round(time.perf_counter() - t, 2))); return r
ep = out / 'ep'
s = stage('pose (MediaPipe)', lambda: run_pose(clip, ep, dancers=1))
stage('robot replay (IK, no render)', lambda: run_robot(ep, render=False))
stage('events (eventify, simulated)', lambda: subprocess.run([str(P/'.venv/bin/eventify'), 'export', str(ep/'clip.mp4'), str(out/'events.h5'), '--sensor-size', '320,180'], check=True, capture_output=True))
stage('steam veil, level 40', lambda: make(clip, out/'steam', 40))
w = Path(os.environ.get('TIMING_OUT', str(Path(__file__).parent / 'data/timings.csv')))
with w.open('w', newline='') as f:
    c = csv.writer(f); c.writerow(['stage', 'seconds']); c.writerows(rows)
    c.writerow(['pose model fps', round(s['model_fps'], 1)])
print(rows, s['model_fps'])

"""One command: clip in, episode folder and combined TV video out.

python -m vlma.run <clip.mp4> [--max-frames M] [--dancers N]
    [--events-cmd "..."] [--score-cmd "..."] [--steam 0,20,40,60,80]

--events-cmd and --score-cmd run the teammate's tools with {clip} and {out}
replaced by the episode's clip.mp4 and the episode folder. Without them the
event and score tiles show "not available yet".
"""
import argparse
import shlex
import subprocess
import time
from pathlib import Path

from vlma.pose import run_pose
from vlma.robot import run_robot
from vlma.viewer import run_viewer

ROOT = Path(__file__).resolve().parents[1]


def _external(template, episode):
    command = [t.format(clip=episode / 'clip.mp4', out=episode) for t in shlex.split(template)]
    print(f'[run] {shlex.join(command)}')
    subprocess.run(command, check=True)


def run(clip, max_frames=None, dancers=None, events_cmd=None, score_cmd=None, steam=None):
    clip = Path(clip).resolve()
    episode = ROOT / 'episodes' / clip.stem
    timings = {}

    def stage(name, fn):
        started = time.perf_counter()
        result = fn()
        timings[name] = round(time.perf_counter() - started, 1)
        print(f'[run] {name} done in {timings[name]} s')
        return result

    stage('pose', lambda: run_pose(clip, episode, dancers=dancers, max_frames=max_frames))
    robot = stage('robot', lambda: run_robot(episode))
    if robot.get('rendering_error'):
        print(f"[run] robot video not rendered: {robot['rendering_error']} (run outside the sandbox)")
    if events_cmd:
        stage('events', lambda: _external(events_cmd, episode))
    if score_cmd:
        stage('score', lambda: _external(score_cmd, episode))
    stage('viewer', lambda: run_viewer(episode))
    if steam:
        from vlma.steam import sweep
        levels = [int(v) for v in steam.split(',')]
        stage('steam', lambda: sweep(episode / 'clip.mp4', levels, max_frames=max_frames, score_cmd=score_cmd))
    print(f'[run] episode: {episode}')
    print(f'[run] timings: {timings}')
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('clip')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--dancers', type=int, choices=range(1, 4))
    parser.add_argument('--events-cmd')
    parser.add_argument('--score-cmd')
    parser.add_argument('--steam', help='comma separated steam levels, e.g. 0,20,40,60,80')
    args = parser.parse_args()
    run(args.clip, args.max_frames, args.dancers, args.events_cmd, args.score_cmd, args.steam)


if __name__ == '__main__':
    main()

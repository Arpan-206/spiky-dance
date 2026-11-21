# spiky-dance

This is the event and scoring half of the hackathon project.

- Input: a video clip of 1 to 3 people who dance the macarena.
- Output 1: a simulated event-camera stream. An event camera reports only brightness changes, pixel by pixel.
- Output 2: an "unusualness" score over time for each dancer.

A spiking neural network gives the score. A spiking neural network is a network of neurons that send short pulses (spikes). The network learns only from clean macarena clips. A score near 1 means "this motion is unusual for the macarena".

## Where the parts come from

| Part | Source |
|---|---|
| Video to events | [eventify-dvs](https://github.com/Arpan-206/eventify-dvs), my package. Used as a dependency. |
| Network, training, calibration | [Mr. Spiky](https://github.com/Arpan-206/mr-spiky), my earlier project. Copied into `dance/spiky/`. |
| Features, scoring commands, tests | New in this repo. |

The code in `dance/spiky/` comes from Mr. Spiky. Mr. Spiky scores lines of code. This repo replaces the feature source: the features now come from events.

Kept from Mr. Spiky: the layer sizes (128 and 32 neurons), the learning rule (STDP) and its settings, the ZCA whitening, the per-neuron baselines, the percentile calibration and the p90 threshold.

Changed, as a design change: an adaptive threshold for each neuron, top-k inhibition in each layer, weight normalisation after each learning update, and the on/off input encoding. The network has 52 inputs: two for each of the 26 features. Each copied file has a note that lists the changes.

No file of the old code-review model is loaded. Training fits everything again. There is no mock scoring mode: if the trained files are missing, scoring stops with an error.

## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Two scorers

Both scorers use the same features and give scores with the same meaning.

| Scorer | Flag | Method |
|---|---|---|
| Spiking network (default) | `--scorer spiking` | The network from Mr. Spiky, with changes, and the on/off input encoding |
| Baseline | `--scorer baseline` | A statistical distance (Mahalanobis distance) to the clean clips. No network. |

### Comparison

Each cell is "mean score / fraction of bins above p90". Training used 5 clean real clips of 1 dancer. `PROGRESS.md` has more tables.

| Real clip (1 dancer) | Spiking, on/off encoding (default) | Baseline | Spiking, Mr. Spiky encoding |
|---|---|---|---|
| Held-out normal | 0.468 / 0.074 | 0.455 / 0.082 | 0.515 / 0.170 |
| Abnormal: freeze for the last 8 s | 0.670 / 0.406 | 0.647 / 0.406 | 0.470 / 0.135 |
| Abnormal: jerky, a jump every 8 frames | 0.991 / 0.992 | 0.976 / 0.911 | 0.288 / 0.041 |

| Synthetic clip | Spiking, on/off encoding (default) | Baseline | Spiking, Mr. Spiky encoding |
|---|---|---|---|
| Held-out normal | 0.489 / 0.095 | 0.462 / 0.103 | 0.513 / 0.112 |
| Sudden jumps | 0.800 / 0.529 | 0.934 / 0.880 | 0.437 / 0.070 |
| Freeze | 0.806 / 0.599 | 0.872 / 0.789 | 0.409 / 0.116 |

What the tables show:

- The spiking scorer with the on/off encoding and the baseline are about equal. The spiking scorer is a little better on the real clips. The baseline is better on the synthetic clips.
- **The weights of the network do not specialise.** After training, all weights are almost equal: their standard deviation is about 0.02 around 0.25. So each neuron computes almost the same sum of its inputs. The score is close to a sum of absolute deviations with a leaky memory. That is the same kind of signal as the baseline. The learning rule (STDP) adds little.

### The two input encodings of the spiking scorer

| Encoding | Flag at training | State |
|---|---|---|
| On/off | `--encoding onoff` | Default. Each whitened value `y` gives two inputs, `relu(y)` and `relu(-y)`. A deviation in either direction raises an input. |
| Mr. Spiky | `--encoding shift` | Opt-in. Known failures, see below. |

**Known failures of the Mr. Spiky encoding.** 4 tests are marked as known failures (`xfail`): the jump, freeze and three-dancer end-to-end tests, and the separation test in `tests/test_train.py`.

The cause: Mr. Spiky adds 0.5 to each whitened value and clamps it to 0..1. So each input is a constant 0.5 plus a small signal, and all weights are positive. A value below the mean lowers the input current. The score counts only activity above the normal level. So this encoding does not detect a freeze, and it does not detect jerky motion.

## Commands

Train on a folder of clean clips. This is one command for both scorers. All clips in one folder must have the same number of dancers.

```bash
uv run python -m dance.train CLIPS_DIR --out models/dance --dancers 1
```

The command holds out one clip, fits on the others, and prints the scores of the held-out clip. A normal clip must give a mean near 0.4 to 0.5 and a fraction near 0.10. Options:

| Option | Effect |
|---|---|
| `--scorer baseline` | Fit only the baseline. This is faster. `--scorer spiking` fits only the network. |
| `--holdout NAME` | Hold out this clip. NAME is the file name with its suffix. The default is the last clip by name. |
| `--no-holdout` | Train on all clips. No check is printed. |
| `--max-seconds S` | Use only the first S seconds of each clip. |
| `--encoding shift` | Spiking scorer: use the input encoding of Mr. Spiky. The default is `onoff`. |

Score one clip. This is the one-command path. All output goes in the episode folder `episodes/<clip name>/`.

```bash
uv run python -m dance.run clip.mp4 --episodes-dir episodes --dancers 1
uv run python -m dance.run episodes/clip                        # the target is an episode folder
uv run python -m dance.run clip.mp4 --scorer baseline           # the other scorer
```

Write a preview video of a scored episode. This is a fallback view: the events (ON yellow, OFF blue), the box of each dancer, and the score and sync traces with the threshold line.

```bash
uv run python -m dance.run clip.mp4 --preview                   # scores, then episodes/clip/preview.mp4
uv run python -m dance.preview episodes/clip                    # for an episode that has scores
uv run python -m dance.score --live --dancers 2 --preview       # the same view in a window, q stops
```

Score only, without the events file. This also writes `scores.json`, and `sync.npy` for 2 or more dancers. The source is a clip or an eventify `.h5` file.

```bash
uv run python -m dance.score clip.mp4 --out scores.npy --dancers 1
```

Score live from the webcam. The command prints a rolling score for each dancer about 10 times a second.

```bash
uv run python -m dance.score --live --dancers 2
uv run python -m dance.score --live --dancers 2 --json    # one JSON object per line
```

Features only.

```bash
uv run python -m dance.features clip.mp4 features.npy --dancers 1
```

Events only. This is the eventify command with this project's settings.

```bash
uv run eventify export clip.mp4 events.h5 --interp 3 --sensor-size 320,180
```

## Episode folders

One clip has one folder: `episodes/<clip name>/`. The clip name is the file name without its suffix. The pose build and this build share the folder.

| File | Writer | Content |
|---|---|---|
| `poses.npy` | Pose build | `(frames, dancers, 17, 3)`, pixels of the original video, leftmost dancer = 0, missing joints have confidence 0 |
| `meta.json` | Pose build | Frame rate, width, height and frame count of the original video |
| `pose3d.npz` | Pose build | Not used here |
| `events.h5` | This build | Events, in sensor pixels (at most 320 wide) |
| `scores.npy` | This build | Scores, `(bins, dancers)` |
| `sync.npy` | This build | Sync scores, `(bins, dancers)`. Only for 2 or more dancers. |
| `scores_meta.json` | This build | `bin_ms`, `bins`, `sensor_width`, `sensor_height`, `scorer`, `threshold`, `sync`, and the ranking |

This build only reads `poses.npy` and `meta.json`. It never writes them.

Events are in sensor pixels. Poses are in pixels of the original video. So the pose boxes are scaled with the width and height from `meta.json` before the crop.

If `poses.npy` is absent, the run uses vertical strips (`--dancers N`). The run prints which of the two it used.

If the target folder has no video but has `events.h5`, the scores come from that file.

## Data formats

| Data | Format |
|---|---|
| Events | eventify `EVENT_DTYPE`: `x` int16, `y` int16, `t` int64 microseconds, `p` int8 (0 = OFF, 1 = ON) |
| Events file | eventify HDF5 export: `/events/xs`, `ys`, `ts`, `ps`, attribute `sensor_shape` = (height, width) |
| Binned events | float32 `(n_bins, 2, H, W)`, the layout of eventify `frame_stream`. Channel 0 = OFF, channel 1 = ON. |
| Features | float32 `(bins, dancers, 26)`: 13 features, then their 13 bin-to-bin deltas |
| Scores | float32 `(bins, dancers)`, values 0 to 1 |
| Sync scores | float32 `(bins, dancers)`, values 0 to 2 |
| Poses (optional input) | `(frames, dancers, 17, 3)`, COCO keypoint order, columns x, y, confidence |

One bin is one 33 ms window. Bin `k` covers the time from `k * 33 ms` to `(k + 1) * 33 ms`, counted from the first frame.

Settings: sensor width at most 320 pixels, `interp=3` (3 interpolated sub-frames between real frames), threshold 0.05 (the eventify default).

## The features

There are 26 numbers for each bin and each dancer. The first 13 are the features in the table. The last 13 are their bin-to-bin deltas: the value in this bin minus the value in the bin before. The deltas of the first bin are 0.

The features are computed inside the box of each dancer. Coordinates are normalised to the box.

| Index | Name | Meaning |
|---|---|---|
| 0 | rate | Event count against the dancer's own running mean (about 10 s). 0.5 is the own mean. |
| 1 | polarity | (ON - OFF) / count, -1 to 1 |
| 2 | centroid_x | Mean x of the events, 0 to 1 |
| 3 | centroid_y | Mean y of the events, 0 to 1 |
| 4 to 12 | grid | Event counts in a 3x3 grid, sum 1 |

An empty bin gives rate 0, polarity 0, centroid 0.5 and a grid of zeros. It gives no NaN.

## Dancer boxes

- With poses: the box around the keypoints of each dancer, padded by 15% on each side.
- Without poses: equal vertical strips, one for each dancer (`--dancers N`). Dancer 0 is the left strip. Inside its strip, the box centres itself on the dancer: its centre and size follow a slow running median (about 5 s) of the event positions.

`dance.run` reads `poses.npy` and `meta.json` only from the episode folder. `dance.score` looks for `<clip name>.poses.npy` or `poses.npy` next to the clip, and for `meta.json` next to the poses. You can also give `--poses FILE` and `--pose-meta FILE`.

## Scores

A score is a percentile against the training clips. A score of 0.5 is the training median. The threshold is 0.90: a bin above it is in the top 10% of unusualness for clean clips.

For each dancer the commands report two numbers:

- the mean score,
- the fraction of bins above the threshold.

## Sync score

The sync score needs no training. It needs 2 or more dancers.

For each dancer it is 1 minus the correlation between the feature series of the dancer and the median series of the other dancers. The correlation covers a rolling window of 2 seconds.

| Value | Meaning |
|---|---|
| 0 | The dancer moves like the others |
| 1 | No relation to the others |
| 2 | The exact opposite of the others |

## Layout

| Path | Content |
|---|---|
| `dance/contracts.py` | The interfaces between the parts |
| `dance/stream.py` | Calls eventify and puts events into 33 ms windows |
| `dance/features.py` | Binned events to features |
| `dance/baseline.py` | Baseline scorer: a statistical distance, no network |
| `dance/sync.py` | Sync score between dancers, no training |
| `dance/spiky/` | Network and calibration, copied from Mr. Spiky |
| `dance/train.py` | Training of both scorers on clean clips |
| `dance/score.py` | Scoring, offline and live |
| `dance/run.py` | One command for a clip |
| `dance/preview.py` | Fallback preview video and live preview window |
| `models/dance/` | Trained files |
| `tests/` | Tests and synthetic clips |

Video files are never committed. `.gitignore` excludes `*.mp4` and `*.mov`.

See `PROGRESS.md` for what runs and what is untested.

## The other half: pose, robot, steam and the TV video (`vlma/`)

`vlma/` turns the same clip into a pose skeleton, a Unitree G1 humanoid in MuJoCo that replays the arms, a sauna steam stress test, and one combined video for the TV. It writes into the same `episodes/<clip name>/` folder as the scorer.

### What is real and what is simulated

| Stream | Status |
|---|---|
| Phone video | Real |
| Event-camera stream | Simulated from the video with eventify |
| Pose skeleton | Estimated by MediaPipe Pose Landmarker |
| Humanoid | A kinematic replay of the pose with joint limits. Not a controller, no balance. |
| Unusualness score | The spiking scorer above, on simulated events |
| Steam | A simulated veil over the image. Not fitted to real steam. |

Every tile in the combined video carries one of these labels. A missing stream shows "not available yet".

### Setup

```bash
uv sync --group vlma
```

Two downloads go in `assets/` (not committed):

- `assets/pose_landmarker_full.task`: MediaPipe Pose Landmarker (full), from the MediaPipe models page.
- `assets/mujoco_menagerie/unitree_g1/`: from github.com/google-deepmind/mujoco_menagerie.

Tests: `uv run --group vlma pytest tests/test_pose.py tests/test_robot.py tests/test_steam.py tests/test_viewer.py`. The pose and viewer tests that need `clips/sample_walk.mp4` (`one-by-one-person-detection.mp4` from github.com/intel-iot-devkit/sample-videos) skip without it.

### One command

```bash
uv run --group vlma python -m vlma.run clip.mp4 \
  --events-cmd "eventify export {clip} {out}/events.h5 --interp 3 --sensor-size 320,180" \
  --score-cmd "python -m dance.score {clip} --out {out}/scores.npy --dancers 1" \
  --steam 0,10,20,30,40,60
```

Stages in order: pose, robot, events, score, combined video, steam sweep. Output in `episodes/<clip name>/`: `panel.mp4` (the TV video), `poses.npy`, `meta.json`, `pose3d.npz`, `skeleton.mp4`, `robot.npz`, `robot.mp4`, `events.h5`, `scores.npy`, `steam_metrics.csv`, `steam_curve.png`. Steam levels go in `episodes/<clip name>__steam<NN>/`.

On macOS the robot render needs a normal user session (MuJoCo offscreen rendering uses the window server).

### Steam result so far

A 6-second test clip of one person standing (not a dance), simulated steam:

| Steam level | Frames with the person tracked | Mean unusualness |
|---|---|---|
| 0 | 100% | 0.89 |
| 20 | 100% | 0.98 |
| 40 | 32% | 0.98 |
| 60 | 0% | 0.98 |

Pose tracking breaks between levels 30 and 60. The scorer flags the steam as unusual from level 20 and stays at its ceiling. A real macarena clip will replace this table.

### Layout

| Path | Content |
|---|---|
| `vlma/pose.py` | MediaPipe video mode, up to 3 dancers, left-to-right identity, COCO-17 in pixels plus 3D world landmarks |
| `vlma/robot.py` | Arm angles from the 3D pose onto the G1, smoothed and clamped, rendered offscreen |
| `vlma/steam.py` | Drifting steam veil, sweep over levels, tracking and drift metrics, plot |
| `vlma/viewer.py` | 2x2 TV video plus score strip, reads `events.h5`, `scores.npy`, `sync.npy`, `scores_meta.json` |
| `vlma/run.py` | Runs the stages in order |

### The robot's ears (`vlma/ears.py`)

The audio twin of the event camera: a simulated silicon cochlea. A phone recording of someone snoring goes through 64 band-pass channels (50 Hz to 4 kHz, second-order Butterworth). Each channel fires ON and OFF events when its log amplitude changes by a threshold, like a DVS pixel. The events use the same `events.h5` layout, with the channel as `xs`.

From the events: breaths per minute and pauses in breathing (gaps longer than 10 s, or 2.5 times the median gap). The robot slumps asleep and rises a little with each snore; during a pause it wakes up with a scripted alarm pose.

```bash
uv run --group vlma python -m vlma.ears clips/snore.m4a
uv run --group vlma python -m vlma.ears clips/snore.m4a --demo-pause 12,16
```

On our 45 s recording: about 11,500 simulated events per second, 16.2 breaths per minute, no pause. `--demo-pause` silences a stretch on purpose to show the detector; that video says "demo: pause inserted by us" on every frame. Not a medical device. The recording itself is not in the repo.

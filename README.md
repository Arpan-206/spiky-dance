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

Changed, as a design change: an adaptive threshold for each neuron, top-k inhibition in each layer, and weight normalisation after each learning update. The network now has 26 inputs. Each copied file has a note that lists the changes.

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
| Baseline (default) | `--scorer baseline` | A statistical distance (Mahalanobis distance) to the clean clips. No network. |
| Spiking network | `--scorer spiking` | The network from Mr. Spiky, with changes |

**The baseline is the default. The spiking network does not beat it on the real-clip table.** With the input encoding of Mr. Spiky, the spiking scorer does not detect a freeze or jerky motion. With the opt-in on/off encoding (`--encoding onoff` at training), it is about equal to the baseline. `PROGRESS.md` has the measured numbers.

## Commands

Train on a folder of clean clips. This is one command for both scorers. All clips in one folder must have the same number of dancers.

```bash
uv run python -m dance.train CLIPS_DIR --out models/dance --dancers 1
```

The command holds out one clip, fits on the others, and prints the scores of the held-out clip. A normal clip must give a mean near 0.4 to 0.5 and a fraction near 0.10. Options:

| Option | Effect |
|---|---|
| `--scorer baseline` | Fit only the baseline. This is faster. |
| `--holdout NAME` | Hold out this clip. The default is the last clip by name. |
| `--no-holdout` | Train on all clips. No check is printed. |
| `--max-seconds S` | Use only the first S seconds of each clip. |
| `--encoding onoff` | Spiking scorer: use the on/off input encoding. |

Score one clip. This is the one-command path. All output goes in the episode folder `episodes/<clip name>/`.

```bash
uv run python -m dance.run clip.mp4 --episodes-dir episodes --dancers 1
uv run python -m dance.run episodes/clip                        # the target is an episode folder
uv run python -m dance.run clip.mp4 --scorer spiking            # the other scorer
```

Score only, without the events file. The source is a clip or an eventify `.h5` file.

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
| `models/dance/` | Trained files |
| `tests/` | Tests and synthetic clips |

Video files are never committed. `.gitignore` excludes `*.mp4` and `*.mov`.

See `PROGRESS.md` for what runs and what is untested.

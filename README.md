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

The code in `dance/spiky/` comes from Mr. Spiky. Mr. Spiky scores lines of code. This repo replaces only the feature source: the 13 features now come from events. The network (13 inputs, 128 neurons, 32 neurons), the learning rule and the calibration method are the same. Each copied file has a note that lists the changes.

No file of the old code-review model is loaded. Training fits everything again. There is no mock scoring mode: if the trained files are missing, scoring stops with an error.

## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Commands

Train on a folder of clean clips. All clips in one folder must have the same number of dancers.

```bash
uv run python -m dance.train CLIPS_DIR --out models/dance --dancers 1
```

Score one clip. This is the one-command path. It writes `events.h5` and `scores.npy` and prints the ranking.

```bash
uv run python -m dance.run clip.mp4 --out-dir out --dancers 1
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

## Data formats

| Data | Format |
|---|---|
| Events | eventify `EVENT_DTYPE`: `x` int16, `y` int16, `t` int64 microseconds, `p` int8 (0 = OFF, 1 = ON) |
| Events file | eventify HDF5 export: `/events/xs`, `ys`, `ts`, `ps`, attribute `sensor_shape` = (height, width) |
| Binned events | float32 `(n_bins, 2, H, W)`, the layout of eventify `frame_stream`. Channel 0 = OFF, channel 1 = ON. |
| Features | float32 `(bins, dancers, 13)` |
| Scores | float32 `(bins, dancers)`, values 0 to 1 |
| Poses (optional input) | `(frames, dancers, 17, 3)`, COCO keypoint order, columns x, y, confidence |

One bin is one 33 ms window. Bin `k` covers the time from `k * 33 ms` to `(k + 1) * 33 ms`, counted from the first frame.

Settings: sensor width at most 320 pixels, `interp=3` (3 interpolated sub-frames between real frames), threshold 0.05 (the eventify default).

## The 13 features

The features are computed inside the box of each dancer. Coordinates are normalised to the box.

| Index | Name | Meaning |
|---|---|---|
| 0 | rate | Event count, log scaled, 0 to 1 |
| 1 | polarity | (ON - OFF) / count, -1 to 1 |
| 2 | centroid_x | Mean x of the events, 0 to 1 |
| 3 | centroid_y | Mean y of the events, 0 to 1 |
| 4 to 12 | grid | Event counts in a 3x3 grid, sum 1 |

An empty bin gives rate 0, polarity 0, centroid 0.5 and a grid of zeros. It gives no NaN.

## Dancer boxes

- With poses: the box around the keypoints of each dancer, padded by 15% on each side.
- Without poses: equal vertical strips, one for each dancer (`--dancers N`).

The command looks for `<clip name>.poses.npy` or `poses.npy` next to the clip. You can also give `--poses FILE`.

Pose coordinates are read as pixels of the source video. If all coordinates are 1.5 or less, they are read as fractions of the frame.

## Scores

A score is a percentile against the training clips. A score of 0.5 is the training median. The threshold is 0.90: a bin above it is in the top 10% of unusualness for clean clips.

For each dancer the commands report two numbers:

- the mean score,
- the fraction of bins above the threshold.

## Layout

| Path | Content |
|---|---|
| `dance/contracts.py` | The interfaces between the parts |
| `dance/stream.py` | Calls eventify and puts events into 33 ms windows |
| `dance/features.py` | Binned events to 13 features |
| `dance/spiky/` | Network and calibration, copied from Mr. Spiky |
| `dance/train.py` | Training on clean clips |
| `dance/score.py` | Scoring, offline and live |
| `dance/run.py` | One command for a clip |
| `models/dance/` | Trained files |
| `tests/` | Tests and synthetic clips |

Video files are never committed. `.gitignore` excludes `*.mp4` and `*.mov`.

See `PROGRESS.md` for what runs and what is untested.

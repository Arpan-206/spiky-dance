# Progress

Last update: 15:27.

## What runs

| Part | State | Tests |
|---|---|---|
| Events through eventify-dvs | Runs | Pass |
| Features | Runs | Pass |
| Contracts | Written | Not applicable |
| Baseline scorer (`--scorer baseline`) | Runs | Pass |
| Scoring offline, live loop, one-command run | Runs | Pass |
| Synthetic fixtures | Run | Pass |
| End-to-end tests for jump, freeze, three dancers | Fail | 3 fail, they wait for the spiking model |
| Spiking model and training | In work | Not merged |

## Real-clip table

Training: 5 clean clips, 1 dancer, 25 s each. The test clips come from a held-out segment of the same video. The three abnormal clips are that segment, changed with ffmpeg.

### Baseline scorer, 13 features

| Real clip (1 dancer) | Bins | Mean score | Fraction above p90 |
|---|---|---|---|
| Held-out normal | 757 | 0.392 | 0.040 |
| Abnormal: 3x speed | 255 | 0.341 | 0.031 |
| Abnormal: freeze for the last 8 s | 756 | 0.602 | 0.362 |
| Abnormal: jerky, a jump every 8 frames | 756 | 0.898 | 0.791 |

### Baseline scorer, 26 features (13 features and their bin-to-bin deltas)

| Real clip (1 dancer) | Bins | Mean score | Fraction above p90 |
|---|---|---|---|
| Held-out normal | 757 | 0.420 | 0.074 |
| Abnormal: 3x speed | 255 | 0.468 | 0.043 |
| Abnormal: freeze for the last 8 s | 756 | 0.644 | 0.393 |
| Abnormal: jerky, a jump every 8 frames | 756 | 0.973 | 0.903 |

The baseline separates the freeze clip and the jerky clip from the normal clip. The deltas make both a little clearer. The 3x speed clip is still not separated: its fraction above p90 is below the normal clip.

### Spiking model

No numbers yet. The first port of Mr. Spiky gave the score 1.000 for every bin: every neuron fired in every bin.

## Untested

- Live mode with a real webcam. The live loop is tested only with a video file.
- Real pose files. All tests use made-up poses or vertical strips.
- Clips with 2 or 3 real dancers.

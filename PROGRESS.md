# Progress

Last update: 15:55.

## Decision on the default scorer

The default scorer is the spiking network with the on/off input encoding. This is the user's decision of 15:50. The baseline is available with `--scorer baseline`.

- The spiking scorer with the on/off encoding and the baseline are about equal. The spiking scorer is a little better on the real clips. The baseline is better on the synthetic clips.
- The weights of the network do not specialise. The learning rule adds little.
- The input encoding of Mr. Spiky (`--encoding shift`) is opt-in. It fails: it gives the freeze clip and the jerky clip a lower score than the normal clip. Its 4 tests are marked as known failures.

## What runs

| Part | State | Tests |
|---|---|---|
| Events through eventify-dvs | Runs | Pass |
| Features: 13 features and 13 deltas, self-centred boxes, relative rate | Runs | Pass |
| Spiking scorer, on/off encoding | Runs, default | Pass |
| Baseline scorer | Runs | Pass |
| Spiking scorer, encoding of Mr. Spiky | Runs, opt-in, does not separate | 4 known failures |
| Sync score for 2 or more dancers | Runs | Pass |
| Scoring offline, episode folders | Runs | Pass |
| Live mode on the real webcam | Runs, tested by hand | Loop tested with a video file |
| One-command training with a held-out check | Runs | Error paths tested |
| Full test suite | 111 pass, 4 known failures | About 75 s |

## Real-clip table

Training: 5 clean clips, 1 dancer, 25 s each. The test clips come from a held-out segment of the same video. The two abnormal clips are that segment, changed with ffmpeg. Each cell is "mean score / fraction of bins above p90".

| Real clip (1 dancer) | Bins | Baseline | Spiking, Mr. Spiky encoding | Spiking, on/off encoding |
|---|---|---|---|---|
| Held-out normal | 757 | 0.455 / 0.082 | 0.515 / 0.170 | 0.468 / 0.074 |
| Abnormal: freeze for the last 8 s | 756 | 0.647 / 0.406 | 0.470 / 0.135 | 0.670 / 0.406 |
| Abnormal: jerky, a jump every 8 frames | 756 | 0.976 / 0.911 | 0.288 / 0.041 | 0.991 / 0.992 |

The 3x speed clip is no longer in this table. No scorer separated it. Real good and bad clips will replace it.

## Synthetic end-to-end table

Training: 7 synthetic normal clips. Each cell is "mean score / fraction of bins above p90".

| Synthetic clip | Baseline | Spiking, Mr. Spiky encoding | Spiking, on/off encoding |
|---|---|---|---|
| Held-out normal | 0.462 / 0.103 | 0.513 / 0.112 | 0.489 / 0.095 |
| Sudden jumps | 0.934 / 0.880 | 0.437 / 0.070 | 0.800 / 0.529 |
| Freeze | 0.872 / 0.789 | 0.409 / 0.116 | 0.806 / 0.599 |
| Three dancers: dancer 1, jumps | 0.918 / 0.868 | 0.493 / 0.103 | 0.754 / 0.512 |
| Three dancers: dancer 0, normal | 0.475 / 0.066 | 0.510 / 0.087 | 0.485 / 0.074 |
| Three dancers: dancer 2, normal | 0.436 / 0.095 | 0.495 / 0.079 | 0.474 / 0.070 |

## Effect of the feature changes, baseline scorer, real clips

Each cell is "mean score / fraction of bins above p90".

| Real clip | 13 features | 26 features (with deltas) | 26 features, self-centred box, relative rate |
|---|---|---|---|
| Held-out normal | 0.392 / 0.040 | 0.420 / 0.074 | 0.455 / 0.082 |
| Abnormal: freeze | 0.602 / 0.362 | 0.644 / 0.393 | 0.647 / 0.406 |
| Abnormal: jerky | 0.898 / 0.791 | 0.973 / 0.903 | 0.976 / 0.911 |

## Sync score

Three synthetic dancers move in step. Only dancer 1 jumps.

| Dancer | Mean sync score |
|---|---|
| 0, normal | 0.244 |
| 1, jumps | 0.538 |
| 2, normal | 0.244 |

0 means in step, 1 means no relation. The sync score is not tested on real clips with 2 or more dancers.

## Spiking network: diagnostics

Measured on the real training features, before the design change.

| Diagnostic | Result |
|---|---|
| Neurons that fire per bin, untrained weights | Hidden layer 99.7%, output layer 100% |
| Whitened features, mean per dimension | 0.000 |
| Whitened features, standard deviation per dimension | 0.112 to 0.145 |
| Path to the input current | Whiten, add 0.5, clamp to 0..1. No absolute value, no rectification. |
| Input current of a hidden neuron | Mean 1.62 for a threshold of 1.0 |
| Weights after training, old network | First layer between 0.1 and 0.5. Second layer: all weights the same value. |

Cause of the first fault: each input is a constant 0.5 plus a small signal. So every neuron fired in every bin, and a value below the mean lowered the current.

After the design change (adaptive threshold for each neuron, top-k inhibition, weight normalisation after each update):

| Measure | Result |
|---|---|
| Neurons that fire per bin, hidden layer | 10.0% |
| Neurons that fire per bin, output layer | 9.7% to 10.0% |
| Target | 5% to 15% |
| Weights after training | Almost equal: standard deviation 0.017 to 0.019 around 0.25 |

The weights do not specialise. With the on/off encoding the score is close to a sum of absolute deviations with a leaky memory. That is the same kind of signal as the baseline.

## Speed

| Run | Time |
|---|---|
| One-command run of a 25 s real clip, baseline | 2.2 s |
| Training, both scorers, 5 real clips of 25 s | 17 to 40 s |
| Live mode | 10.1 updates per second |

## Live mode on the real webcam

Tested by hand with both scorers and the built-in camera.

| Check | Time | Result |
|---|---|---|
| Text output, 1 strip, 8 s | 15:25 | 81 updates, final table printed |
| JSON output, 2 strips, 5 s, baseline scorer, current features and sync score | 15:46 | 50 updates, 10.1 updates per second |
| JSON output, 2 strips, 8 s, spiking scorer | 15:52 | 81 updates, 10.1 updates per second |

The trained models in `models/dance/` come from one performer on a white background. So live scores in another room are high (0.75 in the test). Train again on clean clips from the demo room.

## Untested

- Real pose files from the pose build. The tests use made-up poses.
- Real clips with 2 or 3 dancers.
- Ctrl-C in live mode on the real webcam.
- Training on eventify `.h5` files through the command.

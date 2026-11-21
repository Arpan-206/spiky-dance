# Progress

Last update: 16:25.

## Decision on the default scorer

The default scorer is the spiking network with the on/off input encoding. This is the user's decision of 15:50. The baseline is available with `--scorer baseline`.

- The spiking scorer with the on/off encoding and the baseline are about equal. The spiking scorer is a little better on the real clips. The baseline is better on the synthetic clips.
- The weights of the network do not specialise. The learning rule adds little.
- The input encoding of Mr. Spiky (`--encoding shift`) is opt-in. It fails: it gives the freeze clip and the jerky clip a lower score than the normal clip. Its 4 tests are marked as known failures.

## YouTube clip set (16:25)

The list of URLs and cut times is in `data/youtube_clips.csv`. No video is in the repo. I looked at 53 videos and kept 13: most were crowds, cartoons, games, or videos with cuts.

Only clips of the classic macarena steps are "normal". Other versions (dance workout, challenge, free style) are in a separate folder and are not used for training.

| Group | Videos | Clips | Use |
|---|---|---|---|
| Classic, 1 dancer | 3 performers | 14 | 11 for training, 1 whole video (3 clips) held out |
| Classic, 2 dancers | 1 video | 2 | Training. There is no second video to hold out. |
| Classic, 3 dancers | 0 | 0 | No classic clip with three dancers was found |
| Bad dancing, 2 dancers | 3 videos | 3 | Never trained on. Each is 13 to 17 s. |
| Other versions, 1 dancer | 1 video | 5 | Scored only |
| Other versions, 2 dancers | 2 videos | 6 | Scored only |
| Other versions, 3 dancers | 3 videos | 9 | Scored only |

Each cell below is "mean score / fraction of bins above p90". The models for this table were trained with the whole held-out video left out.

### 1 dancer

| Clips | Spiking | Baseline | How much to trust it |
|---|---|---|---|
| Held-out classic, a performer the model never saw (3 clips) | 0.471 / 0.033 | 0.518 / 0.037 | Good: 2 training performers, 1 unseen performer |
| Other version, free style (5 clips) | 0.740 / 0.241 | 0.701 / 0.326 | One performer only |

A classic clip of an unseen performer scores as normal. A free-style dance scores clearly higher.

### 2 dancers

| Clips | Spiking | Baseline | How much to trust it |
|---|---|---|---|
| Bad dancing (3 clips, 6 dancers) | 0.451 / 0.133 | 0.439 / 0.089 | Low: the training has 38 s of one 2-dancer video |
| Other versions (6 clips) | 0.274 / 0.014 | 0.237 / 0.001 | Low |

Bad dancing, for each dancer:

| Clip | Dancer | Spiking | Baseline |
|---|---|---|---|
| fail_grass | 0 | 0.290 / 0.020 | 0.256 / 0.008 |
| fail_grass | 1 | 0.315 / 0.018 | 0.288 / 0.005 |
| fail_stairs | 0 | 0.455 / 0.008 | 0.499 / 0.014 |
| fail_stairs | 1 | 0.302 / 0.012 | 0.268 / 0.004 |
| fail_wall | 0 | 0.456 / 0.043 | 0.466 / 0.053 |
| fail_wall | 1 | 0.931 / 0.774 | 0.888 / 0.497 |

**The scorers do not detect bad dancing in these clips.** Only 1 of 6 dancers scores high. The other 5 score like a normal dancer or lower. The other dance versions with 2 dancers also score lower than a normal clip. The score reacts to unusual motion in the box of a dancer: a freeze, a sudden jump, an unusual place of the motion. It does not check the dance steps.

### 3 dancers

| Clips | Spiking | Baseline | Mean sync score | How much to trust it |
|---|---|---|---|---|
| Other versions (9 clips, 3 videos) | 0.388 / 0.023 | 0.397 / 0.015 | 0.77 to 0.96 | Do not trust this row |

Do not trust the 3-dancer row. No classic clip with three dancers is in the training, and none is in the test. The three videos are other dance versions, and one of them is 17 s long.

### Sync score on real clips

| Real clip | Dancers in step? | Mean sync score, each dancer |
|---|---|---|
| Three dancers in a room, dance-workout version, clip 1 | Yes | 0.896, 0.928, 0.934 |
| Same video, clip 2 | Yes | 0.952, 0.956, 0.946 |
| Same video, clip 3 | Yes | 0.866, 0.894, 0.875 |

**The sync score does not work on real clips.** Dancers in step get about 0.9, which means "no relation". On the synthetic clips the same dancers got 0.24. The features of one 33 ms bin are too noisy on real video for a correlation over 2 s.

### Live webcam score in the demo room, before and after

One strip, 6 s, the built-in camera, the same scene.

| Scorer | Before (trained on 1 performer) | After (trained on the classic YouTube set) |
|---|---|---|
| Spiking | 0.684 / 0.189 | 0.806 / 0.139 |
| Baseline | 0.515 / 0.038 | 0.739 / 0.028 |

The mean score in the room did not go down. It went up. The fraction above the threshold went down a little. The camera saw the room as it was, not a dancer.

### Training time

| Run | Time |
|---|---|
| Both scorers, 13 clips of 25 s, with a held-out video | 89 s (events and features 31 s) |

This is above the 60 s target. `--scorer baseline` or `--max-seconds` brings it below.

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
| Preview video and live preview window | Run | Pass. The live window is not tested by hand. |
| Full test suite | 117 pass, 4 known failures | 75 to 90 s |

## Real-clip table, first model (15:50)

These numbers come from the first models, trained on 5 clips of one performer. The models in `models/dance/` are newer: see the YouTube clip set above.

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
| One-command run of a 25 s real clip | 2 to 7 s |
| Training, both scorers, 5 real clips of 25 s | 17 to 41 s |
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

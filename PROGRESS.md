# Progress

Last update: 17:55.

## Decision on the default scorer

The default scorer is the spiking network with the on/off input encoding. This is the user's decision of 15:50. The baseline is available with `--scorer baseline`.

- The spiking scorer with the on/off encoding and the baseline are about equal. The spiking scorer is a little better on the real clips. The baseline is better on the synthetic clips.
- The weights of the network do not specialise. The learning rule adds little.
- The input encoding of Mr. Spiky (`--encoding shift`) is opt-in. It fails: it gives the freeze clip and the jerky clip a lower score than the normal clip. Its 4 tests are marked as known failures.

## Two clean takes from the setting of the bad takes (17:55)

2 more clean takes arrived, filmed at 17:38 in the same setting as the 9 bad takes. They answer the session doubt.

**Step 1, blind:** the models trained on the 5 clean takes of 16:30 scored the 2 new clean takes before any retraining. Each cell is "mean score / fraction of bins above p90".

| Takes | Spiking | Baseline |
|---|---|---|
| 5 clean takes of 16:30, each not seen by its model | 0.511 / 0.114 | 0.518 / 0.113 |
| 2 new clean takes, later setting, not seen | 0.645 / 0.206 | 0.706 / 0.241 |
| 9 bad takes, later setting | 0.744 / 0.422 | 0.769 / 0.423 |

So the session doubt was right in part. A clean take from the later setting scores higher than every clean take of 16:30, and higher than 3 of the 9 bad takes. Of the first gap between clean and bad (0.233 on the mean score, spiking), about 0.134 came from the setting and about 0.099 from the dancing.

**Step 2, retrain:** `models/room` now uses all 7 clean takes. Leave-one-out again: train on 6 clean takes, score the 7th and all 9 bad takes.

| Takes | Spiking | Baseline |
|---|---|---|
| 7 clean takes, each not seen by its model | 0.523 / 0.108 | 0.521 / 0.104 |
| Range over the 7 clean takes | 0.456 to 0.581 / 0.037 to 0.156 | 0.453 to 0.566 / 0.040 to 0.176 |
| The 2 clean takes of the later setting | 0.531 / 0.156 and 0.519 / 0.122 | 0.566 / 0.176 and 0.556 / 0.160 |
| 9 bad takes | 0.714 / 0.341 | 0.723 / 0.300 |

| Scorer | Bad takes with a mean score above the highest clean take |
|---|---|
| Spiking | 8 of 9. The miss is take 170241: 0.578 against 0.581. |
| Baseline | 9 of 9 |

| Takes | Display scores, spiking | Display scores, baseline |
|---|---|---|
| 7 clean takes that the model did not see | 72, 87, 92, 89, 100, 87, 91 | 79, 90, 94, 100, 100, 77, 80 |
| 9 bad takes | 56, 0, 73, 12, 54, 28, 18, 8, 23 | 55, 1, 71, 21, 61, 33, 4, 2, 22 |

- With a clean take of the same setting in the training, bad dancing still scores higher than clean dancing: 0.714 against 0.523.
- The margin is small for the mildest bad takes. Three bad takes have a display score of 54 to 73. The lowest clean take has 72 (spiking) or 77 (baseline).
- The display remap is fitted again: a mean score of 0.523 gives 90, and 0.714 gives 30 (spiking).
- This replaces the numbers of the 17:15 and 17:28 sections below for `models/room`.

Limit that stays: when one of the 2 new clean takes is held out, the other one is in the training. They were filmed 23 s apart. So this shows that the scorer needs one clean take of the same setting, and that it then works.

## Final state of the scorer side (17:50, frozen)

| Check on main `d495a84`, from a clean checkout | Result |
|---|---|
| `uv sync && uv run pytest`, no downloaded files | 150 pass, 17 skip, 4 known failures, 0 fail |
| The same with the two downloaded files in `assets/` | 161 pass, 6 skip, 4 known failures, 0 fail |
| Default combined command on a solo room clip (11.4 s) | Runs in 34.2 s: pose 12.3 s, robot 12.1 s, events and score 4.2 s, combined video 4.6 s |
| Demo path: 4 solo takes, leaderboard, final video | Runs. Display scores 91, 91, 34, 4 for 2 clean and 2 bad takes. |

Backup videos, outside the repo, in `demo_backup/` next to the repo folder:

| File | Content |
|---|---|
| `panel_clean_solo.mp4` | The full combined video of one clean solo take, 9.8 s |
| `demo_clean_solo_then_leaderboard.mp4` | The same, then the leaderboard, 14.8 s |
| `demo_lowest_then_leaderboard.mp4` | The combined video of the lowest-ranked take, then the leaderboard, 16.1 s |
| `leaderboard.png`, `leaderboard.mp4`, `leaderboard.json` | The leaderboard of the 4 takes |

Note for the backup: the two clean takes on that leaderboard are also training takes of `models/room`. Their display scores are not a held-out result. The held-out numbers are in the leave-one-out table below.

What is still open:

| Item | Owner |
|---|---|
| Scorer numbers in `slides/pitch.pdf` and `paper/` | The other half. `PAPER_UPDATE_NOTES.md` has the list. |
| Retrain of `models/room` with new clean takes | Done at 17:55, see the section above |
| Session doubt: clean and bad takes were filmed 30 minutes apart | Answered in part at 17:55, see the section above |

## Pull request #3 and a fix in the demo path (17:40)

| Item | State |
|---|---|
| Pull request #3 (`paper/`, `slides/pitch.pdf`, `vlma/snore_dance.py`) | Merged at 17:35 as `5a87e2a`, on the decision to merge now and update the paper later |
| Scorer numbers in the paper and the slides | Out of date. `PAPER_UPDATE_NOTES.md` lists them, with the correct numbers and the commands. |
| Fault: models trained with strips were scored with pose boxes | Fixed in `7d1a3e8`. A clean take got a display score of 17 instead of 91. |
| Complete demo path from a clean checkout | Runs: 3 takes through `vlma.run`, the leaderboard, and a final video that ends on the leaderboard. 85 s in total. |

## Scope since 17:20: one dancer per clip

The demo and the README describe one dancer per clip. The code for 2 and 3 dancers and the sync score stay in the repo as experimental. No more work goes into them.

New for the single-dancer demo:

| Part | State |
|---|---|
| Display score, 0 to 100, higher is better (`dance/display.py`) | Runs, tested |
| `dance.train --leave-one-out`: leave-one-out table and fit of the display remap | Runs, tested |
| `dance.leaderboard`: ranked table as text, PNG, final-screen video and JSON | Runs, tested, tried on 4 real takes |
| `scores_meta.json` has the new keys `models`, `display_score`, `display_remap_fitted` | Runs, tested |

Leave-one-out on the room clips, by mean score: 9 of 9 bad takes are above the highest clean take, for both scorers.

| Takes | Display score, spiking | Display score, baseline |
|---|---|---|
| 5 clean takes that the model did not see | 79, 89, 90, 91, 100 | 83, 91, 93, 90, 94 |
| 9 bad takes | 60, 9, 69, 22, 59, 34, 4, 0, 12 | 55, 13, 69, 28, 61, 35, 0, 0, 10 |

The display remap for `models/room`: a mean score of 0.511 gives 90 and a mean score of 0.744 gives 30 (spiking). `models/dance` has no fitted remap, so it uses default values.

## Bad takes from the room: a blind test (17:15)

9 more takes of the same dancer in the same room arrived without labels. I scored them before I knew that all 9 are bad takes (changed order, wrong hand positions, random motions). No model was trained on them.

Leave-one-out test with room-only models: train on 4 clean takes, score the 5th clean take and all 9 bad takes. Each cell is "mean score / fraction of bins above p90", the mean over the 5 runs.

| Takes | Spiking | Baseline |
|---|---|---|
| Clean take that the model did not see (5 runs) | 0.511 / 0.114 | 0.518 / 0.113 |
| Range over the 5 clean takes | 0.464 to 0.556 / 0.064 to 0.184 | 0.503 to 0.549 / 0.070 to 0.184 |
| The 9 bad takes | 0.751 / 0.434 | 0.776 / 0.436 |

| Bad take | Spiking | Baseline |
|---|---|---|
| 170241 | 0.592 / 0.157 | 0.604 / 0.160 |
| 170159 | 0.629 / 0.242 | 0.666 / 0.250 |
| 170320 | 0.630 / 0.222 | 0.639 / 0.225 |
| 170339 | 0.727 / 0.408 | 0.747 / 0.321 |
| 170301 | 0.776 / 0.525 | 0.776 / 0.471 |
| 170434 | 0.814 / 0.552 | 0.853 / 0.565 |
| 170220 | 0.826 / 0.623 | 0.840 / 0.573 |
| 170356 | 0.844 / 0.522 | 0.900 / 0.646 |
| 170414 | 0.861 / 0.545 | 0.894 / 0.598 |

- **With room-only models, 8 of the 9 bad takes score above every clean take.** The measure is the fraction of bins above the threshold: the highest clean take has 0.184. The one miss is take 170241 with 0.157.
- **With the YouTube + room models (`models/dance/`), no bad take is flagged.** A clean room take scores 0.192 / 0.003 and the bad takes score 0.318 / 0.002. The order is right, but nothing crosses the threshold.
- The two scorers are equal here, within 0.03.

Limit of this test: the clean takes were filmed at 16:30 and the bad takes at 17:02. The framing is a little wider in the last three bad takes. So a part of the difference can come from the session, not from the dancing. Clean takes from the same session as the bad takes will settle this.

Recommendation for clips from this room: `--models models/room`. The models in the repo did not change: bad takes are never used for training.

## Merge of the other half and combined runs (17:00)

Pull request #1 (`vlma/`: pose, robot, combined video) is merged into main as `4badf4e`.

| Check on main, from a clean checkout | Result |
|---|---|
| `uv sync && uv run pytest`, no `assets/` folder | 139 pass, 15 skip, 4 known failures, 0 fail |
| The same with the two downloaded model files in `assets/` | 148 pass, 6 skip, 4 known failures, 0 fail |

Combined command: `uv run python -m vlma.run CLIP --score-cmd "python -m dance.run {out}"`

| Real clip | Length | Pose | Robot | Events and score | Combined video | Total |
|---|---|---|---|---|---|---|
| 1 dancer, demo room | 13 s | 16.5 s | 11.3 s | 3.8 s | 4.7 s | 37.1 s |
| 3 dancers, room, other dance version | 25 s | 24.7 s | 26.5 s | 5.6 s | 12.3 s | 69.8 s |

Both runs wrote `panel.mp4`, `events.h5`, `scores.npy`, `scores_meta.json`, and used pose boxes. The 3-dancer run also wrote `sync.npy`.

## Clips from the demo room (17:00)

5 clips of 1 dancer, 10 to 13 s each, the classic steps, one performer. I read all 5 as clean clips. They are outside the repo.

Each cell is "mean score / fraction of bins above p90". One room clip was held out of each training.

| Training set | Held-out room clip, spiking | Held-out room clip, baseline | Held-out YouTube performer, spiking |
|---|---|---|---|
| YouTube classic only | 0.172 / 0.008 (clip 1) | not measured | 0.471 / 0.033 |
| YouTube classic + 4 room clips | 0.192 / 0.003 | 0.142 / 0.000 | 0.515 / 0.042 |
| 4 room clips only | 0.464 / 0.064 | 0.503 / 0.070 | 0.926 / 0.799 (all YouTube clips) |

- With YouTube clips in the training, a clean room clip scores far below the normal mark of 0.4 to 0.5. The room clips are cleaner than the YouTube clips, and they are only 12% of the training bins. The threshold then comes mostly from the YouTube clips, so the scorer is less sensitive in the room.
- With room clips only, a clean room clip scores as normal: 0.464 / 0.064.

Two model folders are in the repo:

| Folder | Training | Use |
|---|---|---|
| `models/dance/` (default) | 16 classic YouTube clips + 5 room clips | Live webcam, clips from other places |
| `models/room/` | 5 room clips only | Clips filmed in the demo room with the same camera: `--models models/room` |

Live webcam score in this room, 1 strip, 5 s, the laptop camera:

| Models | Spiking | Baseline |
|---|---|---|
| First models (1 YouTube performer) | 0.684 / 0.189 | 0.515 / 0.038 |
| Classic YouTube set | 0.806 / 0.139 | 0.739 / 0.028 |
| `models/dance/`: classic YouTube + room clips | 0.599 / 0.073 | 0.425 / 0.000 |
| `models/room/`: room clips only | 0.958 / 0.953 | 1.000 / 1.000 |

The room-only models fit the room camera, not the laptop camera: on the laptop camera they are at the ceiling. Bad clips and three-dancer clips from the room did not arrive yet, so there is no room table for them.

Bad clips and three-dancer clips with `models/dance/` trained with held-out clips (YouTube only, see the limits below):

| Clips | Spiking | Baseline |
|---|---|---|
| Bad dancing, 2 dancers (3 clips) | 0.495 / 0.137 | 0.485 / 0.096 |
| Other versions, 3 dancers (9 clips) | 0.448 / 0.021 | 0.452 / 0.016 |

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

# Notes for the paper and slides update

Written at 17:40 by the scorer side. Pull request #3 was merged at 17:35 with the paper as it was. Some scorer numbers in `paper/` and `slides/` are out of date. This file lists them, the cause, the correct numbers, and how to make them again.

"Each cell" below means "mean score / fraction of bins above p90".

## 1. Cause

Two things changed after the paper numbers were made.

| Change | Commit | Effect |
|---|---|---|
| Pose-box fault fixed | `7d1a3e8` | Before it, `dance.run` scored with pose boxes when `poses.npy` was in the episode folder. The models were trained with vertical strips. So every clip looked too unusual, or the numbers had no clear meaning. |
| Models trained again | `edb502a`, `bace402` | `models/dance/` now uses 16 classic YouTube clips and 5 room takes. `models/room/` uses the 5 room takes only. |

Every unusualness number that was made before `7d1a3e8` from an episode folder with `poses.npy` has the fault.

## 2. Values in `paper/numbers.tex` that are out of date

| Macro | Value in the paper | Problem |
|---|---|---|
| `\SoloScore`, `\SoloAbove` | 0.840, 0.425 | First models (1 YouTube performer) and pose boxes |
| `\UnuSZero` to `\UnuSSixty` | 0.840, 0.943, 0.984, 0.993, 0.993, 0.993 | Steam sweep on the public clip: the same fault |
| `\TakeAScore` to `\TakeEScore` | 0.229, 0.232, 0.212, 0.215, 0.222 | Pose boxes on strip-trained models |
| `\TakeAAbove` to `\TakeEAbove` | 0.030, 0.026, 0.038, 0.034, 0.028 | The same |
| Steam sweep on take 3 | In `steam_metrics.csv` | The same |

`\TSoloScore` (7.6 s) is a time, not a score. It is still about right: scoring took 3 to 6 s for each clip.

Figures to make again: `paper/figures/out/scorer.pdf`, `steam_curve.pdf`, `steam_strip.pdf`, and `teaser.pdf` if it shows a score trace.

## 3. Correct numbers, measured by the scorer side

All of these use vertical strips, as in the training. All are in `PROGRESS.md` with more detail.

### Public clip (the held-out YouTube performer)

| Models | Spiking | Baseline |
|---|---|---|
| Classic YouTube clips, this performer not in the training | 0.471 / 0.033 | 0.518 / 0.037 |
| The same plus 4 room takes | 0.515 / 0.042 | 0.514 / 0.037 |

### Room takes: leave-one-out with room-only models

Train on 4 clean takes, score the 5th clean take and all 9 bad takes. This is the main result for the single-dancer demo.

| Takes | Spiking | Baseline |
|---|---|---|
| Clean take that the model did not see, mean of 5 runs | 0.511 / 0.114 | 0.518 / 0.113 |
| Range over the 5 clean takes | 0.464 to 0.556 / 0.064 to 0.184 | 0.503 to 0.549 / 0.070 to 0.184 |
| 9 bad takes (changed order, wrong hand positions, random motions) | 0.744 / 0.422 | 0.769 / 0.423 |

- By mean score, 9 of 9 bad takes are above the highest clean take, for both scorers.
- By fraction above p90, 8 of 9 bad takes are above the highest clean take.
- The bad takes were scored blind: the scorer side did not know the labels.

### Room takes with the YouTube + room models (`models/dance/`)

| Takes | Spiking | Baseline |
|---|---|---|
| Clean take that the model did not see | 0.192 / 0.003 | 0.142 / 0.000 |
| 9 bad takes | 0.318 / 0.002 | 0.332 / 0.001 |

The order is right, but no bad take crosses the threshold.

### Display score (new, for the leaderboard)

| Takes | Display scores, spiking, room-only models |
|---|---|
| 5 clean takes that the model did not see | 79, 89, 90, 91, 100 |
| 9 bad takes | 60, 9, 69, 22, 59, 34, 4, 0, 12 |

The display score is 0 to 100 and higher is better. A mean score of 0.511 gives 90 and a mean score of 0.744 gives 30.

## 4. Text in `paper/main.tex` to change

| Place | Text now | Change |
|---|---|---|
| Findings, point (i) | "Our five takes score below its own held-out normal clip ... while the public clip scores 0.840 with 0.425 flagged" | The 0.840 is the fault. With correct regions the public clip scores 0.471 / 0.033. The finding that stays true: with YouTube-trained models a clean room take scores low (0.172 to 0.192), and with room-only models it scores as normal (0.511). |
| Findings, point (i), conclusion | "calibration needs clean clips of the people being scored" | Still true, and now measured: see the leave-one-out table. |
| Scorer section | "A normal held-out clip should give a mean near 0.5 and about 10% of bins above threshold" | True for the same room and camera (0.511 / 0.114). Not true across rooms. |
| New finding to add | Not in the paper | With room-only models, bad dancing scores higher than clean dancing: 0.744 against 0.511, 9 of 9 bad takes above every clean take. |
| Sync score, equation `eq:sync` | Presented as a part of the method | The sync score does not work on real clips: real dancers in step get 0.83 to 0.96, which means "no relation". Mark it as experimental or remove it. |
| Scope | "per dancer", up to 3 dancers | The demo scope is one dancer per clip since 17:20. Multi-dancer code is experimental. |
| Steam finding | "The scorer flags steam from level 10" | Run the sweep again before you keep this sentence. |

## 5. Limits to state next to the numbers

1. **Session:** the 5 clean takes were filmed at 16:30 and the 9 bad takes at 17:02. The last three bad takes are framed a little wider. A part of the gap can come from the session.
2. **In-sample:** `models/room/` and `models/dance/` in the repo were trained on all 5 clean room takes. A score of those takes with these models is not a held-out result. Use the leave-one-out numbers.
3. **Small data:** the room-only models have 55 s of clean video from one dancer.
4. **YouTube "fail" clips:** only 1 of 6 dancers scored high. Bad dancing is detected only with models from the same room and camera.
5. **Weights:** the spiking network and the baseline are equal within 0.03 on the room takes. The weights of the network do not specialise.

## 6. How to make the numbers again

```bash
git pull origin main                      # 7d1a3e8 or later

# score each episode again, with the regions that the models were trained with
uv run python -m dance.run episodes/<clip> --models models/room      # room takes
uv run python -m dance.run episodes/<clip> --models models/dance     # public clip
uv run python -m vlma.viewer episodes/<clip>                         # panel with the new scores

# leave-one-out table and display remap for a folder of clean_ and bad_ takes
uv run python -m dance.train ROOM_CLIPS --out models/room --leave-one-out

# then the paper
cd paper && ./build.sh
```

Check in the output of `dance.run` that the line says `regions: vertical strips`.

For the steam sweep, the score command is `--score-cmd "python -m dance.run {out} --models models/dance"`.

## 7. Slides

`slides/pitch.pdf` is a PDF, so the scorer side could not read its numbers. Check every unusualness number in it against section 3.

"""Display score: a friendly 0 to 100 number for one single-dancer clip.

The unusualness scores (`scores.npy`) do not change. The display score is a
fixed remap of the mean unusualness of the clip. A higher display score means
a better dance.

    mean unusualness = clean_ref  ->  display score 90
    mean unusualness = bad_ref    ->  display score 30
    linear between and outside, cut off at 0 and 100

`clean_ref` and `bad_ref` are fitted with the leave-one-out test of
`dance.train --leave-one-out`: the mean of the clean takes that the model did
not see, and the mean of the bad takes. They are saved as `display.json` in
the model folder, for each scorer. A model folder without that file uses the
default values, and every output says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DISPLAY_FILE = "display.json"
TRAIN_INFO_FILE = "train_info.json"
CLEAN_DISPLAY = 90.0
BAD_DISPLAY = 30.0
DEFAULT_CLEAN_REF = 0.50     # a clip the model did not see has a mean near 0.5
DEFAULT_BAD_REF = 0.75
MIN_GAP = 0.05               # the smallest difference between the two references


def fit_remap(clean_means: list[float], bad_means: list[float]) -> dict:
    """References from the mean scores of unseen clean clips and of bad clips."""
    if not clean_means:
        raise ValueError("the display remap needs at least 1 unseen clean clip")
    clean_ref = float(np.mean(clean_means))
    if bad_means:
        bad_ref = max(float(np.mean(bad_means)), clean_ref + MIN_GAP)
    else:
        bad_ref = clean_ref + (DEFAULT_BAD_REF - DEFAULT_CLEAN_REF)
    return {
        "fitted": True,
        "statistic": "mean score of the clip",
        "clean_ref": clean_ref,
        "bad_ref": bad_ref,
        "clean_display": CLEAN_DISPLAY,
        "bad_display": BAD_DISPLAY,
        "n_clean": len(clean_means),
        "n_bad": len(bad_means),
    }


def default_remap() -> dict:
    return {
        "fitted": False,
        "statistic": "mean score of the clip",
        "clean_ref": DEFAULT_CLEAN_REF,
        "bad_ref": DEFAULT_BAD_REF,
        "clean_display": CLEAN_DISPLAY,
        "bad_display": BAD_DISPLAY,
        "n_clean": 0,
        "n_bad": 0,
    }


def save_remaps(model_dir: str | Path, remaps: dict[str, dict]) -> Path:
    """Write `display.json`: one remap for each scorer name."""
    path = Path(model_dir) / DISPLAY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(remaps, indent=2) + "\n")
    return path


def load_remap(model_dir: str | Path, scorer: str) -> dict:
    """The remap of a scorer. The default remap if the folder has no fitted one."""
    path = Path(model_dir) / DISPLAY_FILE
    if path.exists():
        remaps = json.loads(path.read_text())
        if scorer in remaps:
            return remaps[scorer]
    return default_remap()


def save_train_info(model_dir: str | Path, regions: list[str]) -> Path:
    """Record which dancer regions the training clips used: "strips", "poses" or "mixed"."""
    kinds = sorted(set(regions))
    info = {"regions": kinds[0] if len(kinds) == 1 else "mixed", "clips": len(regions)}
    path = Path(model_dir) / TRAIN_INFO_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")
    return path


def trained_regions(model_dir: str | Path) -> str | None:
    """ "strips", "poses" or "mixed"; None if the model folder does not say."""
    path = Path(model_dir) / TRAIN_INFO_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("regions")


def display_score(mean_score: float, remap: dict) -> float:
    """Mean unusualness of a clip (0 to 1) -> display score (0 to 100, higher is better)."""
    span = remap["bad_ref"] - remap["clean_ref"]
    slope = (remap["bad_display"] - remap["clean_display"]) / span
    value = remap["clean_display"] + slope * (float(mean_score) - remap["clean_ref"])
    return float(min(100.0, max(0.0, value)))

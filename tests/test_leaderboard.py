"""Display score, leave-one-out fit of the remap, and the leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dance import display, leaderboard
from dance import train as train_module


def _episode(root: Path, name: str, mean: float, scorer="baseline", models="models/x", dancers=1) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    scores = np.clip(np.full((100, dancers), mean, dtype=np.float32), 0, 1)
    np.save(folder / "scores.npy", scores)
    (folder / "scores_meta.json").write_text(json.dumps(
        {"bin_ms": 33, "bins": 100, "scorer": scorer, "models": models, "threshold": 0.9}
    ))
    return folder


def test_display_score_anchors_and_limits():
    remap = display.fit_remap([0.48, 0.52], [0.70, 0.80])
    assert remap["fitted"] and remap["clean_ref"] == pytest.approx(0.50) and remap["bad_ref"] == pytest.approx(0.75)
    assert display.display_score(0.50, remap) == pytest.approx(90.0)     # a clean take
    assert display.display_score(0.75, remap) == pytest.approx(30.0)     # a bad take
    assert display.display_score(0.10, remap) == 100.0 and display.display_score(0.99, remap) == 0.0
    assert display.display_score(0.60, remap) > display.display_score(0.70, remap)   # higher is better
    no_bad = display.fit_remap([0.5], [])
    assert no_bad["bad_ref"] == pytest.approx(0.75)
    assert display.default_remap()["fitted"] is False


def test_remap_is_saved_per_scorer_and_default_without_file(tmp_path):
    display.save_remaps(tmp_path, {"spiking": display.fit_remap([0.4], [0.9])})
    assert display.load_remap(tmp_path, "spiking")["clean_ref"] == pytest.approx(0.4)
    assert display.load_remap(tmp_path, "baseline")["fitted"] is False
    assert display.load_remap(tmp_path / "none", "spiking")["fitted"] is False


def test_leaderboard_ranks_best_first_and_writes_three_files(tmp_path, capsys):
    display.save_remaps(tmp_path / "m", {"baseline": display.fit_remap([0.5], [0.75])})
    eps = [_episode(tmp_path / "episodes", n, m, models=str(tmp_path / "m"))
           for n, m in (("a", 0.70), ("b", 0.45), ("c", 0.58))]
    out = tmp_path / "episodes" / "leaderboard.png"
    leaderboard.main([*map(str, eps), "--names", "Ann", "Ben", "Cy", "--out", str(out), "--seconds", "1"])

    board = json.loads(out.with_suffix(".json").read_text())
    assert [r["name"] for r in board["ranking"]] == ["Ben", "Cy", "Ann"]         # lowest unusualness first
    assert [r["rank"] for r in board["ranking"]] == [1, 2, 3]
    assert board["ranking"][0]["display_score"] == 100.0 or board["ranking"][0]["display_score"] > 90
    assert board["lowest"] == {"name": "Ann", "episode": str(eps[0])}            # the robot replays this one
    assert board["scorer"] == "baseline" and board["remap"]["fitted"] is True
    picture = cv2.imread(str(out))
    assert picture.shape == (720, 1280, 3)
    cap = cv2.VideoCapture(str(out.with_suffix(".mp4")))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 25
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1280, 720)
    cap.release()
    printed = capsys.readouterr().out
    assert "lowest ranked: Ann" in printed and printed.index("Ben") < printed.index("Cy") < printed.index("Ann")


def test_leaderboard_is_for_one_dancer_per_clip(tmp_path):
    two = _episode(tmp_path, "duo", 0.5, dancers=2)
    with pytest.raises(SystemExit) as stopped:
        leaderboard.main([str(two), "--out", str(tmp_path / "lb.png")])
    assert "one dancer" in str(stopped.value.code)


def test_leaderboard_stops_on_missing_scores_wrong_names_and_mixed_models(tmp_path):
    a = _episode(tmp_path, "a", 0.5)
    b = _episode(tmp_path, "b", 0.6, models="models/other")
    (tmp_path / "empty").mkdir()
    for argv, text in (
        ([str(tmp_path / "empty")], "dance.run"),
        ([str(a), "--names", "x", "y"], "names"),
        ([str(a), str(b)], "different"),
    ):
        with pytest.raises(SystemExit) as stopped:
            leaderboard.main([*argv, "--out", str(tmp_path / "lb.png")])
        assert text in str(stopped.value.code)


def test_default_remap_is_named_in_the_output(tmp_path, capsys):
    a = _episode(tmp_path, "a", 0.5)
    leaderboard.main([str(a), "--out", str(tmp_path / "lb.png"), "--seconds", "1"])
    assert "not fitted" in capsys.readouterr().out


def test_leave_one_out_fits_the_display_remap(tmp_path, capsys):
    from tests.conftest import square_positions, write_square_clip

    root = tmp_path / "clips"
    root.mkdir()
    for i in range(4):
        write_square_clip(root / f"clean_{i}.mp4", square_positions(60, phase=0.3 * i), 60)
    write_square_clip(root / "bad_0.mp4", square_positions(60, jumps=(15, 30, 45)), 60)
    out = tmp_path / "models"
    train_module.main([str(root), "--out", str(out), "--scorer", "baseline", "--leave-one-out", "--jobs", "1"])
    printed = capsys.readouterr().out

    assert "Leave-one-out test: 4 clean clip(s), 1 bad clip(s)" in printed
    assert printed.count("| clean, not seen: clean_") == 4 and "| bad: bad_0 |" in printed
    assert "Trained on 4 clean clip(s)" in printed                # the saved model uses all clean clips
    remap = json.loads((out / display.DISPLAY_FILE).read_text())["baseline"]
    assert remap["fitted"] and remap["n_clean"] == 4 and remap["n_bad"] == 1
    assert remap["bad_ref"] > remap["clean_ref"]

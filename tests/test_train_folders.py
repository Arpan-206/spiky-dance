"""Training and evaluation on the 1p / 2p / 3p folder layout, with clean_ and bad_ clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dance import baseline, evaluate
from dance import train as train_module
from tests.conftest import square_positions, write_square_clip


def _clip(path: Path, phase: float, jumps=()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_square_clip(path, square_positions(60, phase=phase, jumps=jumps), 60)


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("clips")
    for i in range(3):
        _clip(root / "1p" / f"clean_a_{i}.mp4", 0.3 * i)
    _clip(root / "1p" / "clean_b_0.mp4", 1.0)                    # a second "video"
    _clip(root / "1p" / "bad_x.mp4", 0.5, jumps=(20, 40))
    for i in range(2):
        _clip(root / "2p" / f"clean_c_{i}.mp4", 0.2 * i)
    _clip(root / "2p" / "bad_y.mp4", 0.1, jumps=(15, 30, 45))
    (root / "notes").mkdir()                                      # not a dancer-count folder
    return root


def test_collect_clips_reads_dancer_count_and_role(tree):
    clips = evaluate.collect_clips([tree])
    assert [(c.group, c.path.name, c.role) for c in clips] == [
        ("1p", "bad_x.mp4", "bad"), ("1p", "clean_a_0.mp4", "clean"), ("1p", "clean_a_1.mp4", "clean"),
        ("1p", "clean_a_2.mp4", "clean"), ("1p", "clean_b_0.mp4", "clean"),
        ("2p", "bad_y.mp4", "bad"), ("2p", "clean_c_0.mp4", "clean"), ("2p", "clean_c_1.mp4", "clean"),
    ]
    assert [c.dancers for c in clips] == [1, 1, 1, 1, 1, 2, 2, 2]


def test_flat_folder_form_still_works(tree):
    clips = evaluate.collect_clips([tree / "1p"], dancers=2)
    assert len(clips) == 5 and {c.dancers for c in clips} == {2}


def test_held_out_choice(tree):
    clips = evaluate.collect_clips([tree])
    default = train_module.choose_held_out(clips, None, False)
    assert [c.path.name for c in default] == ["clean_b_0.mp4", "clean_c_1.mp4"]     # last clean clip of each count
    by_prefix = train_module.choose_held_out(clips, ["clean_a"], False)
    assert [c.path.name for c in by_prefix] == ["clean_a_0.mp4", "clean_a_1.mp4", "clean_a_2.mp4"]   # a whole video
    assert train_module.choose_held_out(clips, None, True) == []
    with pytest.raises(ValueError):
        train_module.choose_held_out(clips, ["bad_"], False)         # a bad clip is never a held-out clean clip
    with pytest.raises(ValueError):
        train_module.choose_held_out(clips, ["nothing_"], False)


def test_one_command_trains_on_all_counts_and_never_on_bad_clips(tree, tmp_path, monkeypatch, capsys):
    trained_on = []
    real_train = baseline.train

    def spy(feature_clips, out_dir):
        trained_on.extend(f.shape for f in feature_clips)
        real_train(feature_clips, out_dir)

    monkeypatch.setattr(baseline, "train", spy)
    out = tmp_path / "models"
    train_module.main([str(tree), "--out", str(out), "--scorer", "baseline", "--jobs", "1"])
    printed = capsys.readouterr().out

    assert (out / baseline.ARTEFACT).exists()
    # 5 clean clips with 1 dancer and 2 with 2 dancers; 1 held out of each count; the 2 bad clips are not used.
    assert sorted(s[1] for s in trained_on) == [1, 1, 1, 2]
    assert "Trained on 4 clean clip(s): 3 with 1 dancer(s), 1 with 2 dancer(s)" in printed
    assert "1p/bad_x.mp4" in printed and ", bad" in printed
    assert "### 1 dancer(s)" in printed and "### 2 dancer(s)" in printed
    assert "| held-out clean | clean_b_0 |" in printed and "| bad | bad_x |" in printed
    assert "| bad | bad_y | 1 |" in printed                      # one row for each dancer of the 2-dancer clip
    assert "**bad, all 1 clip(s)**" in printed


def test_two_roots_and_a_whole_video_held_out(tree, tmp_path, capsys):
    other = tmp_path / "mine"
    _clip(other / "1p" / "clean_room_0.mp4", 0.7)
    out = tmp_path / "models"
    train_module.main([str(tree), str(other), "--out", str(out), "--scorer", "baseline",
                       "--holdout", "clean_a", "--jobs", "1"])
    printed = capsys.readouterr().out
    assert "Trained on 4 clean clip(s): 2 with 1 dancer(s), 2 with 2 dancer(s)" in printed
    assert printed.count("| held-out clean | clean_a_") == 3


def test_evaluate_prints_the_table_for_trained_models(tree, tmp_path, capsys):
    out = tmp_path / "models"
    train_module.main([str(tree), "--out", str(out), "--scorer", "baseline", "--no-holdout", "--jobs", "1"])
    capsys.readouterr()
    evaluate.main([str(tree), "--models", str(out), "--scorer", "baseline", "--jobs", "1"])
    printed = capsys.readouterr().out
    assert "| clean | clean_a_0 | 0 |" in printed and "| bad | bad_y | 1 |" in printed
    assert "**clean, all 4 clip(s)**" in printed and "**bad, all 1 clip(s)**" in printed


def test_evaluate_stops_without_trained_models(tree, tmp_path):
    with pytest.raises(SystemExit) as stopped:
        evaluate.main([str(tree), "--models", str(tmp_path / "none")])
    assert "dance.train" in str(stopped.value.code)

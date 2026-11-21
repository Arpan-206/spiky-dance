"""Step 4: scoring commands, offline and live.

The trained model does not have to exist for these tests. `FakeModel` below
stands in for it. A fake model is allowed only in this file.
"""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dance import run, score, stream
from dance.contracts import BIN_MS, FEATURE_DIM
from dance.features import Regions

REPO = Path(__file__).resolve().parents[1]


class FakeModel:
    """Deterministic scores of the right shape. Dancer d gets a different level."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def score(self, features: np.ndarray) -> np.ndarray:
        self.calls.append(features.shape)
        bins, dancers, dim = features.shape
        assert dim == FEATURE_DIM
        level = np.linspace(0.2, 0.95, dancers, dtype=np.float32)
        wave = 0.04 * np.sin(np.arange(bins, dtype=np.float32))[:, None]
        return np.clip(level[None, :] + wave, 0, 1).astype(np.float32)


@pytest.fixture
def fake_model(monkeypatch) -> FakeModel:
    """Make `score.load` return a FakeModel, as if training was done."""
    model = FakeModel()
    monkeypatch.setattr(score, "load", lambda model_dir=None, scorer=None: model)
    return model


@pytest.fixture
def no_artefacts(monkeypatch):
    """A `dance.spiky` whose `load_model` finds no artefacts, as the contract says."""
    def load_model(model_dir):
        raise FileNotFoundError(f"no artefacts in {model_dir}")

    module = types.ModuleType("dance.spiky")
    module.load_model = load_model
    monkeypatch.setitem(sys.modules, "dance.spiky", module)


def _forbid_event_work(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("event work started before the model was loaded")

    monkeypatch.setattr(score, "clip_features", fail)


# ------------------------------------------------------------------- summary

def test_summary_numbers_on_hand_made_scores():
    scores = np.array(
        [[0.10, 0.95, 0.50],
         [0.20, 0.90, 0.50],
         [0.30, 0.89, 0.92],
         [0.40, 1.00, 0.08]],
        dtype=np.float32,
    )
    summary = score.summarise(scores)
    rows = summary["dancers"]
    assert [r["dancer"] for r in rows] == [0, 1, 2]
    assert [r["bins"] for r in rows] == [4, 4, 4]
    assert np.allclose([r["mean_score"] for r in rows], [0.25, 0.935, 0.50], atol=1e-6)
    # 0.90 itself counts as above. 0.89 does not.
    assert [r["fraction_above_p90"] for r in rows] == [0.0, 0.75, 0.25]
    assert summary["ranking"] == [1, 2, 0]
    assert [r["rank"] for r in rows] == [3, 1, 2]
    assert summary["bins"] == 4 and summary["threshold"] == 0.90 and summary["bin_ms"] == BIN_MS


def test_summary_refuses_zero_bins():
    with pytest.raises(ValueError):
        score.summarise(np.zeros((0, 2), dtype=np.float32))


def test_ranked_table_is_ordered_by_mean_score():
    scores = np.tile(np.array([[0.3, 0.8, 0.1, 0.6]], dtype=np.float32), (10, 1))
    table = score.format_table(score.summarise(scores))
    lines = table.splitlines()
    assert lines[0].split() == ["rank", "dancer", "mean", "score", "fraction", "above", "p90", "bins"]
    body = [line.split() for line in lines[2:]]
    assert [row[0] for row in body] == ["1", "2", "3", "4"]
    assert [row[1] for row in body] == ["1", "3", "0", "2"]
    means = [float(row[2]) for row in body]
    assert means == sorted(means, reverse=True)
    assert [row[4] for row in body] == ["10"] * 4


def test_score_features_shape_and_range():
    feats = np.zeros((40, 3, FEATURE_DIM), dtype=np.float32)
    scores, summary = score.score_features(FakeModel(), feats)
    assert scores.shape == (40, 3) and scores.dtype == np.float32
    assert scores.min() >= 0 and scores.max() <= 1
    assert summary["ranking"] == [2, 1, 0]


@pytest.mark.parametrize(
    "bad",
    [
        lambda f: np.zeros((f.shape[0],), dtype=np.float32),            # wrong shape
        lambda f: np.full(f.shape[:2], 1.5, dtype=np.float32),          # above 1
        lambda f: np.full(f.shape[:2], -0.1, dtype=np.float32),         # below 0
        lambda f: np.full(f.shape[:2], np.nan, dtype=np.float32),       # NaN
    ],
)
def test_model_output_outside_the_contract_is_an_error(bad):
    model = types.SimpleNamespace(score=bad)
    with pytest.raises(ValueError):
        score.score_features(model, np.zeros((5, 2, FEATURE_DIM), dtype=np.float32))


# ------------------------------------------------------------------- offline

def test_offline_cli_writes_scores_and_summary(moving_clip, tmp_path, fake_model, capsys):
    out = tmp_path / "result" / "scores.npy"
    score.main([str(moving_clip), "--out", str(out), "--dancers", "2"])

    scores = np.load(out)
    assert scores.dtype == np.float32
    assert scores.ndim == 2 and scores.shape[1] == 2
    assert 58 <= scores.shape[0] <= 61          # 60 frames at 30 fps, 33 ms bins
    assert scores.min() >= 0 and scores.max() <= 1
    assert fake_model.calls == [(scores.shape[0], 2, FEATURE_DIM)]

    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["bins"] == scores.shape[0]
    assert summary["regions"] == "strips"
    assert summary["scores_file"] == "scores.npy"
    assert summary["ranking"] == [1, 0]
    for row in summary["dancers"]:
        d = row["dancer"]
        assert np.isclose(row["mean_score"], scores[:, d].mean(), atol=1e-6)
        assert np.isclose(row["fraction_above_p90"], (scores[:, d] >= 0.90).mean())

    printed = capsys.readouterr().out
    assert "rank" in printed and "fraction above p90" in printed
    assert re.search(r"elapsed: \d+\.\d s", printed)


def test_offline_scores_an_eventify_h5_file(moving_clip, tmp_path, fake_model):
    summary = run.run_clip(moving_clip, tmp_path / "a", dancers=2)
    out = tmp_path / "b" / "scores.npy"
    score.main([summary["events_file"], "--out", str(out), "--dancers", "2"])
    from_clip = np.load(summary["scores_file"])
    from_events = np.load(out)
    assert from_events.shape[1] == 2
    assert abs(from_events.shape[0] - from_clip.shape[0]) <= 1


def test_offline_uses_poses_when_given(moving_clip, tmp_path, fake_model):
    poses = np.zeros((60, 3, 17, 3), dtype=np.float32)
    for d in range(3):
        poses[:, d, :, 0] = np.linspace(10 + 50 * d, 40 + 50 * d, 17)
        poses[:, d, :, 1] = np.linspace(20, 100, 17)
        poses[:, d, :, 2] = 1.0
    poses_path = tmp_path / "poses_in.npy"
    np.save(poses_path, poses)
    out = tmp_path / "scores.npy"
    score.main([str(moving_clip), "--out", str(out), "--poses", str(poses_path)])
    assert np.load(out).shape[1] == 3
    assert json.loads(out.with_suffix(".json").read_text())["regions"] == "poses"


REQUIRED_SCORES_META = ("bin_ms", "bins", "sensor_width", "sensor_height", "scorer", "threshold")


def test_run_writes_the_episode_folder(moving_clip, tmp_path, fake_model, capsys):
    episodes = tmp_path / "episodes"
    run.main([str(moving_clip), "--episodes-dir", str(episodes), "--dancers", "3", "--scorer", "baseline"])

    episode = episodes / "moving"                      # the clip name without its suffix
    events, shape = stream.read_hdf5(episode / "events.h5")
    assert events.size > 0 and shape == (120, 160)
    scores = np.load(episode / "scores.npy")
    assert scores.dtype == np.float32 and scores.shape[1] == 3
    assert scores.min() >= 0 and scores.max() <= 1
    meta = json.loads((episode / "scores_meta.json").read_text())
    for key in REQUIRED_SCORES_META:
        assert key in meta, key
    assert meta["bin_ms"] == BIN_MS and meta["bins"] == scores.shape[0]
    assert (meta["sensor_width"], meta["sensor_height"]) == (160, 120)
    assert meta["scorer"] == "baseline" and meta["threshold"] == 0.90
    assert meta["regions"] == "strips" and meta["dancers"] == 3
    assert not (episode / "meta.json").exists()        # their file: never written here

    printed = capsys.readouterr().out
    assert "rank" in printed and "events.h5" in printed and "elapsed" in printed
    assert "vertical strips" in printed


def _episode_with_poses(tmp_path):
    """An episode folder as the pose build leaves it: clip, poses.npy, meta.json, pose3d.npz."""
    from tests.conftest import square_positions, write_square_clip

    episode = tmp_path / "episodes" / "take1"
    episode.mkdir(parents=True)
    write_square_clip(episode / "take1.mp4", square_positions(60), 60, upscale=4)   # 640x480
    poses = np.zeros((60, 2, 17, 3), dtype=np.float32)
    poses[:, 0, :, 0] = np.linspace(92, 548, 17)       # pixels of the ORIGINAL video
    poses[:, 0, :, 1] = np.linspace(72, 408, 17)
    poses[:, 0, :, 2] = 0.9
    poses[:, 1, :, 0] = np.linspace(10, 60, 17)
    poses[:, 1, :, 1] = np.linspace(430, 470, 17)
    poses[:, 1, :, 2] = 0.9
    poses[:, 1, 5:, 2] = 0.0                           # missing joints have confidence 0
    np.save(episode / "poses.npy", poses)
    (episode / "meta.json").write_text(json.dumps({"fps": 30.0, "width": 640, "height": 480, "frames": 60}))
    (episode / "pose3d.npz").write_bytes(b"theirs")
    return episode


def test_run_on_an_episode_folder_uses_its_poses_and_meta(tmp_path, fake_model, capsys):
    episode = _episode_with_poses(tmp_path)
    their_meta = (episode / "meta.json").read_bytes()

    run.main([str(episode), "--scorer", "baseline", "--models", str(tmp_path / "no_train_info")])

    meta = json.loads((episode / "scores_meta.json").read_text())
    assert meta["regions"] == "poses" and meta["dancers"] == 2
    assert (meta["sensor_width"], meta["sensor_height"]) == (320, 240)
    assert meta["pose_scale"] == [0.5, 0.5]            # 640x480 original -> 320x240 sensor
    assert np.load(episode / "scores.npy").shape == (meta["bins"], 2)
    assert (episode / "events.h5").exists()
    assert (episode / "meta.json").read_bytes() == their_meta
    assert (episode / "pose3d.npz").read_bytes() == b"theirs"
    # Two dancers: the sync score is a second file, and scores_meta.json records it.
    sync = np.load(episode / "sync.npy")
    assert sync.shape == (meta["bins"], 2) and sync.min() >= 0 and sync.max() <= 2
    assert meta["sync"]["file"] == "sync.npy" and meta["sync"]["window_s"] == 2.0
    assert len(meta["sync"]["mean_per_dancer"]) == 2
    assert "pose boxes" in capsys.readouterr().out


def test_run_scores_with_the_regions_that_the_models_were_trained_with(tmp_path, fake_model, capsys):
    """Models trained on strips must be scored with strips, also when the episode has poses.npy."""
    from dance import display

    episode = _episode_with_poses(tmp_path)
    strips_models = tmp_path / "strips_models"
    display.save_train_info(strips_models, ["strips", "strips"])
    run.main([str(episode), "--models", str(strips_models)])            # auto
    meta = json.loads((episode / "scores_meta.json").read_text())
    assert meta["regions"] == "strips" and meta["pose_scale"] is None
    assert meta["dancers"] == 1                                         # --dancers default, meta.json has no count
    assert "poses.npy is present but not used" in capsys.readouterr().out

    run.main([str(episode), "--models", str(strips_models), "--regions", "poses"])     # forced
    assert json.loads((episode / "scores_meta.json").read_text())["regions"] == "poses"

    pose_models = tmp_path / "pose_models"
    display.save_train_info(pose_models, ["poses"])
    run.main([str(episode), "--models", str(pose_models)])              # auto, models trained on poses
    assert json.loads((episode / "scores_meta.json").read_text())["regions"] == "poses"

    run.main([str(episode), "--models", str(pose_models), "--regions", "strips"])
    assert json.loads((episode / "scores_meta.json").read_text())["regions"] == "strips"


def test_committed_models_say_how_they_were_trained():
    from dance import display

    assert display.trained_regions("models/room") == "strips"
    assert display.trained_regions("models/dance") == "strips"
    assert display.load_remap("models/room", "spiking")["fitted"] is True


def test_run_on_an_episode_folder_with_only_events(moving_clip, tmp_path, fake_model):
    episodes = tmp_path / "episodes"
    run.main([str(moving_clip), "--episodes-dir", str(episodes)])
    episode = episodes / "moving"
    first = np.load(episode / "scores.npy")
    events_before = (episode / "events.h5").stat().st_mtime_ns
    (episode / "scores.npy").unlink()

    run.main([str(episode), "--dancers", "2"])         # no video in the folder: reads events.h5

    again = np.load(episode / "scores.npy")
    assert again.shape[1] == 2 and abs(len(again) - len(first)) <= 1
    assert first.shape[1] == 1                         # the first run had 1 dancer: no sync score
    assert (episode / "sync.npy").exists()             # the second run has 2 dancers
    run.main([str(episode), "--dancers", "1"])
    assert not (episode / "sync.npy").exists()         # 1 dancer again: the old file is removed
    assert json.loads((episode / "scores_meta.json").read_text())["sync"] is None
    assert (episode / "events.h5").stat().st_mtime_ns == events_before


def test_run_uses_clip_mp4_when_the_folder_has_other_videos(moving_clip, static_clip, tmp_path, fake_model, capsys):
    """The other build puts skeleton.mp4, robot.mp4 and panel.mp4 in the same episode folder."""
    import shutil

    episode = tmp_path / "episodes" / "take7"
    episode.mkdir(parents=True)
    shutil.copy(moving_clip, episode / "clip.mp4")
    for other in ("skeleton.mp4", "robot.mp4", "panel.mp4", "preview.mp4"):
        shutil.copy(static_clip, episode / other)          # static: they would give no events
    (episode / "meta.json").write_text(json.dumps(
        {"fps": 30.0, "width": 160, "height": 120, "n_frames": 60, "dancers": 3, "clip_id": "take7"}
    ))

    run.main([str(episode)])                               # no --dancers: read it from meta.json

    meta = json.loads((episode / "scores_meta.json").read_text())
    assert meta["source"] == "clip.mp4"
    assert meta["dancers"] == 3 and meta["regions"] == "strips"
    events, _ = stream.read_hdf5(episode / "events.h5")
    assert events.size > 0                                 # the moving clip, not a static one
    assert np.load(episode / "scores.npy").shape[1] == 3
    assert "from meta.json" in capsys.readouterr().out

    run.main([str(episode), "--dancers", "2"])             # --dancers wins over meta.json
    assert np.load(episode / "scores.npy").shape[1] == 2


def test_run_without_clip_mp4_uses_the_one_other_video_or_stops(moving_clip, static_clip, tmp_path, fake_model):
    import shutil

    episode = tmp_path / "episodes" / "take8"
    episode.mkdir(parents=True)
    shutil.copy(moving_clip, episode / "dance.mp4")
    shutil.copy(static_clip, episode / "preview.mp4")      # our own preview is never the source
    run.main([str(episode)])
    assert json.loads((episode / "scores_meta.json").read_text())["source"] == "dance.mp4"

    shutil.copy(static_clip, episode / "robot.mp4")        # now two other videos and no clip.mp4
    with pytest.raises(SystemExit) as stopped:
        run.main([str(episode)])
    assert "more than one other video" in str(stopped.value.code)
    assert "clip.mp4" in str(stopped.value.code)


def test_meta_json_keys_of_the_other_build(tmp_path):
    from dance.features import read_pose_meta

    path = tmp_path / "meta.json"
    path.write_text(json.dumps(
        {"fps": 25.0, "width": 1280, "height": 720, "n_frames": 750, "dancers": 3, "clip_id": "room_01"}
    ))
    assert read_pose_meta(path) == {
        "width": 1280, "height": 720, "fps": 25.0, "frames": 750, "dancers": 3, "clip_id": "room_01",
    }


def test_events_file_keeps_a_freeze_at_the_end_of_the_clip(tmp_path, fake_model):
    """A clip that ends without events must give the same bins from events.h5 as from the clip."""
    from tests.conftest import square_positions, write_square_clip

    positions = square_positions(90)
    positions[45:] = positions[45]                         # the last 1.5 s: no motion, no events
    clip = write_square_clip(tmp_path / "freeze_end.mp4", positions, 90)
    episodes = tmp_path / "episodes"
    run.main([str(clip), "--episodes-dir", str(episodes)])
    episode = episodes / "freeze_end"
    from_clip = np.load(episode / "scores.npy")
    events, _ = stream.read_hdf5(episode / "events.h5")
    assert events["t"].max() < 1_700_000                   # the events stop long before the end
    assert stream.read_hdf5_duration(episode / "events.h5") == int(89 * 1_000_000 / 30.0)

    run.main([str(episode)])                               # no video in the folder: reads events.h5
    from_events = np.load(episode / "scores.npy")
    assert from_events.shape == from_clip.shape            # the frozen bins are not lost
    assert len(from_clip) >= 88


def test_run_stops_on_an_empty_episode_folder(tmp_path, fake_model):
    empty = tmp_path / "episodes" / "nothing"
    empty.mkdir(parents=True)
    with pytest.raises(SystemExit) as stopped:
        run.main([str(empty)])
    assert "no video" in str(stopped.value.code)


# --------------------------------------------------------- missing artefacts

def test_load_names_the_train_command(tmp_path, no_artefacts):
    with pytest.raises(FileNotFoundError, match="dance.train"):
        score.load(tmp_path / "empty")


def test_score_stops_when_artefacts_are_missing(moving_clip, tmp_path, no_artefacts, monkeypatch, capsys):
    _forbid_event_work(monkeypatch)
    out = tmp_path / "scores.npy"
    with pytest.raises(SystemExit) as stopped:
        score.main([str(moving_clip), "--out", str(out), "--models", str(tmp_path / "empty")])
    assert stopped.value.code not in (0, None)
    assert "dance.train" in str(stopped.value.code)
    assert not out.exists() and not out.with_suffix(".json").exists()
    assert "rank" not in capsys.readouterr().out


def test_score_clip_raises_when_artefacts_are_missing(moving_clip, tmp_path, no_artefacts, monkeypatch):
    _forbid_event_work(monkeypatch)
    with pytest.raises(FileNotFoundError, match="dance.train"):
        score.score_clip(moving_clip, tmp_path / "empty")


def test_run_stops_when_artefacts_are_missing(moving_clip, tmp_path, no_artefacts, monkeypatch):
    _forbid_event_work(monkeypatch)
    out_dir = tmp_path / "episodes"
    with pytest.raises(SystemExit) as stopped:
        run.main([str(moving_clip), "--episodes-dir", str(out_dir), "--models", str(tmp_path / "empty")])
    assert stopped.value.code not in (0, None)
    assert "dance.train" in str(stopped.value.code)
    assert not out_dir.exists()


def test_live_stops_when_artefacts_are_missing(tmp_path, no_artefacts, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("the webcam was opened before the model was loaded")

    monkeypatch.setattr(stream, "probe_source", fail)
    monkeypatch.setattr(stream, "event_chunks", fail)
    with pytest.raises(SystemExit) as stopped:
        score.main(["--live", "--models", str(tmp_path / "empty")])
    assert "dance.train" in str(stopped.value.code)


def test_commands_exit_non_zero_without_a_trained_model(moving_clip, tmp_path):
    """The real commands, in a new process, with an empty model folder."""
    empty = tmp_path / "empty_models"
    empty.mkdir()
    for module, args in (
        ("dance.score", ["--out", str(tmp_path / "scores.npy")]),
        ("dance.run", ["--out-dir", str(tmp_path / "out")]),
    ):
        done = subprocess.run(
            [sys.executable, "-m", module, str(moving_clip), "--models", str(empty), *args],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        assert done.returncode != 0, done.stdout
        assert done.stderr.strip()
    assert not (tmp_path / "scores.npy").exists()
    assert not (tmp_path / "scores.json").exists()
    assert not (tmp_path / "out").exists()


def test_real_loader_refuses_an_empty_folder(tmp_path):
    """Runs when the model code exists. The contract says FileNotFoundError."""
    try:
        from dance.spiky import load_model  # noqa: F401
    except ImportError:
        pytest.skip("dance.spiky.load_model is not written yet")
    with pytest.raises(FileNotFoundError, match="dance.train"):
        score.load(tmp_path / "empty")


# ---------------------------------------------------------------------- live

def _file_windows(clip: Path) -> tuple[list[np.ndarray], stream.SourceInfo]:
    info = stream.probe_source(str(clip))
    return list(score.source_windows(str(clip), info)), info


def test_live_loop_from_a_file_emits_every_three_windows(moving_clip):
    windows, info = _file_windows(moving_clip)
    assert 55 <= len(windows) <= 61
    model = FakeModel()
    regions = Regions(info.sensor_shape, None, 2)
    updates: list[dict] = []

    session = score.live_loop(iter(windows), model, regions, updates.append)

    assert len(updates) == len(windows) // 3
    for i, update in enumerate(updates):
        assert set(update) == {"t", "scores", "latest", "above", "sync"}
        assert len(update["scores"]) == len(update["latest"]) == len(update["above"]) == 2
        assert all(0 <= v <= 1 for v in update["scores"] + update["latest"])
        assert all(isinstance(v, bool) for v in update["above"])
        assert np.isclose(update["t"], (i + 1) * 3 * BIN_MS / 1000, atol=1e-3)
        assert update["above"] == [v >= 0.90 for v in update["scores"]]
    # About 10 updates a second.
    assert np.isclose(np.diff([u["t"] for u in updates]).mean(), 0.099, atol=1e-3)
    # Dancer 1 of the FakeModel is above p90. Dancer 0 is not.
    assert updates[-1]["above"] == [False, True]
    # The model gets the growing buffer: 3, 6, 9, ... feature rows.
    assert model.calls[:3] == [(3, 2, FEATURE_DIM), (6, 2, FEATURE_DIM), (9, 2, FEATURE_DIM)]
    assert session.shape == (len(updates) * 3, 2) and session.dtype == np.float32


def test_live_loop_keeps_a_buffer_of_three_seconds(moving_clip):
    windows, info = _file_windows(moving_clip)
    windows = windows * 4                       # about 8 seconds of windows
    model = FakeModel()
    updates: list[dict] = []
    score.live_loop(iter(windows), model, Regions(info.sensor_shape, None, 1), updates.append)
    assert len(updates) == len(windows) // 3
    assert max(shape[0] for shape in model.calls) == score.BUFFER_WINDOWS == 90
    assert model.calls[-1] == (90, 1, FEATURE_DIM)


def test_live_rolling_score_is_the_mean_of_the_last_second():
    class Ramp:
        def score(self, features):
            bins = features.shape[0]
            return (np.arange(bins, dtype=np.float32)[:, None] / 100).repeat(features.shape[1], axis=1)

    windows = [np.zeros((1, 2, 12, 16), dtype=np.float32) for _ in range(60)]
    updates: list[dict] = []
    score.live_loop(iter(windows), Ramp(), Regions((12, 16), None, 1), updates.append)
    last = updates[-1]                          # buffer has 60 rows: scores 0.00 .. 0.59
    assert np.isclose(last["latest"][0], 0.59, atol=1e-4)
    assert np.isclose(last["scores"][0], np.arange(30, 60).mean() / 100, atol=1e-4)


def test_live_loop_stops_cleanly_on_ctrl_c():
    def windows():
        for k in range(100):
            if k == 20:
                raise KeyboardInterrupt
            yield np.zeros((1, 2, 12, 16), dtype=np.float32)

    updates: list[dict] = []
    session = score.live_loop(windows(), FakeModel(), Regions((12, 16), None, 2), updates.append)
    assert len(updates) == 6                    # 20 windows -> 6 full groups of 3
    assert session.shape == (18, 2)
    assert "rank" in score.format_table(score.summarise(session))


def test_live_command_from_a_file_prints_json_lines(moving_clip, fake_model, monkeypatch, capsys):
    """`--live --json` with the webcam replaced by a clip file. No webcam is opened."""
    info = stream.probe_source(str(moving_clip))
    opened: list = []

    def event_chunks(source, info_, **kwargs):
        opened.append((source, kwargs))
        return real_chunks(str(moving_clip), info_)

    real_chunks = stream.event_chunks
    monkeypatch.setattr(stream, "probe_source", lambda device: info)
    monkeypatch.setattr(stream, "event_chunks", event_chunks)

    score.main(["--live", "--device", "7", "--dancers", "2", "--duration", "5", "--json"])

    assert opened == [(7, {"duration_s": 5.0, "warmup_frames": 30})]
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) >= 5
    for line in lines:                          # stdout has only JSON objects
        update = json.loads(line)
        assert len(update["scores"]) == len(update["latest"]) == len(update["above"]) == 2
    assert "rank" in captured.err and "fraction above p90" in captured.err


def test_emitters_write_one_line_per_update():
    update = {"t": 1.089, "scores": [0.42, 0.93], "latest": [0.4, 0.97], "above": [False, True]}
    as_json, as_text = io.StringIO(), io.StringIO()
    score.json_emitter(as_json)(update)
    score.line_emitter(as_text)(update)
    assert json.loads(as_json.getvalue()) == update and as_json.getvalue().endswith("\n")
    text = as_text.getvalue()
    assert text.count("\n") == 1
    assert "d0 0.42 " in text and "d1 0.93*" in text


# -------------------------------------------------------------- source check

FORBIDDEN_WORDS = ("mock", "fake", "fallback", "dummy", "stub", "placeholder", "random", "synthetic")
FORBIDDEN_CALLS = {"rand", "randn", "randint", "uniform", "normal", "default_rng", "zeros_like", "ones_like", "full"}


@pytest.mark.parametrize("name", ["score.py", "run.py"])
def test_no_mock_scoring_path_in_source(name):
    text = (REPO / "dance" / name).read_text()
    lowered = text.lower()
    for word in FORBIDDEN_WORDS:
        assert word not in lowered, f"dance/{name} contains '{word}'"

    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "random" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "random"
        if isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_CALLS, f"dance/{name} uses .{node.attr}"
        # Only FileNotFoundError, Ctrl-C and a closed pipe are caught. No handler
        # can replace the model by something else.
        if isinstance(node, ast.ExceptHandler):
            caught = ast.unparse(node.type) if node.type is not None else "bare except"
            assert caught in {
                "FileNotFoundError", "BrokenPipeError", "(KeyboardInterrupt, BrokenPipeError)",
            }, f"dance/{name} catches {caught}"


def test_every_score_comes_from_model_score():
    """`model.score` is called in one place. `load_model` is the only loader."""
    text = (REPO / "dance" / "score.py").read_text()
    assert text.count("model.score(") == 1
    assert text.count("from dance.spiky import load_model") == 1
    run_text = (REPO / "dance" / "run.py").read_text()
    assert ".score(" not in run_text.replace("score.score_clip(", "")

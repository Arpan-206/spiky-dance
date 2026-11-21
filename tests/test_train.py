"""Tests for dance.train and dance.spiky, on synthetic feature arrays only.

The network is `SparseSpikyNet`: Mr. Spiky's network with adaptive thresholds,
top-k inhibition and row-normalised weights (see dance/spiky/sparse.py).

KNOWN LIMIT, measured (see `test_separation_numbers`): the score counts only
activity above the baseline. A freeze (no events) scores low, not high.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import dance.spiky
from dance.contracts import FEATURE_DIM
from dance.spiky import ARTEFACT_FILES, Model, load_model
from dance.spiky.encode import encode_rate, fit_whitening, normalize, whiten
from dance.spiky.model import (
    SpikyNet,
    ecdf_rescale,
    ecdf_rescale_tensor,
    forward_stateful,
    init_state,
)
from dance.spiky.encode import BASE_FEATURES
from dance.spiky.sparse import TOPK_FRACTION, SparseSpikyNet, normalize_rows
from dance.train import ACTIVITY_TARGET, DegenerateCalibrationError, find_clips, main
from dance.train import train as _train
import dance.train as train_module

FAST = 3000  # threshold settle steps in the tests that do not check the activity


def train(clips, out, **kwargs):
    kwargs.setdefault("settle_steps", FAST)
    _train(clips, out, **kwargs)


def add_deltas(base: np.ndarray) -> np.ndarray:
    """13 base features (bins, dancers, 13) -> the contract layout with deltas."""
    base = base.astype(np.float32)
    if FEATURE_DIM == BASE_FEATURES:
        return base
    delta = np.zeros_like(base)
    delta[1:] = base[1:] - base[:-1]
    return np.concatenate([base, delta], axis=-1)


# ---- Synthetic features ------------------------------------------------------

def random_clip(bins: int, dancers: int, seed: int) -> np.ndarray:
    """Random features with the value ranges of dance.contracts."""
    rng = np.random.default_rng(seed)
    f = np.zeros((bins, dancers, BASE_FEATURES), dtype=np.float32)
    f[..., 0] = rng.uniform(0, 1, (bins, dancers))
    f[..., 1] = rng.uniform(-1, 1, (bins, dancers))
    f[..., 2:4] = rng.uniform(0, 1, (bins, dancers, 2))
    f[..., 4:] = rng.dirichlet(np.ones(9) * 0.5, (bins, dancers))
    return add_deltas(f)


def periodic_base(bins: int, dancers: int, seed: int, period: float = 30.0) -> np.ndarray:
    """Smooth periodic "normal" motion, 13 base features (no deltas)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((bins, dancers, BASE_FEATURES), dtype=np.float32)
    k = np.arange(bins)
    centers = (np.arange(3) + 0.5) / 3
    for d in range(dancers):
        a = 2 * np.pi * k / period + rng.uniform(0, 2 * np.pi)
        cx = 0.5 + 0.25 * np.sin(a)
        cy = 0.5 + 0.2 * np.sin(2 * a)
        out[:, d, 0] = 0.45 + 0.15 * np.sin(2 * a) + rng.normal(0, 0.02, bins)
        out[:, d, 1] = 0.2 * np.sin(a) + rng.normal(0, 0.05, bins)
        out[:, d, 2] = cx + rng.normal(0, 0.01, bins)
        out[:, d, 3] = cy + rng.normal(0, 0.01, bins)
        gx = np.exp(-((cx[:, None] - centers[None]) ** 2) / 0.05)
        gy = np.exp(-((cy[:, None] - centers[None]) ** 2) / 0.05)
        g = (gy[:, :, None] * gx[:, None, :]).reshape(bins, 9) + rng.uniform(0, 0.05, (bins, 9))
        out[:, d, 4:] = g / g.sum(1, keepdims=True)
    out[..., 0] = out[..., 0].clip(0, 1)
    out[..., 1] = out[..., 1].clip(-1, 1)
    out[..., 2:4] = out[..., 2:4].clip(0, 1)
    return out


def periodic_clip(bins: int, dancers: int, seed: int) -> np.ndarray:
    return add_deltas(periodic_base(bins, dancers, seed))


def with_jumps(clip: np.ndarray, starts, length: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Sudden jumps in 13 base features: a burst of event rate, a different grid."""
    c = clip.copy()
    mask = np.zeros(clip.shape[:2], dtype=bool)
    grid = np.full(9, 0.02, dtype=np.float32)
    grid[2] = 0.84
    for s in starts:
        c[s:s + length, :, 0] = 0.98
        c[s:s + length, :, 1] = 0.8
        c[s:s + length, :, 2] = 0.9
        c[s:s + length, :, 3] = 0.1
        c[s:s + length, :, 4:] = grid
        mask[s:s + length] = True
    return add_deltas(c), mask


def with_freeze(clip: np.ndarray, starts, length: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Freeze in 13 base features: empty bins (no events), as in dance.contracts."""
    c = clip.copy()
    mask = np.zeros(clip.shape[:2], dtype=bool)
    for s in starts:
        c[s:s + length] = 0.0
        c[s:s + length, :, 2:4] = 0.5
        mask[s:s + length] = True
    return add_deltas(c), mask


RANDOM_CLIPS = [random_clip(60, 1, 1), random_clip(45, 3, 2), random_clip(7, 2, 3), random_clip(1, 1, 4)]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> Path:
    """A small model, trained on random clips with different dancer counts and lengths."""
    out = tmp_path_factory.mktemp("model")
    train(RANDOM_CLIPS, out, epochs=1, seed=0, regions=["strips"] * len(RANDOM_CLIPS))
    return out


@pytest.fixture(scope="module")
def model(model_dir: Path) -> Model:
    return load_model(model_dir)


# ---- Artefacts and load_model ------------------------------------------------

def test_artefacts_are_written(model_dir: Path):
    for name in ARTEFACT_FILES:
        assert (model_dir / name).is_file(), name
    assert set(ARTEFACT_FILES) == {
        "snn_weights.pt", "whitening.pt", "snn_baselines.pt", "snn_ecdf.pt", "threshold.json",
    }
    meta = json.loads((model_dir / "threshold.json").read_text())
    assert meta["threshold"] == 0.90
    assert meta["n_clips"] == len(RANDOM_CLIPS)
    assert meta["n_samples"] == sum(c.shape[0] * c.shape[1] for c in RANDOM_CLIPS)
    assert meta["regions"] == ["strips"]
    assert meta["n_distinct_raw_scores"] > 1
    assert meta["feature_dim"] == FEATURE_DIM
    assert "method" in meta and "normal_stats" in meta and "weights" in meta
    assert set(meta["activity"]) >= {"hidden", "output"}


def test_load_model_missing_folder(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "no_such_folder")


def test_load_model_empty_folder(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as err:
        load_model(tmp_path)
    for name in ARTEFACT_FILES:
        assert name in str(err.value)


@pytest.mark.parametrize("missing", ARTEFACT_FILES)
def test_load_model_one_artefact_missing(model_dir: Path, tmp_path: Path, missing: str):
    for name in ARTEFACT_FILES:
        if name != missing:
            (tmp_path / name).write_bytes((model_dir / name).read_bytes())
    with pytest.raises(FileNotFoundError) as err:
        load_model(tmp_path)
    assert missing in str(err.value)


def test_no_mock_scoring_path():
    """The vendored package has no mock mode, no path-based loader, no infer module."""
    pkg = Path(dance.spiky.__file__).parent
    names = {p.name for p in pkg.glob("*.py")}
    assert "infer.py" not in names
    for path in list(pkg.glob("*.py")) + [pkg.parent / "train.py"]:
        text = path.read_text().lower()
        assert "mock" not in text, path
        assert "def _load_whitening" not in text, path
        assert "_whitening_cache" not in text, path
    assert not hasattr(dance.spiky, "mock_score")


def test_whiten_needs_an_explicit_transform():
    with pytest.raises(TypeError):
        whiten(torch.zeros(2, FEATURE_DIM))  # type: ignore[call-arg]


# ---- Score contract ----------------------------------------------------------

def test_score_shape_dtype_range(model: Model):
    f = random_clip(50, 3, 10)
    s = model.score(f)
    assert s.shape == (50, 3)
    assert s.dtype == np.float32
    assert np.isfinite(s).all()
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_score_is_deterministic(model: Model, model_dir: Path):
    f = random_clip(40, 2, 11)
    a = model.score(f)
    assert np.array_equal(a, model.score(f))
    assert np.array_equal(a, load_model(model_dir).score(f))


def test_dancers_have_separate_membranes(model: Model):
    f = random_clip(40, 3, 12)
    together = model.score(f)
    for d in range(3):
        assert np.array_equal(together[:, d:d + 1], model.score(f[:, d:d + 1]))
    changed = f.copy()
    changed[:, 0] = random_clip(40, 1, 99)[:, 0]
    assert np.array_equal(model.score(changed)[:, 1:], together[:, 1:])


def test_membrane_is_carried_across_bins(model: Model):
    """The raw score of a bin depends on the bins before it."""
    f = random_clip(40, 1, 13)
    whole = model.raw_score(f)
    tail = model.raw_score(f[20:])
    assert not np.allclose(whole[20:], tail)


def test_no_nan_on_all_zero_bins(model: Model):
    empty = np.zeros((30, 2, FEATURE_DIM), dtype=np.float32)
    empty[..., 2:4] = 0.5          # the contract's empty bin
    all_zero = np.zeros((30, 2, FEATURE_DIM), dtype=np.float32)
    for f in (empty, all_zero):
        s = model.score(f)
        assert s.shape == (30, 2) and np.isfinite(s).all()
        assert s.min() >= 0.0 and s.max() <= 1.0


def test_short_and_empty_clips(model: Model):
    assert model.score(random_clip(1, 1, 14)).shape == (1, 1)
    out = model.score(np.zeros((0, 2, FEATURE_DIM), dtype=np.float32))
    assert out.shape == (0, 2) and out.dtype == np.float32


def test_score_rejects_bad_input(model: Model):
    with pytest.raises(ValueError):
        model.score(np.zeros((5, 2, FEATURE_DIM - 1), dtype=np.float32))
    bad = random_clip(5, 1, 15)
    bad[2, 0, 0] = np.nan
    with pytest.raises(ValueError):
        model.score(bad)


def test_score_speed(model: Model):
    f = random_clip(910, 3, 16)
    t0 = time.perf_counter()
    model.score(f)
    assert time.perf_counter() - t0 < 5.0


# ---- Training ----------------------------------------------------------------

def test_train_with_empty_bins_in_corpus(tmp_path: Path):
    clip = random_clip(40, 2, 20)
    clip[10:25] = 0.0
    clip[10:25, :, 2:4] = 0.5
    train([clip, random_clip(30, 1, 21)], tmp_path, epochs=1)
    s = load_model(tmp_path).score(clip)
    assert np.isfinite(s).all()


def test_train_rejects_bad_input(tmp_path: Path):
    with pytest.raises(ValueError):
        train([], tmp_path)
    with pytest.raises(ValueError):
        train([np.zeros((10, 1, 5), dtype=np.float32)], tmp_path)


def test_train_does_not_change_global_random_state(tmp_path: Path):
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    train([random_clip(20, 1, 23)], tmp_path, epochs=1)
    assert torch.equal(torch.rand(3), expected)


def test_train_is_repeatable(tmp_path: Path):
    clips = [random_clip(30, 2, 24)]
    train(clips, tmp_path / "a", epochs=1, seed=5)
    train(clips, tmp_path / "b", epochs=1, seed=5)
    f = random_clip(25, 2, 25)
    assert np.array_equal(load_model(tmp_path / "a").score(f), load_model(tmp_path / "b").score(f))


def test_weights_after_stdp(tmp_path: Path):
    """Weights are in 0..1, not all zero, not all at 1, and STDP changed them."""
    train(RANDOM_CLIPS, tmp_path, epochs=1, seed=0)
    net = load_model(tmp_path).net
    torch.manual_seed(0)
    init = SparseSpikyNet(net.fc1.in_features)
    for name in ("fc1", "fc2"):
        w = getattr(net, name).weight.data
        assert torch.isfinite(w).all()
        assert w.min() >= 0.0 and w.max() <= 1.0
        assert (w > 1e-6).any(), f"{name}: all weights are zero"
        assert (w < 1.0 - 1e-6).any(), f"{name}: all weights are at 1"
        assert not torch.allclose(w, getattr(init, name).weight.data), f"{name}: no STDP update"


def test_degenerate_calibration_is_loud(tmp_path: Path, monkeypatch):
    """One distinct raw score in calibration: train must raise and write no artefact."""
    monkeypatch.setattr(
        train_module.cal, "corpus_raw_scores",
        lambda net, clips, *a, **k: torch.zeros(sum(c.shape[0] * c.shape[1] for c in clips)),
    )
    with pytest.raises(DegenerateCalibrationError) as err:
        train(RANDOM_CLIPS, tmp_path / "m", epochs=1)
    assert "raw scores of the normal clips are equal" in str(err.value)
    assert not (tmp_path / "m").exists()


def test_activity_is_in_target(tmp_path: Path, capsys):
    """With the default settings, 5 to 15 % of the neurons fire per bin on the training clips."""
    clips = [periodic_clip(300, 1 + i % 3, i) for i in range(4)]
    _train(clips, tmp_path, epochs=2)
    meta = json.loads((tmp_path / "threshold.json").read_text())
    act = meta["activity"]
    with capsys.disabled():
        print(f"\n[activity on periodic training clips] hidden {act['hidden']:.4f}, output {act['output']:.4f}")
    lo, hi = ACTIVITY_TARGET
    assert lo <= act["hidden"] <= hi and lo <= act["output"] <= hi
    assert meta["activity_in_target"] is True


# ---- The sparse network --------------------------------------------------------

def test_sparse_net_run_in_parts_matches_forward():
    torch.manual_seed(0)
    net = SparseSpikyNet().eval()
    x = torch.rand(40, 3, FEATURE_DIM)
    ref = net(x)
    mem1, mem2 = net.init_state(3)
    parts = []
    for start, end in ((0, 7), (7, 25), (25, 40)):
        out, mem1, mem2 = net.run(x[start:end], mem1, mem2)
        parts.append(out)
    for field in ("hidden_spikes", "output_spikes", "hidden_mem", "output_mem"):
        assert torch.equal(torch.cat([getattr(p, field) for p in parts]), getattr(ref, field)), field


def test_sparse_net_topk_and_homeostasis():
    torch.manual_seed(0)
    net = SparseSpikyNet().eval()
    x = torch.rand(200, 2, FEATURE_DIM)
    before = net.theta1.clone()
    out = net(x)
    assert torch.equal(net.theta1, before), "scoring must not change the thresholds"
    assert out.hidden_spikes.sum(dim=2).max() <= round(TOPK_FRACTION * out.hidden_spikes.shape[2])
    assert out.output_spikes.sum(dim=2).max() <= round(TOPK_FRACTION * out.output_spikes.shape[2])
    net.run(x, *net.init_state(2), adapt=True)
    assert not torch.equal(net.theta1, before), "training must adapt the thresholds"


def test_normalize_rows():
    w = torch.rand(8, FEATURE_DIM) * 0.5
    normalize_rows(w)
    assert torch.allclose(w.sum(dim=1), torch.full((8,), 0.25 * FEATURE_DIM), atol=1e-4)
    assert w.min() >= 0.0 and w.max() <= 1.0


# ---- Vendored code -----------------------------------------------------------

def test_forward_stateful_matches_forward():
    torch.manual_seed(0)
    net = SpikyNet().eval()
    x = (torch.rand(40, 3, FEATURE_DIM) < 0.5).float()
    with torch.no_grad():
        ref = net(x)
        whole, _, _ = forward_stateful(net, x, *init_state(net))
        mem1, mem2 = init_state(net)
        parts = []
        for start in (0, 7, 25):
            end = {0: 7, 7: 25, 25: 40}[start]
            out, mem1, mem2 = forward_stateful(net, x[start:end], mem1, mem2)
            parts.append(out)
    for field in ("hidden_spikes", "output_spikes", "hidden_mem", "output_mem"):
        assert torch.equal(getattr(whole, field), getattr(ref, field)), field
        joined = torch.cat([getattr(p, field) for p in parts])
        assert torch.equal(joined, getattr(ref, field)), field


def test_ecdf_tensor_matches_original():
    ref = torch.tensor([0.0, 0.1, 0.1, 0.2, 0.5, 0.9]).sort().values
    raw = [-1.0, 0.0, 0.05, 0.1, 0.3, 0.9, 2.0]
    assert ecdf_rescale_tensor(torch.tensor(raw), ref).tolist() == pytest.approx(ecdf_rescale(raw, ref))


def test_normalize_and_whitening():
    f = random_clip(200, 2, 30)
    n = normalize(f)
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert torch.allclose(n[..., 1], torch.as_tensor((f[..., 1] + 1) / 2), atol=1e-6)
    if FEATURE_DIM > BASE_FEATURES:
        k = BASE_FEATURES + 1  # the polarity delta, -2..2
        assert torch.allclose(n[..., k], torch.as_tensor((f[..., k] + 2) / 4), atol=1e-6)
        other = [i for i in range(BASE_FEATURES, FEATURE_DIM) if i != k]
        assert torch.allclose(n[..., other], torch.as_tensor((f[..., other] + 1) / 2), atol=1e-6)
        no_change = normalize(np.zeros((1, 1, FEATURE_DIM), dtype=np.float32))
        assert torch.all(no_change[..., BASE_FEATURES:] == 0.5)
    t = fit_whitening(n.reshape(-1, FEATURE_DIM))
    w = whiten(n, t)
    assert torch.isfinite(w).all() and w.min() >= 0.0 and w.max() <= 1.0
    assert abs(float(w.mean()) - 0.5) < 0.05
    spikes = encode_rate(f, t, 4)
    assert spikes.shape == (800, 2, FEATURE_DIM)
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}


# ---- Command line ------------------------------------------------------------

def test_cli_fails_without_clips(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("not a clip")
    with pytest.raises(SystemExit) as err:
        main([str(tmp_path), "--out", str(tmp_path / "model")])
    assert "no clips" in str(err.value)
    assert not (tmp_path / "model").exists()
    with pytest.raises(SystemExit):
        main([str(tmp_path / "missing"), "--out", str(tmp_path / "model")])


def test_find_clips(tmp_path: Path):
    for name in ("b.mp4", "a.MOV", "c.h5", "c.mp4", "c.poses.npy", "readme.md"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in find_clips(tmp_path)] == ["a.MOV", "b.mp4", "c.h5"]


# ---- Separation on synthetic sequences (KNOWN FAULT: see the module docstring) ----

def _fractions(scores: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    s = scores if mask is None else scores[mask]
    return float(s.mean()), float((s >= 0.90).mean())


@pytest.mark.xfail(
    strict=False,
    reason="known failure of the opt-in Mr. Spiky encoding (shift): each input is 0.5 plus a "
           "small signal and all weights are positive, so a value below the mean lowers the "
           "current and the score counts only activity above the baseline",
)
def test_separation_numbers(tmp_path: Path, capsys):
    """Mr. Spiky encoding. Train on smooth periodic clips. Compare held-out normal, jumps and freezes."""
    clips = [periodic_clip(300, 1 + i % 3, i) for i in range(6)]
    _train(clips, tmp_path, encoding="shift")
    model = load_model(tmp_path)
    held_base = periodic_base(600, 3, 999)
    held = add_deltas(held_base)
    jumps, jump_mask = with_jumps(held_base, range(100, 600, 100))
    freeze, freeze_mask = with_freeze(held_base, range(100, 600, 100))
    normal = _fractions(model.score(held))
    jump_in = _fractions(model.score(jumps), jump_mask)
    freeze_in = _fractions(model.score(freeze), freeze_mask)
    meta = json.loads((tmp_path / "threshold.json").read_text())
    with capsys.disabled():
        print("\n[periodic clips]")
        print("| Sequence | Mean score | Fraction of bins >= 0.90 |")
        print("|---|---|---|")
        print(f"| Held-out normal | {normal[0]:.3f} | {normal[1]:.3f} |")
        print(f"| Jump bins | {jump_in[0]:.3f} | {jump_in[1]:.3f} |")
        print(f"| Freeze bins | {freeze_in[0]:.3f} | {freeze_in[1]:.3f} |")
        print(f"weights: {meta['weights']}")
        print(f"raw scores of normal clips: {meta['normal_stats']}")
        print(f"activity: {meta['activity']}")
    assert jump_in[1] > 0.5 and jump_in[1] > 3 * normal[1]
    assert freeze_in[1] > 0.5 and freeze_in[1] > 3 * normal[1]


# ---- The two input encodings: "onoff" (default) and "shift" (Mr. Spiky, opt-in) ----

def test_default_encoding_is_onoff(model_dir: Path):
    assert json.loads((model_dir / "threshold.json").read_text())["encoding"] == "onoff"
    assert load_model(model_dir).net.fc1.in_features == 2 * FEATURE_DIM


def test_shift_encoding_of_mr_spiky_is_opt_in(tmp_path: Path):
    train(RANDOM_CLIPS, tmp_path, epochs=1, encoding="shift")
    model = load_model(tmp_path)
    assert model.net.fc1.in_features == FEATURE_DIM
    assert model.meta["encoding"] == "shift"
    s = model.score(random_clip(50, 3, 40))
    assert s.shape == (50, 3) and s.min() >= 0.0 and s.max() <= 1.0


def test_unknown_encoding_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        train([random_clip(10, 1, 22)], tmp_path, encoding="other")


def test_onoff_encoding_contract(tmp_path: Path):
    train(RANDOM_CLIPS, tmp_path, epochs=1, encoding="onoff")
    model = load_model(tmp_path)
    assert model.net.fc1.in_features == 2 * FEATURE_DIM
    assert model.meta["encoding"] == "onoff"
    f = random_clip(50, 3, 40)
    s = model.score(f)
    assert s.shape == (50, 3) and s.dtype == np.float32
    assert np.isfinite(s).all() and s.min() >= 0.0 and s.max() <= 1.0
    assert np.array_equal(s, load_model(tmp_path).score(f))
    empty = np.zeros((30, 2, FEATURE_DIM), dtype=np.float32)
    assert np.isfinite(model.score(empty)).all()


def test_onoff_separation_numbers(tmp_path: Path, capsys):
    """With "onoff", jumps and freezes on synthetic sequences score above the threshold."""
    clips = [periodic_clip(300, 1 + i % 3, i) for i in range(6)]
    _train(clips, tmp_path, encoding="onoff")
    model = load_model(tmp_path)
    held_base = periodic_base(600, 3, 999)
    jumps, jump_mask = with_jumps(held_base, range(100, 600, 100))
    freeze, freeze_mask = with_freeze(held_base, range(100, 600, 100))
    normal = _fractions(model.score(add_deltas(held_base)))
    jump_in = _fractions(model.score(jumps), jump_mask)
    freeze_in = _fractions(model.score(freeze), freeze_mask)
    with capsys.disabled():
        print("\n[periodic clips, encoding onoff]")
        print("| Sequence | Mean score | Fraction of bins >= 0.90 |")
        print("|---|---|---|")
        print(f"| Held-out normal | {normal[0]:.3f} | {normal[1]:.3f} |")
        print(f"| Jump bins | {jump_in[0]:.3f} | {jump_in[1]:.3f} |")
        print(f"| Freeze bins | {freeze_in[0]:.3f} | {freeze_in[1]:.3f} |")
        print(f"activity: {model.meta['activity']}")
    assert normal[1] < 0.2
    assert jump_in[1] > 0.8 and freeze_in[1] > 0.8

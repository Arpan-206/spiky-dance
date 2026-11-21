"""End-to-end test: synthetic clips -> events -> features -> train -> score.

The test uses the contracts only (`dance/contracts.py`):
    dance.train.train(feature_clips, out_dir)
    dance.spiky.load_model(model_dir).score(features)

All clips are written to pytest's temporary folders. No video stays in the repo.
Run with `-s` to see the printed numbers:
    uv run pytest tests/test_end_to_end.py -s

Margins for "clearly higher". They were set before any model number was seen:
    fraction of bins at or above 0.90 >= 2 x the fraction of the held-out normal clip
    fraction of bins at or above 0.90 >= the fraction of the held-out normal clip + 0.05
    mean score                        >= the mean score of the held-out normal clip + 0.05
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dance.contracts import FEATURE_DIM
from dance.features import clip_features
from tests.fixtures import make_synthetic as ms

THRESHOLD = 0.90            # the p90 threshold of the contract
MIN_FRACTION_RATIO = 2.0    # abnormal fraction >= ratio * normal fraction
MIN_FRACTION_GAP = 0.05     # and abnormal fraction >= normal fraction + gap
MIN_MEAN_GAP = 0.05         # and abnormal mean >= normal mean + gap

# name -> (kinds, seed, seconds). The training clips are all normal.
# They differ in length and in the number of dancers, as the contract permits.
TRAIN_PLAN = {
    "train_1": (("normal",), 1, 8.0),
    "train_2": (("normal",), 2, 7.0),
    "train_3": (("normal",), 3, 8.0),
    "train_4": (("normal",), 4, 6.0),
    "train_5": (("normal",), 5, 8.0),
    "train_6": (("normal",), 6, 7.0),
    "train_duo": (("normal", "normal"), 7, 8.0),
}
EVAL_PLAN = {
    "normal": (("normal",), 100, 8.0),
    "jump": (("jump",), 101, 8.0),
    "freeze": (("freeze",), 102, 8.0),
    "trio": (("normal", "jump", "normal"), 103, 8.0),
    "still": (("still",), 104, 4.0),
}
TRIO_ABNORMAL_DANCER = 1    # not dancer 0: argmax gives 0 when all scores are equal


@dataclass
class Sample:
    clip: ms.SyntheticClip
    features: np.ndarray    # (bins, dancers, 13)
    truth: np.ndarray       # bool (bins, dancers); True = abnormal


@pytest.fixture(scope="module")
def samples(tmp_path_factory) -> dict[str, Sample]:
    """All clips of this file, with their features. Made one time."""
    folder = tmp_path_factory.mktemp("clips")
    out = {}
    for name, (kinds, seed, seconds) in {**TRAIN_PLAN, **EVAL_PLAN}.items():
        clip = ms.make_clip(folder / f"{name}.mp4", kinds, seed, seconds)
        features, _ = clip_features(clip.path, dancers=clip.dancers)
        out[name] = Sample(clip, features, clip.abnormal_bins(len(features)))
    return out


@pytest.fixture(scope="module", params=["spiking", "spiking-shift", "baseline"])
def model(request, samples, tmp_path_factory):
    """Train one time for each scorer, then load. Both scorers have the same contract."""
    clips = [samples[name].features for name in TRAIN_PLAN]
    out_dir = tmp_path_factory.mktemp(f"models_{request.param}")
    if request.param == "baseline":
        from dance.baseline import load_model, train
        train(clips, out_dir)
    else:
        from dance.spiky import load_model
        from dance.train import train
        if request.param == "spiking":
            train(clips, out_dir)                       # the contract call: on/off encoding, the default
        else:
            train(clips, out_dir, encoding="shift")     # the opt-in input encoding of Mr. Spiky
    print(f"\n=== scorer: {request.param} ===")
    loaded = load_model(out_dir)
    loaded.scorer_name = request.param
    return loaded


SPIKING_KNOWN_FAULT = (
    "Known failure of the opt-in Mr. Spiky input encoding (shift): each input is 0.5 plus a "
    "small signal and all weights are positive. A value below the mean lowers the current, so "
    "the score counts only activity above the normal level. It does not detect a freeze or a "
    "jump. See the README."
)


def known_spiking_fault(request, model) -> None:
    """Mark the test as a known failure for the Mr. Spiky encoding. The numbers are still printed."""
    if model.scorer_name == "spiking-shift":
        request.applymarker(pytest.mark.xfail(reason=SPIKING_KNOWN_FAULT, strict=False))


def final_numbers(scores: np.ndarray) -> tuple[float, float]:
    """(mean score, fraction of bins with score >= THRESHOLD)."""
    scores = np.asarray(scores)
    return float(scores.mean()), float((scores >= THRESHOLD).mean())


def relative_rate(features: np.ndarray) -> np.ndarray:
    """The `rate` feature (index 0): event count against the dancer's own running mean.

    0.5 is the own mean. 0 is an empty bin.
    """
    return features[..., 0]


def report(name: str, sample: Sample, scores: np.ndarray) -> None:
    mean, fraction = final_numbers(scores)
    print(f"\n{name}: mean score {mean:.3f}, fraction >= {THRESHOLD:.2f}: {fraction:.3f}")
    for d in range(scores.shape[1]):
        truth = sample.truth[:, d]
        d_mean, d_fraction = final_numbers(scores[:, d])
        line = f"  dancer {d} ({sample.clip.kinds[d]}): mean {d_mean:.3f}, fraction {d_fraction:.3f}"
        if truth.any() and not truth.all():
            line += (f", mean inside abnormal bins {scores[truth, d].mean():.3f}"
                     f", outside {scores[~truth, d].mean():.3f}")
        print(line)


def assert_clearly_higher(abnormal: np.ndarray, normal: np.ndarray, what: str) -> None:
    a_mean, a_fraction = final_numbers(abnormal)
    n_mean, n_fraction = final_numbers(normal)
    need_fraction = max(MIN_FRACTION_RATIO * n_fraction, n_fraction + MIN_FRACTION_GAP)
    assert a_fraction >= need_fraction, (
        f"{what}: fraction >= {THRESHOLD} is {a_fraction:.3f}, need >= {need_fraction:.3f} "
        f"(held-out normal: {n_fraction:.3f})"
    )
    assert a_mean >= n_mean + MIN_MEAN_GAP, (
        f"{what}: mean score is {a_mean:.3f}, need >= {n_mean + MIN_MEAN_GAP:.3f} "
        f"(held-out normal: {n_mean:.3f})"
    )


# ---------------------------------------------------------------------------
# The fixtures. These tests do not need the model.
# ---------------------------------------------------------------------------

def test_clips_decode_with_opencv(samples):
    for name, sample in samples.items():
        clip = sample.clip
        frames = ms.read_clip(clip.path)
        assert frames.shape == (clip.n_frames, ms.HEIGHT, ms.STRIP_W * clip.dancers), name
        assert clip.abnormal.shape == (clip.n_frames, clip.dancers), name
        assert frames.max() > 150 and np.median(frames) < 60, name


def test_fixture_is_deterministic_and_seeded():
    kinds = ("jump", "freeze")
    frames_a, truth_a = ms.render_clip(kinds, seed=9, seconds=3.0)
    frames_b, truth_b = ms.render_clip(kinds, seed=9, seconds=3.0)
    frames_c, _ = ms.render_clip(kinds, seed=10, seconds=3.0)
    assert np.array_equal(frames_a, frames_b) and np.array_equal(truth_a, truth_b)
    assert not np.array_equal(frames_a, frames_c)


def test_ground_truth_marks_the_right_dancer(samples):
    for name in TRAIN_PLAN:
        assert not samples[name].clip.abnormal.any(), name
    assert not samples["normal"].clip.abnormal.any()
    # A substantial part of an abnormal clip is abnormal, not one single frame.
    assert 0.25 <= samples["jump"].clip.abnormal.mean() <= 0.60
    assert 0.25 <= samples["freeze"].clip.abnormal.mean() <= 0.60
    per_dancer = samples["trio"].clip.abnormal.any(axis=0)
    assert per_dancer.tolist() == [d == TRIO_ABNORMAL_DANCER for d in range(3)]


def test_features_have_the_contract_shape(samples):
    for name, sample in samples.items():
        f = sample.features
        assert f.ndim == 3 and f.shape[1:] == (sample.clip.dancers, FEATURE_DIM), name
        assert abs(f.shape[0] - sample.clip.n_frames) <= 3, name
        assert f.dtype == np.float32 and np.isfinite(f).all(), name
        assert sample.truth.shape == f.shape[:2], name


def test_static_clip_gives_no_events(samples):
    """The dark grey background keeps compression noise below the event threshold."""
    rate = relative_rate(samples["still"].features)
    print(f"\nstill clip: {(rate > 0).sum()} bins with events of {rate.size}")
    assert (rate > 0).mean() < 0.05


def test_abnormal_frames_show_in_the_raw_features(samples):
    normal = relative_rate(samples["normal"].features)[:, 0]
    jump, jump_truth = relative_rate(samples["jump"].features)[:, 0], samples["jump"].truth[:, 0]
    freeze, freeze_truth = relative_rate(samples["freeze"].features)[:, 0], samples["freeze"].truth[:, 0]
    print(f"\nrate, normal clip: mean {normal.mean():.3f}, max {normal.max():.3f}")
    print(f"rate, jump clip: inside {jump[jump_truth].mean():.3f}, outside {jump[~jump_truth].mean():.3f}, "
          f"max inside {jump[jump_truth].max():.3f}")
    print(f"rate, freeze clip: inside {freeze[freeze_truth].mean():.3f}, "
          f"outside {freeze[~freeze_truth].mean():.3f}, "
          f"empty bins inside {(freeze[freeze_truth] == 0).mean():.2f}")
    # Jump: bursts far above the dancer's own mean, higher than any normal bin.
    assert jump[jump_truth].mean() > jump[~jump_truth].mean()
    assert jump[jump_truth].max() > normal.max()
    # Freeze: almost no events inside the abnormal bins.
    assert freeze[freeze_truth].mean() < 0.25 * freeze[~freeze_truth].mean()
    assert (freeze[freeze_truth] == 0).mean() > 0.6
    assert (freeze[~freeze_truth] == 0).mean() < 0.05


# ---------------------------------------------------------------------------
# The model. These tests need `dance.train` and `dance.spiky`.
# ---------------------------------------------------------------------------

def test_scores_follow_the_contract(samples, model):
    for name in ("normal", "trio"):
        features = samples[name].features
        scores = model.score(features)
        assert isinstance(scores, np.ndarray), name
        assert scores.shape == features.shape[:2], name
        assert scores.dtype == np.float32, name
        assert np.isfinite(scores).all() and scores.min() >= 0.0 and scores.max() <= 1.0, name
        assert np.array_equal(scores, model.score(features)), f"{name}: the second call gives other scores"


def test_a_jump_clip_scores_clearly_higher_than_normal(samples, model, request):
    known_spiking_fault(request, model)
    normal = model.score(samples["normal"].features)
    jump = model.score(samples["jump"].features)
    report("held-out normal", samples["normal"], normal)
    report("jump", samples["jump"], jump)
    assert_clearly_higher(jump, normal, "jump clip")


def test_b_freeze_clip_scores_clearly_higher_than_normal(samples, model, request):
    known_spiking_fault(request, model)
    normal = model.score(samples["normal"].features)
    freeze = model.score(samples["freeze"].features)
    report("held-out normal", samples["normal"], normal)
    report("freeze", samples["freeze"], freeze)
    assert_clearly_higher(freeze, normal, "freeze clip")


def test_c_abnormal_dancer_of_three_ranks_first(samples, model, request):
    known_spiking_fault(request, model)
    scores = model.score(samples["trio"].features)
    report("trio, only one dancer jumps", samples["trio"], scores)
    means = scores.mean(axis=0)
    assert scores.shape[1] == 3
    assert int(np.argmax(means)) == TRIO_ABNORMAL_DANCER, f"mean score per dancer: {means.tolist()}"
    assert means[TRIO_ABNORMAL_DANCER] > np.delete(means, TRIO_ABNORMAL_DANCER).max()


def test_d_load_model_on_an_empty_folder_raises(tmp_path):
    from dance.spiky import load_model

    empty = tmp_path / "empty_models"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_model(empty)


def test_e_sync_score_marks_the_dancer_out_of_step(samples, tmp_path):
    """The sync score needs no training. Three dancers move in step; only dancer 1 jumps."""
    from dance.sync import sync_score

    clip = ms.make_clip(tmp_path / "group.mp4", ("normal", "jump", "normal"), seed=105, seconds=8.0, in_step=True)
    features, _ = clip_features(clip.path, dancers=3)
    sync = sync_score(features)
    assert sync.shape == features.shape[:2]
    means = sync.mean(axis=0)
    truth = clip.abnormal_bins(len(features))[:, TRIO_ABNORMAL_DANCER]
    inside = sync[truth, TRIO_ABNORMAL_DANCER].mean()
    outside = sync[~truth, TRIO_ABNORMAL_DANCER].mean()
    print(f"\nsync, three dancers in step: mean per dancer {np.round(means, 3).tolist()}; "
          f"jump dancer inside abnormal bins {inside:.3f}, outside {outside:.3f}")
    assert int(np.argmax(means)) == TRIO_ABNORMAL_DANCER
    assert means[TRIO_ABNORMAL_DANCER] > np.delete(means, TRIO_ABNORMAL_DANCER).max() + 0.05
    assert sync_score(samples["normal"].features) is None      # 1 dancer: no sync score

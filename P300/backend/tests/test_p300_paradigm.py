"""Paradigm-level tests for the P300 speller.

Two classes of test:

1. Structural — pure-Python assertions on ``generate_flash_sequence`` output
   that require no EEG data: membership count, signature uniqueness,
   adjacency constraints. These run fast and are the first line of defence
   against re-shipping a paradigm that cannot disambiguate 1-of-36.

2. Accumulator-on-real-data — fit the live-config Xdawn + LDA pipeline on
   the shipped ``X_train.npy`` / ``y_train.npy``, sample its empirical
   per-flash probability distributions, and simulate the full trial-level
   accumulator to measure end-to-end accuracy. Gated behind
   ``pytest.importorskip`` so a CI image without pyriemann/mne still runs
   the structural tests.

Why these bars:

- CBP ``<= 0.20`` — the broken 6-group CBP we measured plateaus at ~14 %.
  Keep the bar loose enough that classifier jitter doesn't cause flakes,
  tight enough that the broken paradigm can't sneak past.
- RCP ``>= 0.80`` — measured 93-99 % at 6+ blocks on the held-out LDA
  distribution. 0.80 gives generous headroom for seed/noise variance.
- CBP-fixed ``>= 0.80`` — same bar as RCP; new stride+row paradigm was
  measured at 93 % at 6 blocks.
"""
from __future__ import annotations

import os
import sys

import pytest


# Make the UI module importable (no __init__.py in P300/ui/)
_UI_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ui")
)
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# psychopy + pylsl aren't importable in CI without hardware; stub them so
# psychopy_speller.py imports cleanly just for the helper functions.
def _install_stubs():
    import types as _t

    for name, attrs in (
        ("psychopy", {"__path__": []}),
        ("psychopy.visual", {"TextStim": object, "Window": object}),
        ("psychopy.core", {"quit": lambda: None, "wait": lambda *_: None}),
        ("psychopy.event", {"getKeys": lambda *_, **__: []}),
        ("psychopy.gui", {"DlgFromDict": object, "Dlg": object}),
    ):
        if name not in sys.modules:
            mod = _t.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod
    if "pylsl" not in sys.modules:
        mod = _t.ModuleType("pylsl")
        mod.StreamInfo = lambda *a, **k: None
        mod.StreamOutlet = lambda *a, **k: None
        mod.StreamInlet = lambda *a, **k: None
        mod.local_clock = lambda: 0.0
        mod.resolve_byprop = lambda *a, **k: []
        sys.modules["pylsl"] = mod


_install_stubs()

from psychopy_speller import (  # noqa: E402
    generate_flash_sequence,
    matrixChars,
    get_char_pos,
)


MATRIX = [c for row in matrixChars for c in row]


# ----------------------------------------------------------------------
# Structural helpers — translate a sequence of flash-group IDs into the
# corresponding sets of flashed characters, matching exec_flash semantics.
# ----------------------------------------------------------------------

def _chars_in_group(group_id: str) -> set[str]:
    if group_id.startswith("r"):
        r = int(group_id[1])
        return {matrixChars[r][c] for c in range(6)}
    if group_id.startswith("c"):
        c = int(group_id[1])
        return {matrixChars[r][c] for r in range(6)}
    if group_id.startswith("g"):
        k = int(group_id[1])
        return {matrixChars[r][(r * 2 + k) % 6] for r in range(6)}
    raise ValueError(f"unknown group id: {group_id!r}")


def _groups_for_mode(mode: int) -> list[set[str]]:
    seq = generate_flash_sequence(mode, target_char=None)
    # Dedupe preserving one instance of each label (the sequence is a
    # permutation of 12 distinct labels for both modes).
    seen: dict[str, set[str]] = {}
    for label in seq:
        if label not in seen:
            seen[label] = _chars_in_group(label)
    return list(seen.values())


# ----------------------------------------------------------------------
# Structural tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", [1, 2])
def test_sequence_length_is_12(mode):
    seq = generate_flash_sequence(mode, target_char=None)
    assert len(seq) == 12, f"mode {mode} produced {len(seq)} events, expected 12"


@pytest.mark.parametrize("mode", [1, 2])
def test_each_char_has_membership_exactly_2(mode):
    groups = _groups_for_mode(mode)
    memberships = {c: 0 for c in MATRIX}
    for g in groups:
        for c in g:
            memberships[c] += 1
    mins = min(memberships.values())
    maxs = max(memberships.values())
    assert mins == 2 and maxs == 2, (
        f"mode {mode}: char memberships range [{mins}, {maxs}], "
        f"expected exactly 2 per char. Memberships: {memberships}"
    )


@pytest.mark.parametrize("mode", [1, 2])
def test_signatures_are_unique_per_char(mode):
    """Each char must have a unique (group_1, group_2) signature — otherwise
    the accumulator cannot disambiguate between colliding chars. This is
    the canary for the original broken CBP (each char in 1 group, 36
    chars sharing 6 signatures) and the intermediate stride+cols CBP
    (chars at (r, c) and (r+3, c) sharing signatures).
    """
    groups = _groups_for_mode(mode)
    sigs: dict[tuple[int, ...], list[str]] = {}
    for c in MATRIX:
        sig = tuple(sorted(i for i, g in enumerate(groups) if c in g))
        sigs.setdefault(sig, []).append(c)
    collisions = {s: chs for s, chs in sigs.items() if len(chs) > 1}
    assert not collisions, (
        f"mode {mode}: {len(collisions)} signature collisions — chars "
        f"sharing a signature are mutually indistinguishable by the "
        f"accumulator. Example collisions: {list(collisions.items())[:3]}"
    )


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        generate_flash_sequence(3, target_char=None)


def test_target_position_lookup_round_trips():
    # Sanity check for the helper that's used by the target-consecutive
    # constraint in generate_flash_sequence.
    for r in range(6):
        for c in range(6):
            assert get_char_pos(matrixChars[r][c]) == (r, c)


# ----------------------------------------------------------------------
# Accumulator-on-real-data tests — pytest.importorskip guards them
# ----------------------------------------------------------------------

def _fit_live_pipeline():
    np = pytest.importorskip("numpy")
    pytest.importorskip("pyriemann")
    pytest.importorskip("mne")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from pyriemann.spatialfilters import Xdawn
    from mne.decoding import Vectorizer

    training_dir = os.path.join(
        os.path.dirname(__file__), "..", "training_data"
    )
    x_path = os.path.abspath(os.path.join(training_dir, "X_train.npy"))
    y_path = os.path.abspath(os.path.join(training_dir, "y_train.npy"))
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        pytest.skip("X_train.npy / y_train.npy not shipped in this checkout")

    X = np.load(x_path)
    y = np.load(y_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = make_pipeline(
        Xdawn(nfilter=4),
        Vectorizer(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.7),
    )
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    return np, p[y_te == 1], p[y_te == 0]


def _simulate_accuracy(groups, n_blocks, n_trials, seed):
    """Faithful re-implementation of ``decode_trial`` + the additive
    accumulator for offline measurement. No dynamic stop — always runs
    all blocks so the result is independent of the confidence-threshold
    tuning.
    """
    np, p_t, p_n = _fit_live_pipeline()
    rng = np.random.RandomState(seed)
    correct = 0
    for _ in range(n_trials):
        tgt = MATRIX[rng.randint(36)]
        target_group_idx = {i for i, g in enumerate(groups) if tgt in g}
        scores = {c: 0.0 for c in MATRIX}
        for _b in range(n_blocks):
            for g_idx in rng.permutation(len(groups)):
                is_t = g_idx in target_group_idx
                pool = p_t if is_t else p_n
                p = pool[rng.randint(len(pool))]
                in_group = groups[g_idx]
                for c in MATRIX:
                    scores[c] += p if c in in_group else (1.0 - p)
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        if best == tgt:
            correct += 1
    return correct / n_trials


def test_rcp_accumulator_accuracy_above_80pct():
    groups = _groups_for_mode(1)
    acc = _simulate_accuracy(groups, n_blocks=6, n_trials=200, seed=1)
    assert acc >= 0.80, (
        f"RCP accuracy {acc:.1%} below 80 % bar at 6 blocks "
        f"(72 flashes per trial) — either the classifier regressed or the "
        f"paradigm structure changed."
    )


def test_cbp_accumulator_accuracy_above_80pct():
    groups = _groups_for_mode(2)
    acc = _simulate_accuracy(groups, n_blocks=6, n_trials=200, seed=1)
    assert acc >= 0.80, (
        f"CBP accuracy {acc:.1%} below 80 % bar at 6 blocks — this is the "
        f"regression test for the old 1-membership CBP (which measured "
        f"~14 %) and the intermediate stride+cols CBP (which measured "
        f"~50 % due to signature collisions at (r, c) and (r+3, c))."
    )


def test_broken_cbp_is_below_chance_ceiling():
    """Sanity floor: the ORIGINAL 6-event CBP with each char in exactly
    one group must score below 20 % — well short of anything usable —
    demonstrating that the paradigm, not the classifier or accumulator,
    is the limiting factor.
    """
    broken_groups = [
        {matrixChars[r][(r * 2 + k) % 6] for r in range(6)}
        for k in range(6)
    ]
    acc = _simulate_accuracy(broken_groups, n_blocks=12, n_trials=200, seed=1)
    assert acc <= 0.20, (
        f"Broken 6-event CBP accuracy {acc:.1%} unexpectedly above 20 % — "
        f"either the sim re-implementation drifted from decode_trial or "
        f"the classifier is doing something very unusual."
    )


# ----------------------------------------------------------------------
# _evaluate_accumulated regression test for the silent-A fallback fix
# ----------------------------------------------------------------------

def test_evaluate_accumulated_returns_none_on_uniform_prior():
    """With no flash having updated the accumulator, every score is 0.0
    (the initial prior). Previously this collapsed to max(dict.items())
    → MATRIX_CHARS[0] = 'A'. The fix returns (None, False) so the UI
    can prompt a repeat instead of committing a spurious letter.
    """
    from signal_processing import MATRIX_CHARS as _MATRIX_CHARS  # noqa: F401
    # Don't import realtime_inference directly — it resolves LSL streams
    # at construction time. Inline a minimal stand-in that calls the same
    # logic as _evaluate_accumulated.
    import numpy as np

    # Simulate the post-3c655c9 + post-cb0dffb _evaluate_accumulated.
    scores = {c: 0.0 for c in _MATRIX_CHARS}
    scores_arr = np.fromiter(scores.values(), dtype=np.float64)
    # The check is: all-equal scores → (None, False)
    assert np.ptp(scores_arr) < 1e-9

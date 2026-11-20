"""Contract tests for the audio-noise-gate exercise.

Every test exercises a stub, so the unmodified repository is expected to show
15 red tests.  Signals are plain ``list[float]``; levels are dBFS, times are
milliseconds, and floating-point assertions use an absolute tolerance of
``1e-6``.
"""

import math

import pytest

from gate.noisegate import (
    GateError,
    apply_gate,
    estimate_noise_floor,
    gate_batch,
    gate_state,
    hold_counter,
    hysteresis_check,
    level_db,
    sidechain_key,
)


def test_level_db_uses_local_window_not_global_mean():
    """A quiet bed followed by a burst must not share one global mean level."""
    samples = [0.001] * 40 + [0.5] * 40

    levels = level_db(samples, window=8, hop=8)

    assert len(levels) == 10
    for value in levels[:5]:
        assert value == pytest.approx(-60.0, abs=1e-6)
    loud_level = 20.0 * math.log10(0.5)
    for value in levels[5:]:
        assert value == pytest.approx(loud_level, abs=1e-6)


def test_gate_state_hysteresis_prevents_chatter():
    """Levels inside the hysteresis band must not toggle the gate."""
    levels = [51.0, 49.0] * 10

    state = gate_state(
        levels,
        open_db=50.0,
        close_db=40.0,
        attack_ms=1.0,
        release_ms=1.0,
        hold_ms=0.0,
        sample_rate=1000.0,
    )

    assert state[0] is True
    assert all(state)
    transitions = sum(state[i] != state[i - 1] for i in range(1, len(state)))
    assert transitions == 0


def test_apply_gate_smooths_gain_on_switch():
    """Opening the gate must ramp the gain instead of jumping to unity."""
    samples = [0.5] * 200
    state = [False] * 10 + [True] * 190

    output = apply_gate(
        samples,
        state,
        attack_ms=10.0,
        release_ms=50.0,
        sample_rate=1000.0,
    )

    assert output[0] == pytest.approx(0.0, abs=1e-6)
    assert output[10] < 0.25
    max_step = max(abs(output[i] - output[i - 1]) for i in range(1, len(output)))
    assert max_step <= 0.05 + 1e-6
    assert output[-1] == pytest.approx(0.5, abs=1e-6)


def test_apply_gate_uses_distinct_attack_and_release():
    """A fast attack must not force the release to be equally fast."""
    samples = [0.5] * 400
    state = [False] * 20 + [True] * 180 + [False] * 200

    output = apply_gate(
        samples,
        state,
        attack_ms=5.0,
        release_ms=100.0,
        sample_rate=1000.0,
    )

    assert output[25] > 0.4
    assert output[205] > 0.3
    assert output[300] < 0.1


def test_hold_time_scales_with_sample_rate():
    """The same hold in milliseconds spans more frames at a higher rate."""
    levels = [60.0] * 4 + [0.0] * 300 + [60.0] * 4

    state_1k = gate_state(
        levels, 50.0, 40.0, 1.0, 1.0, hold_ms=20.0, sample_rate=1000.0
    )
    state_8k = gate_state(
        levels, 50.0, 40.0, 1.0, 1.0, hold_ms=20.0, sample_rate=8000.0
    )

    open_1k = sum(state_1k[4:304])
    open_8k = sum(state_8k[4:304])
    assert 18 <= open_1k <= 22
    assert 156 <= open_8k <= 164
    assert open_8k > open_1k * 6


def test_sidechain_depth_is_clamped():
    """Depth above one must behave like depth one and never invert phase."""
    samples = [0.5] * 8
    key = [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 1.0, 0.5]

    overdriven = sidechain_key(samples, key, depth=1.5)
    full = sidechain_key(samples, key, depth=1.0)
    bypassed = sidechain_key(samples, key, depth=0.0)

    assert overdriven == pytest.approx(full, abs=1e-6)
    assert min(overdriven) >= 0.0
    assert bypassed == pytest.approx(samples, abs=1e-6)
    assert full[1] == pytest.approx(0.125, abs=1e-6)
    assert full[4] == pytest.approx(0.5, abs=1e-6)


def test_noise_floor_uses_percentile_not_minimum():
    """A single very quiet frame must not drag the estimated floor down."""
    levels = [-100.0] + [-60.0] * 99

    floor = estimate_noise_floor(levels, percentile=10.0)

    assert floor == pytest.approx(-60.0, abs=1e-6)
    assert floor > min(levels)


def test_gate_state_rejects_inverted_thresholds():
    """An open threshold below the close threshold must be rejected."""
    with pytest.raises(GateError):
        gate_state(
            [60.0, 60.0, 60.0],
            open_db=40.0,
            close_db=50.0,
            attack_ms=5.0,
            release_ms=50.0,
            hold_ms=0.0,
            sample_rate=1000.0,
        )


def test_hold_counter_never_goes_negative():
    """Remaining hold frames must clamp at zero after the countdown."""
    state = [True, False, False, False, False, True, False]

    counts = hold_counter(state, hold_frames=2)

    assert counts == [2, 1, 0, 0, 0, 2, 1]
    assert all(count >= 0 for count in counts)


def test_gate_batch_keeps_track_state_independent():
    """A loud track must not leave the gate open for a following quiet track."""
    tracks = [[0.5] * 160, [0.001] * 160]
    params = {
        "open_db": -40.0,
        "close_db": -50.0,
        "attack_ms": 5.0,
        "release_ms": 50.0,
        "hold_ms": 10.0,
        "sample_rate": 1000.0,
        "window": 8,
        "hop": 8,
    }

    result = gate_batch(tracks, params)

    assert result["states"][0][0] is True
    assert not any(result["states"][1])
    assert result["outputs"][0][-1] == pytest.approx(0.5, abs=1e-6)
    assert max(abs(value) for value in result["outputs"][1]) == pytest.approx(
        0.0, abs=1e-6
    )


def test_all_zero_input_stays_closed():
    """An all-zero signal is silence, so the gate stays closed throughout."""
    samples = [0.0] * 160

    levels = level_db(samples, window=8, hop=8)
    state = gate_state(
        levels,
        open_db=-40.0,
        close_db=-60.0,
        attack_ms=5.0,
        release_ms=20.0,
        hold_ms=10.0,
        sample_rate=1000.0,
    )
    output = apply_gate(
        samples, state, attack_ms=5.0, release_ms=20.0, sample_rate=1000.0
    )

    assert levels == pytest.approx([-120.0] * 20, abs=1e-6)
    assert not any(state)
    assert output == pytest.approx([0.0] * 160, abs=1e-6)


def test_zero_hold_closes_immediately():
    """With no hold time the gate closes on the first sub-threshold frame."""
    levels = [60.0, 60.0, 0.0, 0.0, 0.0]

    state = gate_state(
        levels,
        open_db=50.0,
        close_db=40.0,
        attack_ms=1.0,
        release_ms=1.0,
        hold_ms=0.0,
        sample_rate=1000.0,
    )

    assert state[:2] == [True, True]
    assert state[2] is False
    assert state[3:] == [False, False]


def test_percentile_zero_equals_minimum():
    """The zeroth percentile is exactly the smallest level."""
    levels = [-70.0, -50.0, -90.0, -60.0]

    assert estimate_noise_floor(levels, percentile=0.0) == pytest.approx(
        -90.0, abs=1e-6
    )


def test_hysteresis_check_rejects_non_increasing_thresholds():
    """Equal or inverted thresholds raise GateError; a valid pair returns None."""
    with pytest.raises(GateError):
        hysteresis_check(40.0, 40.0)
    with pytest.raises(GateError):
        hysteresis_check(30.0, 40.0)

    assert hysteresis_check(45.0, 40.0) is None


def test_gate_batch_preserves_track_order():
    """A successful batch returns per-track results in input order."""
    tracks = [[0.1] * 80, [0.2] * 80, [0.3] * 80]
    params = {
        "open_db": -60.0,
        "close_db": -70.0,
        "attack_ms": 0.0,
        "release_ms": 0.0,
        "hold_ms": 0.0,
        "sample_rate": 1000.0,
        "window": 8,
        "hop": 8,
    }

    result = gate_batch(tracks, params)

    assert {"levels", "states", "outputs"} <= set(result)
    assert len(result["outputs"]) == 3
    for output, source in zip(result["outputs"], tracks):
        assert output == pytest.approx(source, abs=1e-6)

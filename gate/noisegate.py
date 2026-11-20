"""Noise-gate primitives for the audio-noise-gate exercise.

All levels are dBFS, all time parameters are milliseconds, and floating-point
results are expected to agree within ``1e-6`` unless stated otherwise.
"""

from __future__ import annotations

import math


class GateError(ValueError):
    """Raised when gate parameters or track data are invalid."""


_SILENCE_DB = -120.0
_EPS = 1e-6


def _is_finite(value: object) -> bool:
    """Return ``True`` when *value* is a finite real number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite(value: object, name: str) -> float:
    if not _is_finite(value):
        raise GateError(f"{name} must be a finite number")
    return float(value)


def _non_negative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise GateError(f"{name} must be non-negative")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise GateError(f"{name} must be positive")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise GateError(f"{name} must be a positive integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    else:
        raise GateError(f"{name} must be a positive integer")
    if result <= 0:
        raise GateError(f"{name} must be a positive integer")
    return result


def level_db(samples: list[float], window: int, hop: int) -> list[float]:
    """Compute sliding-window RMS levels in dBFS.

    Frame ``k`` covers ``samples[k * hop : k * hop + window]``; only complete
    windows are emitted, so the frame count is
    ``0 if len(samples) < window else 1 + (len(samples) - window) // hop``.
    ``window`` and ``hop`` are independent: ``hop`` may be smaller (overlap)
    or larger (gaps) than ``window``.

    The level is ``20 * log10(sqrt(mean(x**2)))``.  An all-zero window has an
    RMS of zero and maps to the silence floor ``-120.0`` dBFS rather than
    ``-inf``.

    Args:
        samples: Input signal, normally in ``[-1, 1]``.  Must be a sequence of
            finite floats.
        window: Window length in samples; must be positive.
        hop: Window step in samples; must be positive.

    Returns:
        One dBFS level per complete window, in input order.

    Raises:
        GateError: If ``window`` or ``hop`` is not positive, or if a sample is
            not finite.

    Float results are compared with a tolerance of ``1e-6``.
    """
    win = _positive_int(window, "window")
    step = _positive_int(hop, "hop")
    for sample in samples:
        _finite(sample, "sample")

    if len(samples) < win:
        return []

    levels: list[float] = []
    for start in range(0, len(samples) - win + 1, step):
        window_samples = samples[start : start + win]
        mean_square = sum(float(x) * float(x) for x in window_samples) / win
        if mean_square <= 0.0:
            levels.append(_SILENCE_DB)
        else:
            levels.append(10.0 * math.log10(mean_square))
    return levels


def gate_state(
    levels: list[float],
    open_db: float,
    close_db: float,
    attack_ms: float,
    release_ms: float,
    hold_ms: float,
    sample_rate: float,
) -> list[bool]:
    """Run a two-threshold hold state machine over frame levels.

    Each element of ``levels`` is one analysis frame, and ``sample_rate`` is
    the level-frame rate in frames per second (each frame lasts
    ``1000 / sample_rate`` ms); this is normally the audio sample rate divided
    by the analysis hop.  Millisecond values are converted to frame counts
    with ``round(ms * sample_rate / 1000)``.

    The gate starts closed.  It opens on a frame whose level is greater than
    or equal to ``open_db``.  Once open, it stays open while the level is
    above ``close_db``; when the level reaches or falls below ``close_db`` the
    hold countdown starts, and the gate closes only after ``hold_ms`` worth of
    frames have elapsed below the close threshold.  A level at or above
    ``open_db`` during the countdown cancels the pending close.  The two
    thresholds therefore form hysteresis: a level between ``close_db`` and
    ``open_db`` can never make the state chatter.

    ``attack_ms`` and ``release_ms`` are accepted for API parity with
    :func:`apply_gate`; gain smoothing happens there and they do not change
    the boolean state.  ``hold_ms == 0`` means the gate closes on the first
    frame at or below ``close_db`` with no extra hold frames.

    Args:
        levels: Frame levels in dBFS.
        open_db: Open threshold in dBFS; must be strictly greater than
            ``close_db``.
        close_db: Close threshold in dBFS.
        attack_ms: Attack time in milliseconds; must be non-negative.
        release_ms: Release time in milliseconds; must be non-negative.
        hold_ms: Hold time in milliseconds; must be non-negative.
        sample_rate: Frame rate in frames per second; must be positive.

    Returns:
        A list of booleans with the same length as ``levels``: ``True`` means
        the gate is open on that frame.

    Raises:
        GateError: If ``open_db <= close_db``, if a time is negative, if
            ``sample_rate`` is not positive, or if an input is not finite.

    Threshold comparisons use the ``1e-6`` tolerance.
    """
    open_threshold = _finite(open_db, "open_db")
    close_threshold = _finite(close_db, "close_db")
    hysteresis_check(open_threshold, close_threshold)
    _non_negative(attack_ms, "attack_ms")
    _non_negative(release_ms, "release_ms")
    hold_ms_value = _non_negative(hold_ms, "hold_ms")
    rate = _positive(sample_rate, "sample_rate")
    for level in levels:
        _finite(level, "level")

    hold_frames = int(round(hold_ms_value * rate / 1000.0))

    state: list[bool] = []
    is_open = False
    counter = 0
    for level in levels:
        level = float(level)
        if level >= open_threshold - _EPS:
            is_open = True
            counter = hold_frames
            state.append(True)
        elif is_open:
            if level <= close_threshold + _EPS:
                if counter > 0:
                    state.append(True)
                    counter -= 1
                else:
                    is_open = False
                    state.append(False)
            else:
                # Inside the hysteresis band: signal is still above the close
                # threshold, so the gate stays open and the hold countdown is
                # re-armed for the next time it drops below close_db.
                counter = hold_frames
                state.append(True)
        else:
            state.append(False)
    return state


def apply_gate(
    samples: list[float],
    state: list[bool],
    attack_ms: float,
    release_ms: float,
    sample_rate: float,
) -> list[float]:
    """Apply a click-free gain envelope to samples according to a gate state.

    The gain starts at ``0.0``.  For each frame the target gain is ``1.0``
    when ``state[i]`` is true and ``0.0`` otherwise.  The gain moves linearly
    toward that target by ``1 / attack_frames`` per sample while opening, or
    by ``1 / release_frames`` per sample while closing, where
    ``attack_frames = round(attack_ms * sample_rate / 1000)`` and likewise for
    release.  A time of ``0`` ms makes that transition instantaneous.  This
    bounded per-sample change is what keeps switching from producing clicks.

    Args:
        samples: Input signal.
        state: Gate state.  It may be shorter than ``samples`` (for example
            one entry per analysis frame); samples past the last supplied state
            entry are treated as closed.  It must not be longer than
            ``samples``.
        attack_ms: Opening ramp length in milliseconds; non-negative.
        release_ms: Closing ramp length in milliseconds; non-negative.
        sample_rate: Audio sample rate in samples per second; positive.

    Returns:
        ``samples[i] * gain[i]`` for every input sample.

    Raises:
        GateError: If ``state`` is longer than ``samples``, a time is
            negative, or ``sample_rate`` is not positive.

    Float results are compared with a tolerance of ``1e-6``.
    """
    if len(state) > len(samples):
        raise GateError("state must not be longer than samples")
    # Samples past the last analysed frame are treated as closed.
    padded_state = list(state) + [False] * (len(samples) - len(state))
    attack = _non_negative(attack_ms, "attack_ms")
    release = _non_negative(release_ms, "release_ms")
    rate = _positive(sample_rate, "sample_rate")
    for sample in samples:
        _finite(sample, "sample")

    attack_frames = int(round(attack * rate / 1000.0))
    release_frames = int(round(release * rate / 1000.0))
    attack_step = math.inf if attack_frames == 0 else 1.0 / attack_frames
    release_step = math.inf if release_frames == 0 else 1.0 / release_frames

    output: list[float] = []
    gain = 0.0
    for sample, is_open in zip(samples, padded_state):
        target = 1.0 if is_open else 0.0
        if gain < target:
            gain = min(target, gain + attack_step)
        elif gain > target:
            gain = max(target, gain - release_step)
        # Snap accumulated floating-point error onto the exact target.
        if abs(gain - target) <= _EPS:
            gain = target
        output.append(float(sample) * gain)
    return output


def sidechain_key(
    samples: list[float], key: list[float], depth: float
) -> list[float]:
    """Attenuate samples with a sidechain control envelope.

    ``key`` is a per-sample control signal in ``[0, 1]``, where ``1`` means
    fully pass and ``0`` means fully duck.  The applied gain is
    ``1 - depth * (1 - key[i])``, so ``depth == 0`` bypasses the key and
    ``depth == 1`` gives a fully keyed signal.

    ``depth`` is clamped to ``[0, 1]`` before use; values above ``1`` must not
    invert the signal phase.  Key values are also clamped to ``[0, 1]``.

    Args:
        samples: Input signal.
        key: Control envelope, same length as ``samples``.
        depth: Keying amount; finite, clamped to ``[0, 1]``.

    Returns:
        The keyed signal, same length as ``samples``.

    Raises:
        GateError: If the lengths differ, ``depth`` is not finite, or any
            input is not finite.

    Float results are compared with a tolerance of ``1e-6``.
    """
    if len(samples) != len(key):
        raise GateError("samples and key must have the same length")
    amount = min(1.0, max(0.0, _finite(depth, "depth")))
    for sample in samples:
        _finite(sample, "sample")
    for value in key:
        _finite(value, "key")

    output: list[float] = []
    for sample, control in zip(samples, key):
        control = min(1.0, max(0.0, float(control)))
        output.append(float(sample) * (1.0 - amount * (1.0 - control)))
    return output


def estimate_noise_floor(levels: list[float], percentile: float) -> float:
    """Estimate the noise floor as a percentile of frame levels.

    ``levels`` is sorted ascending and the requested percentile is evaluated
    with linear interpolation at rank ``(len(levels) - 1) * percentile / 100``.
    Thus ``percentile == 0`` returns the minimum and ``percentile == 100``
    returns the maximum.  A low percentile (for example ``10``) tracks the
    quiet bed and ignores short bursts, unlike a minimum estimator which is
    dragged down by the quietest single frame.

    Args:
        levels: Frame levels in dBFS; must not be empty.
        percentile: Desired percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile in dBFS.

    Raises:
        GateError: If ``levels`` is empty, if ``percentile`` is outside
            ``[0, 100]``, or if an input is not finite.

    Float results are compared with a tolerance of ``1e-6``.
    """
    if not levels:
        raise GateError("levels must not be empty")
    pct = _finite(percentile, "percentile")
    if not 0.0 <= pct <= 100.0:
        raise GateError("percentile must be within [0, 100]")
    for level in levels:
        _finite(level, "level")

    ordered = sorted(float(level) for level in levels)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def hysteresis_check(open_db: float, close_db: float) -> None:
    """Validate that the open threshold is strictly above the close threshold.

    The gate needs ``open_db > close_db`` so that the two thresholds form a
    hysteresis band.  Equal or inverted thresholds would make the state
    machine lock up or chatter.

    Args:
        open_db: Open threshold in dBFS.
        close_db: Close threshold in dBFS.

    Returns:
        ``None`` when the thresholds are valid.

    Raises:
        GateError: If ``open_db <= close_db`` or either value is not finite.
    """
    open_threshold = _finite(open_db, "open_db")
    close_threshold = _finite(close_db, "close_db")
    if open_threshold <= close_threshold:
        raise GateError("open_db must be strictly greater than close_db")


def hold_counter(state: list[bool], hold_frames: int) -> list[int]:
    """Count down the remaining hold frames for each frame of a gate state.

    A true state frame resets the counter to ``hold_frames``.  A false frame
    decrements the previous counter by one, but the result is clamped at zero
    so the count can never become negative and close a gate early.

    Args:
        state: Gate state, one boolean per frame.
        hold_frames: Number of hold frames; must be non-negative.

    Returns:
        One non-negative integer per input frame, in input order.

    Raises:
        GateError: If ``hold_frames`` is negative.
    """
    if isinstance(hold_frames, bool) or not isinstance(hold_frames, int):
        if isinstance(hold_frames, float) and hold_frames.is_integer():
            hold_frames = int(hold_frames)
        else:
            raise GateError("hold_frames must be an integer")
    if hold_frames < 0:
        raise GateError("hold_frames must be non-negative")

    counts: list[int] = []
    remaining = 0
    for is_open in state:
        if is_open:
            remaining = hold_frames
        else:
            remaining = max(0, remaining - 1)
        counts.append(remaining)
    return counts


def gate_batch(tracks: list[list[float]], params: dict) -> dict:
    """Process several tracks independently and atomically.

    Every track must get a fresh gate state machine; no state may leak from
    one track to the next.  For each track the processing chain is
    :func:`level_db` -> :func:`gate_state` -> :func:`apply_gate`.

    ``gate_state`` is called at the level-frame rate ``sample_rate / hop``
    (not at the audio rate), because ``levels`` contains one value per
    ``hop`` samples.  The resulting frame-level boolean state is expanded for
    :func:`apply_gate` by repeating frame ``k`` over the sample range
    ``[k * hop, (k + 1) * hop)``; samples after the last analysed frame are
    treated as closed.  The audio-rate ``sample_rate`` is passed unchanged to
    :func:`apply_gate` for attack and release smoothing.

    ``params`` must contain ``open_db``, ``close_db``, ``attack_ms``,
    ``release_ms``, ``hold_ms``, ``sample_rate``, ``window``, and ``hop``.
    Missing or invalid parameters are reported as :class:`GateError`, not as
    ``KeyError`` or another exception type.  All tracks and parameters are
    validated before any result is produced; if any track is invalid the whole
    batch fails with :class:`GateError` and no partial result is returned.

    Args:
        tracks: Non-empty list of sample lists.
        params: Shared gate parameters for every track.

    Returns:
        A dict with three parallel lists in input order:
        ``{"levels": [...], "states": [...], "outputs": [...]}``.  Element
        ``i`` of each list belongs to ``tracks[i]``: ``levels`` and ``states``
        have one entry per complete analysis frame, while ``outputs`` has one
        entry per input sample.

    Raises:
        GateError: If ``tracks`` is empty or contains invalid data, if
            ``params`` is missing a required key, or if any parameter is
            invalid.

    Float results are compared with a tolerance of ``1e-6``.
    """
    if not isinstance(tracks, (list, tuple)) or len(tracks) == 0:
        raise GateError("tracks must be a non-empty list of sample lists")
    if not isinstance(params, dict):
        raise GateError("params must be a mapping")

    required = (
        "open_db",
        "close_db",
        "attack_ms",
        "release_ms",
        "hold_ms",
        "sample_rate",
        "window",
        "hop",
    )
    for name in required:
        if name not in params:
            raise GateError(f"missing parameter: {name}")

    # Validate the shared parameters once, up front.
    open_db = _finite(params["open_db"], "open_db")
    close_db = _finite(params["close_db"], "close_db")
    hysteresis_check(open_db, close_db)
    attack_ms = _non_negative(params["attack_ms"], "attack_ms")
    release_ms = _non_negative(params["release_ms"], "release_ms")
    hold_ms = _non_negative(params["hold_ms"], "hold_ms")
    sample_rate = _positive(params["sample_rate"], "sample_rate")
    window = _positive_int(params["window"], "window")
    hop = _positive_int(params["hop"], "hop")

    # Validate every track before doing any work, so a bad track aborts the
    # whole batch with no partial result.
    for index, track in enumerate(tracks):
        if isinstance(track, (str, bytes)) or not isinstance(track, (list, tuple)):
            raise GateError(f"track {index} must be a sequence of samples")
        for sample in track:
            _finite(sample, f"track {index} sample")

    frame_rate = sample_rate / hop

    all_levels: list[list[float]] = []
    all_states: list[list[bool]] = []
    all_outputs: list[list[float]] = []
    for track in tracks:
        samples = [float(sample) for sample in track]
        levels = level_db(samples, window, hop)
        states = gate_state(
            levels,
            open_db,
            close_db,
            attack_ms,
            release_ms,
            hold_ms,
            frame_rate,
        )

        expanded: list[bool] = []
        for index, frame_open in enumerate(states):
            expanded.extend([frame_open] * hop)
        if len(expanded) < len(samples):
            expanded.extend([False] * (len(samples) - len(expanded)))

        outputs = apply_gate(samples, expanded, attack_ms, release_ms, sample_rate)

        all_levels.append(levels)
        all_states.append(states)
        all_outputs.append(outputs)

    return {"levels": all_levels, "states": all_states, "outputs": all_outputs}

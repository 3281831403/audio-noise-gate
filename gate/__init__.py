"""Public surface for the audio-noise-gate exercise."""

from .noisegate import (
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

__all__ = [
    "GateError",
    "apply_gate",
    "estimate_noise_floor",
    "gate_batch",
    "gate_state",
    "hold_counter",
    "hysteresis_check",
    "level_db",
    "sidechain_key",
]

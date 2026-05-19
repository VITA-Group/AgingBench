"""
inference/_verdict.py — Saturation-aware verdict for trajectory metrics.

The naive "slope > 0 = rising = degrading" rule is misleading when a
trajectory has already saturated at a boundary. Example: a goal_anchor
drift that collapses to 0.0 by session 3, then stays at 0.0 forever,
will have a tiny positive OLS slope (numerical noise) — but it's not
"healthy and improving"; it has bottomed out.

This helper looks at the late-window mean of the trajectory and reports
FLOOR / CEILING / RISING_* / FALLING_* / FLAT / NO_SIGNAL accordingly.

Used by every per-mechanism inference module to attach a
`<metric_name>_verdict` field alongside each trajectory.
"""
from __future__ import annotations

from typing import Optional, Sequence


# Verdict labels (returned as plain strings for easy JSON serialisation)
NO_SIGNAL          = "no_signal"           # trajectory too short / mostly None
FLAT               = "flat"                # |slope| below epsilon
RISING_DEGRADATION = "rising_degradation"  # rising in a metric where rising is bad
RISING_HEALTHY     = "rising_healthy"      # rising in a metric where rising is good
FALLING_DEGRADATION = "falling_degradation"
FALLING_HEALTHY     = "falling_healthy"
FLOOR_DEGRADATION  = "floor_degradation"   # collapsed to floor in a metric where lower=worse
FLOOR_HEALTHY      = "floor_healthy"       # bottomed out in a metric where lower=better
CEILING_DEGRADATION = "ceiling_degradation"
CEILING_HEALTHY     = "ceiling_healthy"


def degradation_verdict(
    trajectory: Sequence[Optional[float]],
    slope: Optional[float],
    *,
    rising_is_bad: bool = True,
    floor_threshold: Optional[float] = None,
    ceiling_threshold: Optional[float] = None,
    late_window_frac: float = 0.25,
    slope_eps: float = 1e-3,
) -> str:
    """Saturation-aware verdict for a trajectory.

    Order of checks:
      1. NO_SIGNAL: fewer than 3 non-None points
      2. FLOOR / CEILING: late-window mean below/above the threshold
      3. FLAT: |slope| < slope_eps
      4. RISING / FALLING (with healthy-vs-degrading depending on rising_is_bad)

    Parameters
    ----------
    rising_is_bad : bool
        True if larger values mean degradation (e.g., context_noise_ratio).
        False if smaller values mean degradation (e.g., goal_anchor_drift).
    floor_threshold : float, optional
        If late-window mean ≤ this, report FLOOR. Use for bounded-below metrics.
    ceiling_threshold : float, optional
        If late-window mean ≥ this, report CEILING. Use for bounded-above metrics.
    late_window_frac : float
        Fraction of the trajectory used to compute the late-window mean.
        Minimum 3 sessions.
    slope_eps : float
        |slope| below this is considered FLAT.
    """
    nn = [(i, v) for i, v in enumerate(trajectory) if v is not None]
    if len(nn) < 3:
        return NO_SIGNAL

    # Late-window mean (last 25% of the trajectory or last 3, whichever larger)
    n = len(nn)
    win_size = max(3, int(round(late_window_frac * n)))
    late = nn[-win_size:]
    late_mean = sum(v for _, v in late) / len(late)

    # Floor check: bottomed out below the threshold
    if floor_threshold is not None and late_mean <= floor_threshold:
        # If rising_is_bad → low values are good → floor = healthy
        # If rising_is_bad=False → low values are bad → floor = collapsed
        return FLOOR_HEALTHY if rising_is_bad else FLOOR_DEGRADATION

    # Ceiling check: saturated above the threshold
    if ceiling_threshold is not None and late_mean >= ceiling_threshold:
        return CEILING_DEGRADATION if rising_is_bad else CEILING_HEALTHY

    # Slope dispatch
    if slope is None or abs(slope) < slope_eps:
        return FLAT

    if slope > 0:
        return RISING_DEGRADATION if rising_is_bad else RISING_HEALTHY
    else:
        return FALLING_DEGRADATION if not rising_is_bad else FALLING_HEALTHY


def is_degrading(verdict: str) -> bool:
    """True if the verdict indicates aging/degradation."""
    return verdict in (RISING_DEGRADATION, FALLING_DEGRADATION,
                       FLOOR_DEGRADATION, CEILING_DEGRADATION)

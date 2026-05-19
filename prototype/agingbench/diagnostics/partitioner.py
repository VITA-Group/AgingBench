"""
agingbench/diagnostics/partitioner.py — Diagnostic error partitioning (§5.2).

Computes the three mutually exclusive error components from P1/P2/P3 scores:

  Total Error = 1 − Acc_P1
             = Utilization_Error + Write_Error + Read_Error

  Utilization Error = 1 − Acc_P3         → Revision Aging   (𝒰)
  Write Error       = Acc_P3 − Acc_P2    → Compression Aging (𝒲)
  Read Error        = Acc_P2 − Acc_P1    → Interference Aging (ℛ)

Maintenance aging is disambiguated temporally: a discrete step-change in
Write Error across a lifecycle event at session k indicates exogenous store
mutation rather than endogenous compression.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DiagnosticResult:
    """Per-session diagnostic error partition."""

    session: int
    acc_p1: float  # Baseline accuracy
    acc_p2: float  # Oracle retrieval accuracy (from agent's actual store)
    acc_p3: float  # Oracle context accuracy (ground truth injected)
    n_probes: int = 0

    @property
    def utilization_error(self) -> float:
        """1 − Acc_P3: LLM fails despite perfect context → Revision Aging (𝒰)."""
        return max(0.0, 1.0 - self.acc_p3)

    @property
    def write_error(self) -> float:
        """Acc_P3 − Acc_P2: facts lost by write policy → Compression Aging (𝒲)."""
        return max(0.0, self.acc_p3 - self.acc_p2)

    @property
    def read_error(self) -> float:
        """Acc_P2 − Acc_P1: facts in store but not retrieved → Interference Aging (ℛ)."""
        return max(0.0, self.acc_p2 - self.acc_p1)

    def to_dict(self) -> dict:
        return {
            "session": self.session,
            "acc_p1": round(self.acc_p1, 4),
            "acc_p2": round(self.acc_p2, 4),
            "acc_p3": round(self.acc_p3, 4),
            "n_probes": self.n_probes,
            "utilization_error": round(self.utilization_error, 4),
            "write_error": round(self.write_error, 4),
            "read_error": round(self.read_error, 4),
        }


def partition_errors(
    p1_scores: dict[int, float],
    p2_scores: dict[int, float],
    p3_scores: dict[int, float],
    n_probes_per_session: Optional[dict[int, int]] = None,
) -> dict:
    """Compute diagnostic error partition across all sessions.

    Parameters
    ----------
    p1_scores : dict[int, float]
        {session_idx: accuracy} under P1 (baseline).
    p2_scores : dict[int, float]
        {session_idx: accuracy} under P2 (oracle retrieval from actual store).
    p3_scores : dict[int, float]
        {session_idx: accuracy} under P3 (oracle context / ground truth).
    n_probes_per_session : dict[int, int], optional
        {session_idx: count} — number of probes per session (for weighting).

    Returns
    -------
    dict with:
        per_session: list[dict] — per-session DiagnosticResult dicts
        aggregate: {utilization_error, write_error, read_error, total_error}
        write_error_series: list[tuple[int, float]] — for maintenance detection
    """
    all_sessions = sorted(set(p1_scores) & set(p2_scores) & set(p3_scores))

    per_session: list[DiagnosticResult] = []
    for s in all_sessions:
        n_p = (n_probes_per_session or {}).get(s, 0)
        per_session.append(DiagnosticResult(
            session=s,
            acc_p1=p1_scores[s],
            acc_p2=p2_scores[s],
            acc_p3=p3_scores[s],
            n_probes=n_p,
        ))

    # Aggregate (weighted by n_probes if available, else uniform)
    if per_session:
        weights = [d.n_probes if d.n_probes > 0 else 1 for d in per_session]
        total_w = sum(weights)
        agg_util = sum(d.utilization_error * w for d, w in zip(per_session, weights)) / total_w
        agg_write = sum(d.write_error * w for d, w in zip(per_session, weights)) / total_w
        agg_read = sum(d.read_error * w for d, w in zip(per_session, weights)) / total_w
    else:
        agg_util = agg_write = agg_read = 0.0

    write_series = [(d.session, d.write_error) for d in per_session]

    return {
        "per_session": [d.to_dict() for d in per_session],
        "aggregate": {
            "utilization_error": round(agg_util, 4),
            "write_error": round(agg_write, 4),
            "read_error": round(agg_read, 4),
            "total_error": round(agg_util + agg_write + agg_read, 4),
        },
        "write_error_series": write_series,
        "maintenance_discontinuities": detect_maintenance_discontinuity(write_series),
    }


def detect_maintenance_discontinuity(
    write_errors: list[tuple[int, float]],
    threshold: float = 0.10,
) -> list[dict]:
    """Detect structural discontinuities in the write error time series.

    Maintenance aging manifests as a discrete step-change in WriteError
    (ΔW = WriteError_{k+1} − WriteError_{k−1}), distinct from the smooth
    continuous degradation of endogenous compression aging.

    Parameters
    ----------
    write_errors : list of (session, write_error) tuples, sorted by session.
    threshold : float
        Minimum |ΔW| to flag as a potential maintenance discontinuity.

    Returns
    -------
    list of {session, delta_w, is_discontinuity} dicts for each interior
    session (excluding first and last).
    """
    if len(write_errors) < 3:
        return []

    results = []
    for i in range(1, len(write_errors) - 1):
        s_prev, w_prev = write_errors[i - 1]
        s_curr, w_curr = write_errors[i]
        s_next, w_next = write_errors[i + 1]
        # ΔW = WriteError_{k+1} − WriteError_{k−1}
        delta_w = w_next - w_prev
        results.append({
            "session": s_curr,
            "delta_w": round(delta_w, 4),
            "is_discontinuity": abs(delta_w) >= threshold,
        })

    return results

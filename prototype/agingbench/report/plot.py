"""
agingbench/report/plot.py — Aging curve visualizations (PDF §7).

Produces the primary 2×2 panel figure and per-curve line plots.
Requires matplotlib. Output saved to experiments/results/<run_id>/.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from ..metrics.aging import AgingCurve, summarize, aggregate_curves


def plot_curve(
    curve: AgingCurve,
    output_path: str,
    title: str = "",
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    shock_sessions: Optional[list] = None,
) -> None:
    """Save a single aging curve as a line plot.

    Parameters
    ----------
    ylabel, xlabel : optional scenario-specific axis labels. Defaults adapt
        to the curve's metric name (stored on AgingCurve if available) or
        a generic "m(t)" / "Session".
    shock_sessions : optional list of session indices at which maintenance
        shock events occurred (flush/recompact). Drawn as red dashed vlines.
    """
    if not _HAS_MPL:
        print("[plot] matplotlib not available — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve.exposures, curve.scores, marker="o", linewidth=2, color="#2563EB")
    ax.axhline(y=curve.scores[0] * 0.5, linestyle="--", color="gray", alpha=0.6,
               label="half-life threshold (50%)")

    if shock_sessions:
        for s in shock_sessions:
            ax.axvline(x=s, linestyle=":", color="#DC2626", alpha=0.6, linewidth=1.2)
        ax.plot([], [], linestyle=":", color="#DC2626", label="maintenance event")

    stats = summarize(curve)
    hl = stats["half_life"]
    hl_str = f"{hl:.1f}" if hl != float("inf") else "∞"
    ax.set_title(title or f"{curve.sut_id} — {curve.scenario}")
    ax.set_xlabel(xlabel or "Session")
    ax.set_ylabel(ylabel or _default_ylabel(curve))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.text(0.98, 0.05,
            f"m(0)={stats['m0']:.2f}  m_final={stats['m_final']:.2f}  "
            f"t½={hl_str}  slope={stats['decay_slope']:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {output_path}")


_SCENARIO_YLABEL = {
    "s1_research_literature":  "Keyword retention m(t)",
    "s2_lifestyle_assistant":  "Constraint precision m(t)",
    "s3_knowledge_base":       "Summarization fidelity m(t)",
    "s4_software_engineering": "Dependency recall m(t)",
    "s5_self_planning":        "Workspace recall m(t)",
    "s6_naturalistic":         "Lag recall m(t)",
    "s7_research_notes":       "Workspace recall m(t)",
    "s8_swe_bench":       "Probe recall m(t)",
}


def _default_ylabel(curve: AgingCurve) -> str:
    scen = getattr(curve, "scenario", "") or ""
    # Strip any leading path/prefix like "gen:s2_..." or "curated:s1_..."
    scen_key = scen.split(":")[-1].strip()
    return _SCENARIO_YLABEL.get(scen_key, "Score m(t)")


def compare_curves(
    curves: list[AgingCurve],
    output_path: str,
    title: str = "",
    labels: Optional[list] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    shock_sessions: Optional[list] = None,
    normalize: bool = True,
) -> None:
    """Overlay multiple aging curves on one axes for cross-SUT comparison.

    Parameters
    ----------
    normalize : if True, assume all curves are on a [0,1] scale and set ylim
        to (0, 1.05). Set False when mixing curves of different units (e.g.,
        accumulator_error alongside a [0,1] precision curve) — the caller
        should then hand-draw a twin axis or omit the mixed curve here and
        call plot_dual_axis_curves() instead.
    shock_sessions : optional session indices to mark with red dashed vlines.
    """
    if not _HAS_MPL:
        print("[plot] matplotlib not available — skipping plot")
        return

    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#D97706"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (curve, color) in enumerate(zip(curves, colors)):
        label = labels[i] if labels and i < len(labels) else curve.sut_id
        ax.plot(curve.exposures, curve.scores, marker="o", linewidth=2,
                label=label, color=color)

    if shock_sessions:
        for s in shock_sessions:
            ax.axvline(x=s, linestyle=":", color="#DC2626", alpha=0.6, linewidth=1.2)
        ax.plot([], [], linestyle=":", color="#DC2626", label="maintenance event")

    # Prefer scenario-aware label based on the first curve if not overridden.
    first = curves[0] if curves else None
    ax.set_title(title or "Aging Curves — Multi-metric Overlay")
    ax.set_xlabel(xlabel or "Session")
    ax.set_ylabel(ylabel or (_default_ylabel(first) if first else "Score m(t)"))
    if normalize:
        ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {output_path}")


def plot_dual_axis_curves(
    primary_curve: AgingCurve,
    secondary_curve: AgingCurve,
    output_path: str,
    title: str = "",
    primary_label: str = "",
    secondary_label: str = "",
    primary_ylabel: Optional[str] = None,
    secondary_ylabel: Optional[str] = None,
    shock_sessions: Optional[list] = None,
) -> None:
    """Plot two curves on twin y-axes (e.g., bounded-[0,1] metric + unbounded error).

    Typical use: S2 precision(t) on left axis [0,1], accumulator_error(t) on
    right axis (unbounded positive). Both share the x-axis (session).
    """
    if not _HAS_MPL:
        print("[plot] matplotlib not available — skipping plot")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#2563EB"
    ax1.plot(primary_curve.exposures, primary_curve.scores, marker="o",
             linewidth=2, color=color1, label=primary_label or "primary")
    ax1.set_xlabel("Session")
    ax1.set_ylabel(primary_ylabel or _default_ylabel(primary_curve), color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#DC2626"
    ax2.plot(secondary_curve.exposures, secondary_curve.scores, marker="s",
             linewidth=2, linestyle="--", color=color2,
             label=secondary_label or "secondary")
    ax2.set_ylabel(secondary_ylabel or "Error", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    if len(secondary_curve.scores) > 0:
        ax2.set_ylim(0, max(secondary_curve.scores) * 1.15 + 1e-6)

    if shock_sessions:
        for s in shock_sessions:
            ax1.axvline(x=s, linestyle=":", color="#DC2626", alpha=0.6, linewidth=1.2)

    ax1.set_title(title or "Aging Curve — dual axis")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved dual-axis → {output_path}")


def plot_curves_with_ci(
    curve_groups: dict[str, list[AgingCurve]],
    output_path: str,
    title: str = "",
) -> None:
    """
    Plot multiple metric groups with CI bands from multi-seed runs.

    Args:
        curve_groups: {"metric_name": [seed0_curve, seed1_curve, ...], ...}
        output_path: path to save PNG
        title: plot title
    """
    if not _HAS_MPL:
        print("[plot] matplotlib not available — skipping plot")
        return

    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#D97706"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (name, curves) in enumerate(curve_groups.items()):
        color = colors[i % len(colors)]
        agg = aggregate_curves(curves)
        if not agg:
            continue

        exposures = agg["exposures"]
        mean = agg["mean"]
        ci_lo = agg["ci_lower"]
        ci_hi = agg["ci_upper"]
        n = agg["n_seeds"]

        ax.plot(exposures, mean, marker="o", linewidth=2, color=color,
                label=f"{name} (n={n})", markersize=4)
        ax.fill_between(exposures, ci_lo, ci_hi, alpha=0.2, color=color)

        # Individual seed traces as thin lines
        for seed_scores in agg["per_seed"]:
            ax.plot(exposures[:len(seed_scores)], seed_scores,
                    linewidth=0.5, alpha=0.3, color=color)

    ax.set_title(title or "Aging Curves with 95% CI")
    ax.set_xlabel("Session")
    ax.set_ylabel("Score m(t)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved CI → {output_path}")

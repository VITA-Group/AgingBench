"""S8 SWE-bench-Aging — lifecycle event scheduler + handlers.

Translates PressureConfig dials into per-session lifecycle events,
then applies them to the container/host workspace at runtime.

Event types:
  - workspace_flush: delete /agentmemory/.aging/  (memory shock; matches
    `forget_rate` dial)
  - dep_bump:        upgrade a randomly-chosen dep in the container
                     (real maintenance; matches `update_rate` dial)
  - branch_switch:   simulated upstream rebase — not a real git op since
                     each container is single-commit, but recorded for
                     audit / mechanism scoring
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from agingbench.generators.pressure_config import PressureConfig


@dataclass
class LifecycleEvent:
    session: int
    event_type: str          # workspace_flush | dep_bump | branch_switch
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "session": self.session,
            "event_type": self.event_type,
            "detail": self.detail,
        }


# Legacy sphinx-era default. Per-chain lists live in the issue-chain YAML
# (`dep_bump_candidates`) and are threaded in through LifecycleScheduler.
# This default is kept for back-compat with manifests that don't specify
# their own list. The revision probe is much more discriminative when
# the bumped pkg is part of the chain's actual import surface.
DEP_BUMP_CANDIDATES = ["pytest", "docutils", "Pygments", "Jinja2", "babel"]


@dataclass
class LifecycleScheduler:
    """Pre-computes lifecycle events per session, deterministic per seed."""
    pressure: PressureConfig
    n_sessions: int
    seed: int
    # Per-chain dep-bump pool. None -> falls back to module-level
    # DEP_BUMP_CANDIDATES so existing callers/tests stay green.
    dep_bump_candidates: Optional[list[str]] = None
    # Pinned shocks: deterministic per-session fires regardless of seed
    # or forget_rate. Chain manifests can declare these to GUARANTEE
    # multiple maintenance data points in a run (otherwise the
    # stochastic schedule may produce 0 or 1 shocks, leaving the
    # maintenance probe sparse).
    pinned_workspace_flushes: Optional[list[int]] = None
    pinned_dep_bumps: Optional[list[dict]] = None     # [{"session": int, "pkg": str}, ...]
    # Phase 14b: roll packages BACK to a known-old version at chain start,
    # so later dep_bump events produce a real version change (needed for
    # the belief-revision drift probe to work). Each pin fires at the
    # `session` field (defaults to 0 / warmup) and pip-installs pkg==ver.
    chain_baseline_pins: Optional[list[dict]] = None  # [{"session": int, "pkg": str, "version": str}, ...]
    _rng: random.Random = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Distinct from the agent / generator RNG to keep streams independent.
        self._rng = random.Random(self.seed * 6151 + 19)

    def schedule(self) -> list[LifecycleEvent]:
        """Return all lifecycle events that should fire across the run.

        Multiple events can land at the same session (e.g., a flush AND
        a dep bump). The runner applies them in scheduling order.
        Pinned events are emitted first, then stochastic events are
        added per-session (without duplicating a pinned event at the
        same session+type).
        """
        warmup = max(0, int(self.pressure.warmup_sessions))
        forget_p = max(0.0, min(1.0, float(self.pressure.forget_rate)))
        update_p = max(0.0, min(1.0, float(self.pressure.update_rate)))
        pool = list(self.dep_bump_candidates or DEP_BUMP_CANDIDATES)

        events: list[LifecycleEvent] = []
        # Chain baseline pins. Containers are PER-SESSION ephemeral, so a
        # single pin at session 0 doesn't persist — each subsequent
        # session starts from the image's `latest` pkg state. Re-emit
        # the pin at EVERY session from `pin.session` up to (but not
        # including) the FIRST `dep_bump` on that pkg. After the bump
        # the agent should observe the upgraded version.
        bump_sessions_by_pkg: dict[str, int] = {}
        for entry in (self.pinned_dep_bumps or []):
            pkg = entry["pkg"]
            t_i = int(entry["session"])
            if pkg not in bump_sessions_by_pkg or t_i < bump_sessions_by_pkg[pkg]:
                bump_sessions_by_pkg[pkg] = t_i
        for pin in (self.chain_baseline_pins or []):
            start_t = int(pin.get("session", 0))
            pkg = pin["pkg"]
            ver = pin["version"]
            bump_t = bump_sessions_by_pkg.get(pkg, self.n_sessions)
            for t_i in range(max(0, start_t), min(self.n_sessions, bump_t)):
                events.append(LifecycleEvent(
                    session=t_i, event_type="dep_pin",
                    detail=f"pip install {pkg}=={ver}  (chain baseline pin)",
                ))
        pinned_flush_set: set[int] = set()
        pinned_bump_set: set[int] = set()
        for t in (self.pinned_workspace_flushes or []):
            t_i = int(t)
            if 0 <= t_i < self.n_sessions:
                events.append(LifecycleEvent(
                    session=t_i, event_type="workspace_flush",
                    detail="rm -rf /agentmemory/.aging/  (pinned)",
                ))
                pinned_flush_set.add(t_i)
        for entry in (self.pinned_dep_bumps or []):
            t_i = int(entry["session"])
            pkg = entry["pkg"]
            if 0 <= t_i < self.n_sessions:
                events.append(LifecycleEvent(
                    session=t_i, event_type="dep_bump",
                    detail=f"pip install --upgrade {pkg}  (pinned)",
                ))
                pinned_bump_set.add(t_i)

        for t in range(self.n_sessions):
            if t < warmup:
                continue
            # Independent Bernoulli trials per session, skip if pinned.
            if t not in pinned_flush_set and self._rng.random() < forget_p:
                events.append(LifecycleEvent(
                    session=t, event_type="workspace_flush",
                    detail="rm -rf /agentmemory/.aging/  (forget_rate dial)",
                ))
            if t not in pinned_bump_set and self._rng.random() < update_p:
                pkg = self._rng.choice(pool)
                events.append(LifecycleEvent(
                    session=t, event_type="dep_bump",
                    detail=f"pip install --upgrade {pkg}  (update_rate dial)",
                ))
        return events


def apply_event(session, event: LifecycleEvent) -> dict:
    """Execute a lifecycle event against an open S8DockerSession.

    Returns a dict suitable for embedding in per-session results
    describing what was applied + the outcome.
    """
    # Imported here to keep this module dep-light for tests.
    from agingbench.scenarios.s8_swe_bench.docker_runner import S8DockerSession
    assert isinstance(session, S8DockerSession)

    if event.event_type == "workspace_flush":
        # Host-side delete is fastest + simpler than docker exec rm.
        # Flush ALL of the agent's persistent memory dir, regardless of which
        # internal layout it chose (`.aging/`, a sibling `notes/`, a `.db`,
        # etc.); a narrower flush leaves agents that picked a non-default
        # layout immune to the maintenance shock.
        memory_root = session.memory_dir
        bytes_freed = 0
        import shutil
        if memory_root.exists():
            for child in memory_root.iterdir():
                if child.is_file():
                    bytes_freed += child.stat().st_size
                    child.unlink()
                elif child.is_dir():
                    for f in child.rglob("*"):
                        if f.is_file():
                            bytes_freed += f.stat().st_size
                    shutil.rmtree(child)
        return {
            "event_type": "workspace_flush",
            "session": event.session,
            "outcome": "ok",
            "bytes_freed": bytes_freed,
        }

    if event.event_type == "dep_bump":
        # Parse the package name out of the detail.
        # Detail looks like: "pip install --upgrade <pkg>  (...)"
        parts = event.detail.split()
        pkg = parts[parts.index("--upgrade") + 1] if "--upgrade" in parts else "pytest"
        # Use the container's conda python (pre-installed at /opt/miniconda3).
        result = session.exec(
            f"PY=/opt/miniconda3/envs/testbed/bin/python; [ -x $PY ] || PY=/opt/miniconda3/bin/python; $PY -m pip install --upgrade --quiet {pkg}",
            timeout_sec=120,
        )
        return {
            "event_type": "dep_bump",
            "session": event.session,
            "package": pkg,
            "exit_code": result.exit_code,
            "outcome": "ok" if result.exit_code == 0 else "failed",
            "duration_sec": result.duration_sec,
        }

    if event.event_type == "dep_pin":
        # Pin a package to a SPECIFIC version. Used at chain start to
        # roll a package back to a known-old version so that a later
        # `dep_bump` produces a measurable version change (otherwise the
        # bump is a no-op when the pkg is already at latest). The detail
        # encodes pkg==version.
        parts = event.detail.split()
        spec = next((p for p in parts if "==" in p), None)
        if not spec:
            return {"event_type": "dep_pin", "session": event.session,
                    "outcome": "missing_spec"}
        # Target the TESTBED conda env (where the issue's deps live).
        # The agent's `pip show <pkg>` reads from this env, so the pin
        # only affects the agent's observation when installed here.
        result = session.exec(
            "PY=/opt/miniconda3/envs/testbed/bin/python; "
            "[ -x $PY ] || PY=/opt/miniconda3/bin/python; "
            f"$PY -m pip install --quiet --force-reinstall {spec}",
            timeout_sec=180,
        )
        return {
            "event_type": "dep_pin", "session": event.session,
            "spec": spec,
            "exit_code": result.exit_code,
            "outcome": "ok" if result.exit_code == 0 else "failed",
            "duration_sec": result.duration_sec,
        }

    if event.event_type == "branch_switch":
        # Not a real git op — single-commit container. Recorded for audit.
        return {
            "event_type": "branch_switch",
            "session": event.session,
            "outcome": "noop_audit_only",
            "detail": event.detail,
        }

    return {
        "event_type": event.event_type,
        "session": event.session,
        "outcome": "unknown_event_type",
    }

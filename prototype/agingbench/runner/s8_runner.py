"""S8 SWE-bench-Aging top-level runner (Phase 2).

Orchestrates the longitudinal stream: per-session container lifecycle,
lifecycle events, persistent /agentmemory volume, hand-off to the agent
layer (Phase 3), and per-session metric collection.

Phase 2 contract: real Docker integration + lifecycle events + memory
persistence verified end-to-end. The agent layer (Phase 3) and the
verification layer (Phase 4) are stubs at this point — runner returns
a metrics dict that records what HAPPENED per session (containers
spun, events applied, memory size) but doesn't yet score pass/fail
or fire mechanism probes.

Phase 3 will replace the stub agent step with real Claude Code / OpenHands
invocation via docker exec.

Phase 4 will wire run-tests.sh + grading.py for real pass/fail signal,
and add the 4-mechanism probes.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from agingbench.generators.pressure_config import PressureConfig
from agingbench.scenarios.s8_swe_bench.docker_runner import (
    S8DockerSession,
    docker_available,
    image_exists_locally,
    resolve_image_for_instance,
)
from agingbench.scenarios.s8_swe_bench.lifecycle import (
    LifecycleEvent,
    LifecycleScheduler,
    apply_event,
)


SCENARIO_DIR = Path(__file__).parent.parent / "scenarios" / "s8_swe_bench"


@dataclass
class S8RunnerConfig:
    seed: int
    n_sessions: int
    pressure: PressureConfig
    chain_id: str = "django_orm_query"
    sut_id: str = "unknown"
    docker_image_pattern: str = "sweb.eval.x86_64.{instance_id}:latest"
    workspace_root: Optional[Path] = None       # where /agentmemory mount lives on host
    chain_path: Optional[Path] = None
    seed_manifest_path: Optional[Path] = None
    sut_cfg: Optional[dict] = None              # carries agent.adapter for Phase 3


class S8SweBenchRunner:
    """Top-level runner for S8 SWE-bench-Aging.

    Phase 2: per-session container lifecycle + lifecycle events + memory
    persistence. Agent + verification stubs land in Phases 3-4.
    """

    SCENARIO_ID = "s8_swe_bench"

    def __init__(self, cfg: S8RunnerConfig):
        self.cfg = cfg
        # Resolve workspace root for the run if not provided.
        if cfg.workspace_root is None:
            run_id = f"s8_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            cfg.workspace_root = Path("/tmp") / f"agingbench_s8_{run_id}"
        cfg.workspace_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root = cfg.workspace_root

        # Resolve chain + seed manifest paths if not provided.
        if cfg.chain_path is None:
            cfg.chain_path = SCENARIO_DIR / "issue_chains" / f"{cfg.chain_id}.yaml"
        chain_dir = Path(cfg.chain_path).parent.parent
        if cfg.seed_manifest_path is None:
            scoped = chain_dir / "seed_manifests" / f"{cfg.chain_id}_seed_{cfg.seed}.yaml"
            legacy = chain_dir / "seed_manifests" / f"seed_{cfg.seed}.yaml"
            cfg.seed_manifest_path = scoped if scoped.is_file() else legacy

        self.chain = self._load_yaml(cfg.chain_path)
        self.seed_manifest = self._load_yaml(cfg.seed_manifest_path)
        self.lifecycle_events: list[LifecycleEvent] = LifecycleScheduler(
            pressure=cfg.pressure,
            n_sessions=cfg.n_sessions,
            seed=cfg.seed,
            dep_bump_candidates=self.chain.get("dep_bump_candidates"),
            pinned_workspace_flushes=self.chain.get("pinned_workspace_flushes"),
            pinned_dep_bumps=self.chain.get("pinned_dep_bumps"),
            chain_baseline_pins=self.chain.get("chain_baseline_pins"),
        ).schedule()

        # Phase 3: build the real agent if the SUT yaml has an agent block.
        self._agent_runner = None
        if cfg.sut_cfg and (cfg.sut_cfg.get("agent") or {}).get("adapter"):
            from agingbench.scenarios.s8_swe_bench.agent import (
                build_s8_agent_from_sut,
            )
            self._agent_runner = build_s8_agent_from_sut(
                cfg.sut_cfg, host_workspace_root=self.workspace_root,
            )

        # Phase 11: precompute the shared-file map for declared interference
        # pairs (derived from gold-patch overlap). Used by the active
        # interference probe to phrase the "recall your prior edit to
        # <file>" question at session-build time. Computed once per run.
        self._interference_shared_files = self._compute_interference_shared_files()

    # ---- public API -----------------------------------------------------

    def precondition_check(self) -> dict[str, Any]:
        """Verify environment + image cache before running.

        Phase-2 stub callers should call this and abort on failure rather
        than spending compute on an obviously-broken setup.
        """
        report: dict[str, Any] = {
            "docker_available": docker_available(),
            "n_sessions_planned": self.cfg.n_sessions,
            "missing_images": [],
            "all_images_present": False,
        }
        sessions = self.seed_manifest.get("sessions", [])[: self.cfg.n_sessions]
        for s in sessions:
            iid = s["instance_id"]
            img = resolve_image_for_instance(iid, self.cfg.docker_image_pattern)
            if not image_exists_locally(img):
                report["missing_images"].append({"instance_id": iid, "image": img})
        report["all_images_present"] = len(report["missing_images"]) == 0
        return report

    def run(self) -> dict[str, Any]:
        """Execute the longitudinal run.

        Returns a results dict matching what _run_s8 will write to
        metrics.json (Phase 2 has no aging curve yet — Phase 4 wires it).
        """
        sessions_planned = self.seed_manifest.get("sessions", [])[: self.cfg.n_sessions]
        per_session_results: list[dict] = []

        # Index lifecycle events by session for quick lookup.
        events_by_session: dict[int, list[LifecycleEvent]] = {}
        for e in self.lifecycle_events:
            events_by_session.setdefault(e.session, []).append(e)
        # Stash on the instance so _real_agent_step can read prior events
        # for attestation-question construction.
        self._events_by_session = events_by_session

        # /agentmemory survives across sessions; mount the SAME host dir each time.
        memory_dir = self.workspace_root / "agentmemory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        for session_meta in sessions_planned:
            t = session_meta["session"]
            iid = session_meta["instance_id"]
            image = resolve_image_for_instance(iid, self.cfg.docker_image_pattern)

            # Apply pre-session lifecycle events (e.g., flush BEFORE the
            # session starts so the agent can't recover from its own notes).
            applied_events: list[dict] = []
            for ev in events_by_session.get(t, []):
                if ev.event_type == "workspace_flush":
                    applied_events.append(self._apply_pre_event(ev))

            # Find the chain entry for richer per-session metadata.
            chain_entry = next(
                (c for c in self.chain.get("issues", []) if c["instance_id"] == iid),
                None,
            )

            session_record: dict[str, Any] = {
                "session": t,
                "instance_id": iid,
                "image": image,
                "chain_role": (chain_entry or {}).get("chain_role"),
                "applied_events_pre": applied_events,
                "applied_events_post": [],
                "memory_size_bytes_pre": _du(memory_dir),
                "memory_size_bytes_post": None,
                "container_started": False,
                "container_id": None,
                "agent_action": None,           # Phase 3 fills this
                "verification": None,           # Phase 4 fills this
                "duration_sec": None,
                "phase": "phase_2_scaffold",
            }

            t0 = time.time()
            try:
                with S8DockerSession(image=image, memory_dir=memory_dir,
                                     instance_id=iid) as session:
                    session_record["container_started"] = True
                    session_record["container_id"] = session.container_id

                    # Phase 14b: dep_pin events must fire INSIDE the
                    # container BEFORE the agent runs (containers are
                    # ephemeral, so a pin in a prior session's container
                    # didn't persist). Apply now, with the session open.
                    for ev in events_by_session.get(t, []):
                        if ev.event_type == "dep_pin":
                            applied = apply_event(session, ev)
                            session_record["applied_events_pre"].append(applied)

                    # Phase 3 agent step: real Claude Code / OpenHands /
                    # litellm bridge if SUT yaml has agent.adapter, else
                    # the Phase-2 stub for CI.
                    if self._agent_runner is not None:
                        agent_resp = self._real_agent_step(
                            session, session_meta, chain_entry, memory_dir,
                        )
                        session_record["agent_action"] = agent_resp
                    else:
                        self._stub_agent_step(session, session_meta, chain_entry)
                        session_record["agent_action"] = "stub_phase2"

                    # Apply post-agent lifecycle events (dep_bump etc.).
                    # workspace_flush is pre-session; dep_pin handled above.
                    for ev in events_by_session.get(t, []):
                        if ev.event_type in ("workspace_flush", "dep_pin"):
                            continue
                        applied = apply_event(session, ev)
                        session_record["applied_events_post"].append(applied)
            except Exception as exc:                            # noqa: BLE001
                session_record["container_started"] = False
                session_record["error"] = f"{type(exc).__name__}: {exc}"

            session_record["duration_sec"] = round(time.time() - t0, 3)
            session_record["memory_size_bytes_post"] = _du(memory_dir)
            per_session_results.append(session_record)

        return {
            "scenario": self.SCENARIO_ID,
            "sut_id": self.cfg.sut_id,
            "seed": self.cfg.seed,
            "n_sessions": len(per_session_results),
            "chain_id": self.cfg.chain_id,
            "chain_path": str(self.cfg.chain_path),
            "seed_manifest_path": str(self.cfg.seed_manifest_path),
            "workspace_root": str(self.workspace_root),
            "lifecycle_events_planned": [e.to_dict() for e in self.lifecycle_events],
            "session_results": per_session_results,
            "pressure_used": self.cfg.pressure.to_dict(),
            "phase": "phase_2_scaffold",
        }

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"S8 yaml not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _compute_interference_shared_files(self) -> dict[str, dict[str, list[str]]]:
        """For each declared interference pair, return the files the two
        issues' gold patches both touch. Used by the active interference
        probe to phrase a concrete recall question (`recall your edit to
        <file>`). Returns {b_iid: {"partner": a_iid, "shared_files": [...]}}
        — keyed on the LATER member (the one we probe).
        """
        from agingbench.scenarios.s8_swe_bench.probes import _files_in_diff
        from agingbench.scenarios.s8_swe_bench.verifier import get_instance_metadata
        pairs = self.chain.get("interference_pairs") or []
        if not pairs:
            return {}
        # Figure out the ORDER from the seed manifest so we can pick the
        # "later" member as the probe target.
        order = {s["instance_id"]: int(s["session"])
                 for s in self.seed_manifest.get("sessions", [])}
        out: dict[str, dict[str, list[str]]] = {}
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = pair[0], pair[1]
            if a not in order or b not in order:
                continue
            # Probe the one scheduled LATER; recall reaches BACK to earlier.
            later, earlier = (b, a) if order[b] > order[a] else (a, b)
            meta_e = get_instance_metadata(earlier)
            meta_l = get_instance_metadata(later)
            files_e = _files_in_diff(meta_e.get("patch", ""))
            files_l = _files_in_diff(meta_l.get("patch", ""))
            shared = sorted(files_e & files_l)
            if shared:
                out[later] = {"partner": earlier, "shared_files": shared}
        return out

    def _build_attestation_questions(
        self, t: int, iid: str,
        events_by_session: dict[int, list[LifecycleEvent]],
    ) -> dict[str, str]:
        """Build the active-probe question set for session t.

        Phase 12 design — multi-probe-per-session for smooth trajectories:

        At every session t >= 1 we ALWAYS schedule:
          - "recall" — recall a prior session's edit (a rotating prior P;
            see _pick_recall_prior). Scores into the per-session
            interference trajectory.
          - "env"    — report version of a rotating chain-canonical pkg.
            Scores into the per-session revision trajectory; when the
            pkg got bumped at any prior session, the version differs
            from the chain baseline and the report becomes more
            informative.

        On top of those, we still schedule the EVENT-triggered probes
        from Phase 11 when applicable (so the event-conditional
        signal stays available as a sparse sidecar):
          - "revision_event" — post-bump version-check (S7-style)
          - "interference_pair" — recall the declared partner's edit
        """
        questions: dict[str, str] = {}
        if t < 1:
            return questions

        # ---- Phase 16: ORTHOGONAL FOUR-MECHANISM PROBES ----
        # Each probe is a contrast that isolates ONE mechanism:
        #   - compression_clean_p<N>: recall of a CLEAN baseline prior
        #     (not a partner, not adjacent to shocks). Pure decay signal.
        #   - interference_partner_p<N>: recall of the LATER member's
        #     declared partner. Score is contrasted vs compression_clean
        #     at same gap (the difference IS interference).
        #   - revision_<fact_id>: belief about a state_change fact. Score
        #     1 if reports `after`, 0 if reports `before`.
        #   - maintenance_<prior>_pre/post: same prior queried at both
        #     t-1 (pre-shock) and t+1 (post-shock). Delta = shock impact.
        ortho_qs = self._orthogonal_probe_questions(
            t=t, iid=iid,
            events_by_session=events_by_session,
        )
        questions.update(ortho_qs)

        # ---- Phase 13: MULTI-Q per session for smooth trajectories ----
        # interference: up to 2 distinct prior sessions per t. Aggregating
        # 2 sub-recalls (each continuous in [0,1]) makes the per-session
        # score genuinely smooth (not bimodal {0,1}).
        for prior_t in self._pick_recall_priors(t, k=2):
            prior_iid = self._iid_at_session(prior_t)
            if prior_iid:
                questions[f"recall_p{prior_t}"] = (
                    f"In session {prior_t} you worked on issue "
                    f"{prior_iid}. Briefly recall: which file(s) and "
                    f"function(s) did you change there? One short "
                    f"sentence. Consult your notes if needed."
                )
        # revision: up to 2 chain-canonical pkgs per t. Mean over 2
        # partial-credit answers gives gradient values in [0,1].
        # The runner writes installed_versions.txt at session start (see
        # _write_env_snapshot) from INSIDE the container, capturing the
        # testbed conda env's actual installed version. The agent reads
        # this file rather than running pip itself — Claude Code's bash
        # runs on the HOST, so its `pip show` queries the wrong env.
        for env_pkg in self._pick_env_pkgs(t, k=2):
            questions[f"env_{env_pkg}"] = (
                f"Read `installed_versions.txt` (in the current "
                f"directory) and report the version listed for "
                f"`{env_pkg}`. Format: 'Version: <X.Y.Z>'. The file is "
                f"written fresh at the start of each session from the "
                f"actual /testbed conda env, so this is the ground "
                f"truth for the current session."
            )

        # ---- accumulator probe (latent revision, S7-equivalent) ----
        # Asks the agent for an INCREMENTING COUNT they should be
        # tracking across sessions. Score = |reported − ground_truth|.
        # Maps to S7's m_revision_latent_abs_err / Table 3 "accum. err"
        # column. This is the LATENT revision signal: did the agent's
        # internal count drift from reality? The COUNT is sessions
        # completed (an integer the agent can recompute or remember).
        if t >= 1:
            questions["accumulator_sessions_completed"] = (
                f"How many sessions of this chain have you COMPLETED so far "
                f"(BEFORE the current session)? The current session is "
                f"session {t}, so previously completed sessions count from "
                f"0 up to and including session {t-1}. Answer with just an "
                f"integer."
            )

        # ---- event-triggered probes (sparse sidecar, Phase 11) -----
        latest_bump_pkg = None
        latest_bump_t = -1
        for p_t in range(0, t):
            for e in events_by_session.get(p_t, []):
                if e.event_type != "dep_bump":
                    continue
                parts = (e.detail or "").split()
                if "--upgrade" in parts:
                    pkg = parts[parts.index("--upgrade") + 1]
                    if p_t > latest_bump_t:
                        latest_bump_pkg = pkg
                        latest_bump_t = p_t
        if latest_bump_pkg:
            questions["revision_event"] = (
                f"At session {latest_bump_t} an upgrade event ran "
                f"`pip install --upgrade {latest_bump_pkg}` in /testbed. "
                f"Read `installed_versions.txt` and report the version "
                f"NOW listed for `{latest_bump_pkg}`. "
                f"Format: 'Version: <X.Y.Z>'."
            )

        interference = self._interference_shared_files.get(iid)
        if interference:
            partner = interference["partner"]
            shared = interference["shared_files"]
            partner_t = next(
                (int(s["session"]) for s in self.seed_manifest.get("sessions", [])
                 if s["instance_id"] == partner),
                None,
            )
            if partner_t is not None and partner_t < t:
                files_str = ", ".join(shared)
                questions["interference_pair"] = (
                    f"DECLARED INTERFERENCE PAIR: in session {partner_t} "
                    f"you edited {files_str} for issue {partner}. Recall "
                    f"the function(s) you modified there and the kind "
                    f"of change. Be specific — your current session "
                    f"may edit the same file."
                )
        return questions

    def _orthogonal_probe_questions(
        self, t: int, iid: str,
        events_by_session: dict[int, list[LifecycleEvent]],
    ) -> dict[str, str]:
        """Phase 16: per-session orthogonal probes (one Q per mechanism).

        Each probe is a CONTRAST that isolates one mechanism by design:
        - compression_clean: a "clean" prior (non-partner, not flushed)
        - interference_partner: the declared partner (LATER member's view)
        - revision_<fact_id>: a state-change fact
        - maintenance_<prior>: same-prior at pre/post shock windows
        """
        out: dict[str, str] = {}
        if t < 1:
            return out

        # ---- Build "clean prior" pool: priors NOT involved in pairs / flushes ----
        partner_set: set[str] = set()
        for pair in (self.chain.get("interference_pairs") or []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                partner_set.update(pair)
        flush_window: set[int] = set()
        for ev in self.lifecycle_events:
            if ev.event_type == "workspace_flush":
                flush_window.update({ev.session - 1, ev.session, ev.session + 1})
        clean_priors: list[int] = []
        for p in range(0, t):
            iid_p = self._iid_at_session(p)
            if iid_p is None or iid_p in partner_set:
                continue
            if p in flush_window:
                continue
            clean_priors.append(p)
        if clean_priors:
            # Pick the deterministic-rotation choice.
            p_clean = clean_priors[(t - 1) % len(clean_priors)]
            iid_clean = self._iid_at_session(p_clean)
            out[f"compression_clean_p{p_clean}"] = (
                f"[MECHANISM=COMPRESSION] At session {p_clean} you "
                f"worked on issue {iid_clean}. Without looking at "
                f"./testbed/, recall from your notes: which file(s) "
                f"and function(s) did you edit there? One short sentence."
            )

        # ---- Interference partner (only when current iid is LATER member) ----
        interference = self._interference_shared_files.get(iid)
        if interference:
            partner = interference["partner"]
            shared = interference["shared_files"]
            partner_t = next(
                (int(s["session"]) for s in self.seed_manifest.get("sessions", [])
                 if s["instance_id"] == partner), None,
            )
            if partner_t is not None and partner_t < t:
                files_str = ", ".join(shared)
                out[f"interference_partner_p{partner_t}"] = (
                    f"[MECHANISM=INTERFERENCE] At session {partner_t} "
                    f"you edited {files_str} for issue {partner}. The "
                    f"current session edits some of the same file(s). "
                    f"Recall which SPECIFIC function(s) you modified "
                    f"in session {partner_t} (NOT this session). One "
                    f"sentence; be precise about names."
                )

        # ---- Revision: state-change beliefs ----
        for sc in (self.chain.get("state_changes") or []):
            announce_t = int(sc.get("announce_session", -1))
            if announce_t >= t:
                continue
            fact_id = sc["fact_id"]
            out[f"revision_{fact_id}"] = (
                f"[MECHANISM=REVISION] A team-convention update was "
                f"announced at session {announce_t} under fact_id "
                f"`{fact_id}`. WITHOUT consulting ./testbed/, recall "
                f"from your notes: what is the CURRENT convention for "
                f"`{fact_id}`? Give the exact wording / value, not what "
                f"it WAS before. One short answer."
            )

        # ---- Maintenance: same prior, pre/post shock window ----
        # Schedule a pair: probe SAME prior at session t-1 (pre) AND t+1 (post)
        # around a workspace_flush at session t. We schedule the POST half
        # here (the pre half was scheduled when prior session ran).
        shocks = sorted(int(ev.session) for ev in self.lifecycle_events
                        if ev.event_type == "workspace_flush")
        for shock_t in shocks:
            if t == shock_t + 1 or t == shock_t - 1:
                # Pick a stable prior to probe across the shock.
                p_target = self._iid_at_session(0)
                if p_target:
                    role = "pre" if t == shock_t - 1 else "post"
                    out[f"maintenance_shock{shock_t}_{role}"] = (
                        f"[MECHANISM=MAINTENANCE/{role.upper()}-SHOCK"
                        f"@s{shock_t}] Recall from your notes what you "
                        f"learned at SESSION 0 (issue {p_target}). One "
                        f"sentence: file(s), function(s), what change."
                    )

        return out

    def _write_env_snapshot(self, session, host_session_ws: Path) -> None:
        """Phase 15a: write `installed_versions.txt` into the agent's
        workspace, listing the version of each chain-canonical pkg as
        actually installed in the container's testbed conda env. This
        closes the architectural gap where Claude Code's bash runs on
        the HOST, so container-side pip pins/bumps were invisible.
        """
        pool = self.chain.get("dep_bump_candidates") or []
        if not pool:
            return
        pkgs_str = " ".join(pool)
        # The host-side workspace is bind-mounted into the container
        # at the same relative path beneath /agentmemory. Compute the
        # container-side path from the host path.
        try:
            rel = host_session_ws.relative_to(self.workspace_root / "agentmemory")
        except ValueError:
            return
        container_path = f"/agentmemory/{rel.as_posix()}/installed_versions.txt"
        script = (
            "PY=/opt/miniconda3/envs/testbed/bin/python; "
            "[ -x $PY ] || PY=/opt/miniconda3/bin/python; "
            f"mkdir -p $(dirname {container_path}); "
            f"> {container_path}; "
            f"for pkg in {pkgs_str}; do "
            f"  V=$($PY -m pip show $pkg 2>/dev/null | "
            f"      awk '/^Version:/ {{print $2; exit}}'); "
            f"  echo \"$pkg: ${{V:-<not installed>}}\" >> {container_path}; "
            f"done"
        )
        session.exec(script, timeout_sec=30)

    def _pick_recall_priors(self, t: int, k: int = 2) -> list[int]:
        """Pick up to k distinct prior sessions for the recall probes.

        Deterministic per-t: spreads the picks so consecutive sessions
        probe different priors AND each session covers BOTH a recent
        prior and a far prior (when available). This balances near/far
        recall in the trajectory so per-session aggregate is smooth.
        """
        if t <= 0:
            return []
        # All available priors.
        priors = list(range(t))
        if not priors:
            return []
        # Phase rotation: start index varies with t so the spread shifts.
        start = (t - 1) % len(priors)
        ordering = priors[start:] + priors[:start]
        # Prefer one recent + one far when possible.
        picks: list[int] = []
        if len(priors) >= 2:
            picks.append(t - 1)              # most recent prior
            picks.append(ordering[0] if ordering[0] != t - 1 else
                         ordering[1] if len(ordering) > 1 else ordering[0])
        else:
            picks = ordering[:k]
        # Dedup, cap at k.
        seen = set()
        out = []
        for p in picks + ordering:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            if len(out) >= k:
                break
        return out

    def _pick_env_pkgs(self, t: int, k: int = 2) -> list[str]:
        """Deterministic chain-pkg rotation: each session covers k distinct
        pkgs from the chain's dep_bump_candidates."""
        pool = self.chain.get("dep_bump_candidates") or []
        if not pool:
            return []
        n = len(pool)
        start = ((t - 1) * k) % n
        # Take k consecutive (mod) entries.
        return [pool[(start + i) % n] for i in range(min(k, n))]

    def _iid_at_session(self, t: int) -> Optional[str]:
        for s in self.seed_manifest.get("sessions", []):
            if int(s["session"]) == t:
                return s["instance_id"]
        return None

    @property
    def _scenario_dir(self) -> Path:
        """The scenario directory containing chain manifests + synthetic
        tests. Derived from the chain_path's grand-parent so the runner
        can be reused across S8 and S9 (and any future SWE-anchored
        scenarios that follow this layout)."""
        return Path(self.cfg.chain_path).parent.parent

    def _inject_synthetic_tests(self, session, session_idx: int,
                                  iid: str) -> list[str]:
        """S9 load-bearing lever: copy this session's synthetic
        consistency tests into the container and return the test_ids
        that should be added to the f2p list.

        Test ID format depends on the testing framework:
        - Django (instance_id starts with django__): tests live under
          /testbed/tests/agingbench_syn/ as Django SimpleTestCases. The
          test_id is in SWE-bench's Django format
          `<test_function> (agingbench_syn.<file_stem>.<test_class>)`.
        - Pytest-style: tests live under /testbed/_agingbench_syn/ and
          test_ids are pytest-style `path::test_id`.
        """
        scheduled = []
        for st in (self.chain.get("synthetic_tests") or []):
            if int(st.get("session", -1)) == int(session_idx):
                scheduled.append(st)
        if not scheduled:
            return []
        is_django = iid.startswith("django__")
        if is_django:
            target_dir = "/testbed/tests/agingbench_syn"
        else:
            target_dir = "/testbed/_agingbench_syn"
        session.exec(
            f"mkdir -p {target_dir} && touch {target_dir}/__init__.py",
            timeout_sec=15,
        )
        test_ids: list[str] = []
        for st in scheduled:
            module_path = st["test_module"]
            module_parts = module_path.split(".")
            host_file = self._scenario_dir.joinpath(*module_parts).with_suffix(".py")
            if not host_file.is_file():
                continue
            container_file = f"{target_dir}/{host_file.name}"
            session.cp_to_container(host_file, container_file)
            if is_django:
                # Django runtests test_id: `func (pkg.module.Class)`
                klass = st.get("test_class")
                if not klass:
                    continue
                file_stem = host_file.stem
                test_ids.append(
                    f"{st['test_function']} (agingbench_syn.{file_stem}.{klass})"
                )
            else:
                test_ids.append(
                    f"_agingbench_syn/{host_file.name}::{st['test_function']}"
                )
        return test_ids

    def _apply_pre_event(self, event: LifecycleEvent) -> dict:
        """Apply a pre-session event without needing an open session.

        Currently used for workspace_flush which is host-side (operates on
        the mounted memory dir directly).
        """
        if event.event_type == "workspace_flush":
            aging_dir = (self.workspace_root / "agentmemory" / ".aging")
            bytes_freed = 0
            if aging_dir.exists():
                for f in aging_dir.rglob("*"):
                    if f.is_file():
                        bytes_freed += f.stat().st_size
                import shutil
                shutil.rmtree(aging_dir)
            return {
                "event_type": "workspace_flush",
                "session": event.session,
                "outcome": "ok",
                "bytes_freed": bytes_freed,
                "applied_when": "pre_session",
            }
        return {"event_type": event.event_type, "outcome": "skipped_pre",
                "applied_when": "pre_session"}

    @staticmethod
    def _stub_agent_step(session: S8DockerSession,
                         session_meta: dict,
                         chain_entry: Optional[dict]) -> None:
        """Stub agent step — used when no agent.adapter is configured.

        Writes a minimal note to /agentmemory/.aging/notes.md so we can
        verify memory persistence works end-to-end without API access.
        """
        line = (
            f"\n## session {session_meta['session']} — {session_meta['instance_id']}\n"
            f"role: {(chain_entry or {}).get('chain_role', 'n/a')}\n"
            f"summary: {(chain_entry or {}).get('summary', 'n/a')}\n"
            f"_phase 2 stub agent — set agent.adapter in SUT yaml to engage real agent._\n"
        )
        existing = session.read_memory_file(".aging/notes.md") or ""
        session.write_memory_file(".aging/notes.md", existing + line)
        # Also poke the container to confirm exec works.
        session.exec("ls /testbed | head -3", timeout_sec=15)

    def _real_agent_step(self,
                         session: S8DockerSession,
                         session_meta: dict,
                         chain_entry: Optional[dict],
                         memory_dir: Path) -> dict:
        """Phase 3 + Phase 4: real-agent + verifier.

        After agent produces solution.diff, apply it inside the container
        and run the SWE-bench-Verified test suite. Returns a dict with
        both agent metadata + verification result.
        """
        from agingbench.scenarios.s8_swe_bench.agent import S8AgentRequest
        from agingbench.scenarios.s8_swe_bench.verifier import (
            apply_diff_in_container,
            run_verification,
            get_instance_metadata,
        )

        sess_idx = int(session_meta["session"])
        iid = session_meta["instance_id"]
        host_session_ws = memory_dir / "agent_work" / f"session_{sess_idx}_{iid}"
        host_session_ws.mkdir(parents=True, exist_ok=True)

        prior_notes_path = memory_dir / ".aging" / "notes.md"
        prior_notes = ""
        if prior_notes_path.is_file():
            prior_notes = prior_notes_path.read_text(encoding="utf-8", errors="replace")

        issue_text = self._issue_text(iid)
        # Phase 16b: prepend state_change notices announced at <= this session.
        # The agent reads ISSUE.md and must remember the new convention.
        # Notices are NOT in /testbed source -> the agent can't re-derive them.
        state_changes = self.chain.get("state_changes") or []
        notices_to_prepend = [
            sc for sc in state_changes
            if int(sc.get("announce_session", -1)) == int(sess_idx)
        ]
        if notices_to_prepend:
            preamble = "\n\n".join(
                f"### STATE CHANGE NOTICE — fact_id={sc['fact_id']}\n{sc['notice']}"
                for sc in notices_to_prepend
            )
            issue_text = preamble + "\n\n---\n\n" + issue_text

        # Copy /testbed (the repo at base_commit + test_patch) from the
        # container to the host so the agent has read access. The agent
        # operates on this copy to derive correct line numbers; we apply
        # the agent's solution.diff inside the container, never against
        # this copy. ~50 MB per session for sphinx.
        # Phase 15c: SUT yaml can set `memory_policy.disable_testbed_access`
        # to FORCE the agent to rely on its own notes (no source-code re-
        # derivation). This is the cleanest Tier-2 memory-stress condition.
        sut_mp = (self.cfg.sut_cfg or {}).get("memory_policy") or {}
        disable_testbed = bool(sut_mp.get("disable_testbed_access", False))
        testbed_host = host_session_ws / "testbed"
        cp_result = None
        if not disable_testbed and not testbed_host.exists():
            cp_result = session.cp_to_host("/testbed/.", testbed_host)

        # Phase 15a: write installed_versions.txt from INSIDE the container
        # (testbed conda env) into the agent's workspace. Closes the
        # architectural gap where Claude Code's bash ran on the HOST, not
        # in the container, making container-side pip pins invisible.
        self._write_env_snapshot(session, host_session_ws)

        attestation_qs = self._build_attestation_questions(
            t=sess_idx, iid=iid,
            events_by_session=getattr(self, "_events_by_session", {}),
        )

        request = S8AgentRequest(
            session_idx=sess_idx,
            instance_id=iid,
            issue_text=issue_text,
            chain_role=(chain_entry or {}).get("chain_role", "n/a"),
            chain_summary=(chain_entry or {}).get("summary", "n/a"),
            prior_notes=prior_notes,
            host_workspace_dir=host_session_ws,
            persistent_notes_path=prior_notes_path,
            attestation_questions=attestation_qs,
            testbed_disabled=disable_testbed,
        )
        resp = self._agent_runner.run(request)

        # ---- Phase 4: apply diff + run verification ----
        meta = get_instance_metadata(iid)
        f2p = list(meta.get("fail_to_pass") or [])
        p2p = list(meta.get("pass_to_pass") or [])

        apply_result = apply_diff_in_container(session, resp.solution_diff_text)
        # S9: inject synthetic consistency tests INTO f2p (post-patch so
        # they inspect the agent's modified Django).
        synthetic_test_ids = self._inject_synthetic_tests(session, sess_idx, iid)
        if synthetic_test_ids:
            f2p = list(f2p) + synthetic_test_ids
        if apply_result.success and (f2p or p2p):
            verify = run_verification(
                session, fail_to_pass=f2p, pass_to_pass=p2p,
                test_patch=meta.get("test_patch"),
            )
        else:
            from agingbench.scenarios.s8_swe_bench.verifier import VerifyResult
            verify = VerifyResult(
                passed=False,
                n_fail_to_pass_total=len(f2p),
                n_fail_to_pass_passed=0,
                n_pass_to_pass_total=len(p2p),
                n_pass_to_pass_passed=0,
                error=("patch_apply_failed: " + apply_result.stderr[:200])
                       if not apply_result.success else "no_tests_configured",
            )

        return {
            "adapter": resp.adapter_kind,
            "model": resp.model,
            "success": resp.success,
            "duration_sec": resp.duration_sec,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "produced_solution_diff": resp.solution_diff_text is not None,
            "solution_diff_bytes": (
                len(resp.solution_diff_text) if resp.solution_diff_text else 0
            ),
            "solution_diff_text": resp.solution_diff_text,   # needed by interference probe
            "produced_notes_update": resp.new_notes_text is not None,
            "notes_bytes_after": (
                len(resp.new_notes_text) if resp.new_notes_text else 0
            ),
            "files_changed_count": len(resp.files_changed),
            "error": resp.error,
            "attestation_questions": attestation_qs,
            "attestation_text": resp.attestation_text,
            "testbed_cp": {
                "success": (cp_result.exit_code == 0) if cp_result else None,
                "duration_sec": cp_result.duration_sec if cp_result else None,
                "host_path": str(testbed_host),
            },
            "patch_apply": {
                "success": apply_result.success,
                "exit_code": apply_result.exit_code,
                "diff_bytes": apply_result.diff_bytes,
                "stderr_excerpt": apply_result.stderr[:300] if apply_result.stderr else "",
            },
            "verification": {
                "passed": verify.passed,
                "n_fail_to_pass_total": verify.n_fail_to_pass_total,
                "n_fail_to_pass_passed": verify.n_fail_to_pass_passed,
                "n_pass_to_pass_total": verify.n_pass_to_pass_total,
                "n_pass_to_pass_passed": verify.n_pass_to_pass_passed,
                "duration_sec": verify.duration_sec,
                "error": verify.error,
                "log_excerpt": (verify.raw_log[:1500] if verify.raw_log else ""),
            },
        }

    _ISSUE_CACHE: dict = {}

    def _issue_text(self, instance_id: str) -> str:
        """Fetch issue.problem_statement from SWE-bench-Verified, cached."""
        if instance_id in self._ISSUE_CACHE:
            return self._ISSUE_CACHE[instance_id]
        try:
            from datasets import load_dataset
            ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            for row in ds:
                self._ISSUE_CACHE[row["instance_id"]] = row["problem_statement"]
            return self._ISSUE_CACHE.get(instance_id, "(issue text unavailable)")
        except Exception:                                       # noqa: BLE001
            return f"(issue text load failed for {instance_id})"


def _du(path: Path) -> int:
    """Disk usage of a directory, in bytes."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total

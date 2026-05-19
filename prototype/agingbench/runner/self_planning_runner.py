"""
agingbench/runner/self_planning_runner.py — Layer 2 Self-Planning Runner.

Unlike Layer 1 runners where the runner feeds tasks one-by-one,
this runner:
  1. Gives the agent a high-level goal and environment tools
  2. Lets the agent plan and execute autonomously
  3. After the agent finishes, runs recall probes (same as Layer 1)
  4. Measures the same aging curves

Supported scenarios: S2 (lifestyle), S3 (knowledge base), S6 (naturalistic).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .base import BaseRunner, RunResult
from .trace import TraceLogger
from ..metrics.aging import AgingCurve
from ..core.memory.base import MemoryPolicy
from ..core.agent import AgentInterface, ReferenceAgent
from ..core.tools import ToolSpec, ToolRegistry


# Goal templates per scenario
_GOAL_TEMPLATES = {
    "s2": (
        "You are a personal lifestyle assistant for {user_name}. "
        "This is session {session_idx}. The user has the following profile "
        "and constraints that you MUST follow:\n\n{profile_text}\n\n"
        "{update_text}"
        "The user has the following requests for this session:\n"
        "{tasks_text}\n\n"
        "Handle each request. For every recommendation, check the user's "
        "constraints first using the check_constraints tool. Use search_memory "
        "to recall past interactions when relevant."
    ),
    "s3": (
        "You are maintaining a project knowledge base. This is session {session_idx}. "
        "A new meeting has occurred. Here is the transcript:\n\n{transcript}\n\n"
        "Your tasks:\n"
        "1. Update the knowledge base with key decisions from this meeting.\n"
        "2. Answer the following team queries:\n{queries_text}\n\n"
        "Use search_memory to check what's already recorded. Use update_db to "
        "store new decisions."
    ),
    "s6": (
        "You are a research analyst assistant. This is session {session_idx}. "
        "New data is available from the {domain} domain:\n\n"
        "{environment_data}\n\n"
        "Your task: {task_text}\n\n"
        "Use search_memory to recall findings from previous sessions. "
        "Be precise with names, numbers, and specific details."
    ),
}


class SelfPlanningRunner(BaseRunner):
    """
    Layer 2 Self-Planning Runner.

    Wraps S2, S3, or S6 scenario data but gives the agent autonomy
    to plan its own approach rather than receiving tasks one-by-one.
    """

    SCENARIO_ID = "self_planning"

    def __init__(
        self,
        wrapped_scenario: str,          # "s2", "s3", or "s6"
        memory_policy: MemoryPolicy,
        llm,
        tracer: TraceLogger,
        sut_id: str = "unknown",
        oracle_mode: bool = False,
        agent_class: type[AgentInterface] = ReferenceAgent,
        generated_data: dict | None = None,
        curated_data: dict | None = None,
    ):
        self.wrapped_scenario = wrapped_scenario
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        if self.llm is not None:
            self.llm.tracer = self.tracer
        self.sut_id = sut_id
        self.oracle_mode = oracle_mode
        self.agent_class = agent_class

        self._model_id = getattr(llm, "model_id", None) or getattr(llm, "model", "unknown")
        self._provider = "local_hf" if hasattr(llm, "tok") else "litellm"

        # Load scenario data
        self._load_scenario_data(generated_data, curated_data)

        # In-memory fact database (for query_db / update_db tools)
        self._db: dict[str, str] = {}

    def _load_scenario_data(self, generated_data, curated_data):
        """Load data from generator output or curated files."""
        data = generated_data or curated_data or {}

        if self.wrapped_scenario == "s2":
            self.profile = data.get("source_profile", {})
            self.profile_text = self.profile.get("profile_text", "")
            self.sessions = data.get("session_tasks", {}).get("sessions", [])
            self.eval_probes = data.get("eval_probes", {}).get("probes", [])
            self.constraint_updates = data.get("constraint_updates", {}).get("updates", [])
            self.session_facts = data.get("session_facts", {}).get("facts", [])
            self._update_schedule = {u["session"]: u for u in self.constraint_updates}
        elif self.wrapped_scenario == "s3":
            self.transcripts = data.get("transcripts", {}).get("sessions", [])
            self.gold_decisions = data.get("gold_timeline", {}).get("decisions", [])
            self.queries = data.get("queries", {}).get("sessions", [])
        elif self.wrapped_scenario == "s6":
            session_tasks = data.get("session_tasks", {})
            self.sessions = session_tasks.get("sessions", [])
            self.system_prompt = session_tasks.get("system_prompt", "")

    def run(self, n_sessions: int = 10, seed: int = 42) -> dict:
        """Run the self-planning loop."""
        self.memory_policy.reset()

        task_scores = []
        recall_scores = []
        recall_matrix: dict[int, dict[int, float]] = {}
        session_results = []

        run_span = self.tracer.log(
            "run_start", parent_span_id=None,
            sut_id=self.sut_id, scenario=f"self_planning_{self.wrapped_scenario}",
            seed=seed, n_sessions=n_sessions,
            policy=type(self.memory_policy).__name__,
            oracle_mode=self.oracle_mode,
        )

        for t in range(n_sessions):
            session_span = self.tracer.log(
                "session_start", parent_span_id=run_span,
                session=t, mode="self_planning",
            )

            # Build goal prompt for this session
            goal = self._build_goal(t)

            # Build tools
            tools = self._build_tools()

            # Create agent with higher turn budget for autonomous planning
            agent = self.agent_class(
                llm=self.llm,
                memory_policy=self.memory_policy,
                tools=tools,
                max_turns=12,
            )

            # Let agent plan and execute
            result = agent.run_session(goal, session_id=t)
            agent_output = result.get("output", "")

            # Score primary task
            task_score = self._score_task(t, agent_output)
            task_scores.append((t, task_score))

            # Run recall probes from prior sessions
            probe_results = self._run_recall_probes(agent, t)
            if probe_results["n_total"] > 0:
                recall_scores.append((t, probe_results["recall_rate"]))
                recall_matrix[t] = probe_results.get("per_session", {})
            else:
                recall_scores.append((t, 1.0))

            # Write to memory
            interaction = self._build_interaction_text(t, goal, agent_output)
            self.memory_policy.write(interaction, llm=self.llm)

            self.tracer.log(
                "session_scored", parent_span_id=session_span,
                session=t, task_score=task_score,
                recall_rate=probe_results["recall_rate"],
                n_probes=probe_results["n_total"],
            )

            session_results.append({
                "session": t,
                "task_score": task_score,
                "recall_rate": probe_results["recall_rate"],
                "n_probes": probe_results["n_total"],
                "agent_turns": result.get("turns", 0),
                "tool_calls": len(result.get("tool_calls", [])),
            })

            print(f"  [SP-{self.wrapped_scenario}] Session {t:2d}  "
                  f"task={task_score:.3f}  recall={probe_results['recall_rate']:.3f}  "
                  f"({probe_results['n_recalled']}/{probe_results['n_total']} probes)  "
                  f"turns={result.get('turns', 0)}")

        # Build aging curves
        task_curve = AgingCurve(
            exposures=[t for t, _ in task_scores],
            scores=[s for _, s in task_scores],
            scenario=f"self_planning_{self.wrapped_scenario}",
            sut_id=self.sut_id,
        )
        recall_curve = AgingCurve(
            exposures=[t for t, _ in recall_scores],
            scores=[s for _, s in recall_scores],
            scenario=f"self_planning_{self.wrapped_scenario}",
            sut_id=self.sut_id,
        )

        # Compute lag curves
        lag_curves = self._compute_lag_curves(recall_matrix)

        return {
            "task_curve": task_curve,
            "recall_curve": recall_curve,
            "task_raw": task_scores,
            "recall_raw": recall_scores,
            "recall_matrix": recall_matrix,
            "lag_curves": lag_curves,
            "session_results": session_results,
        }

    # ------------------------------------------------------------------
    # Goal building
    # ------------------------------------------------------------------

    def _build_goal(self, session_idx: int) -> str:
        """Build a high-level goal prompt for the agent."""
        if self.wrapped_scenario == "s2":
            return self._build_s2_goal(session_idx)
        elif self.wrapped_scenario == "s3":
            return self._build_s3_goal(session_idx)
        elif self.wrapped_scenario == "s6":
            return self._build_s6_goal(session_idx)
        return f"Session {session_idx}: complete the assigned tasks."

    def _build_s2_goal(self, t: int) -> str:
        session_data = self.sessions[t] if t < len(self.sessions) else self.sessions[-1]
        tasks = session_data.get("tasks", [])
        tasks_text = "\n".join(f"- {task['text']}" for task in tasks)

        update_text = ""
        update = self._update_schedule.get(t)
        if update:
            update_text = f"IMPORTANT UPDATE: {update['update_text']}\n\n"

        # Inject session fact so it enters memory
        fact_text = ""
        for fact in self.session_facts:
            if fact["session"] == t:
                fact_text = f"\nNote from this session: {fact['text']}\n"
                break

        return _GOAL_TEMPLATES["s2"].format(
            user_name=self.profile.get("user_name", "the user"),
            session_idx=t,
            profile_text=self.profile_text[:1000],
            update_text=update_text,
            tasks_text=tasks_text,
        ) + fact_text

    def _build_s3_goal(self, t: int) -> str:
        transcript = self.transcripts[t]["transcript"] if t < len(self.transcripts) else ""
        queries = self.queries[t]["queries"] if t < len(self.queries) else []
        queries_text = "\n".join(f"- {q['question']}" for q in queries)

        return _GOAL_TEMPLATES["s3"].format(
            session_idx=t,
            transcript=transcript[:2000],
            queries_text=queries_text,
        )

    def _build_s6_goal(self, t: int) -> str:
        session = self.sessions[t] if t < len(self.sessions) else self.sessions[-1]
        return _GOAL_TEMPLATES["s6"].format(
            session_idx=t,
            domain=session.get("domain", "unknown"),
            environment_data=session.get("environment_data", "")[:2000],
            task_text=session["task"]["text"],
        )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        registry = ToolRegistry()

        # search_memory
        registry.register(ToolSpec(
            name="search_memory",
            version="1.0.0",
            description="Search your memory for information from previous sessions.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            fn=lambda args: {"result": self.memory_policy.read() or "(empty)"},
        ))

        # query_db
        registry.register(ToolSpec(
            name="query_db",
            version="1.0.0",
            description="Query the environment database for information.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            fn=lambda args: self._handle_query(args.get("query", "")),
        ))

        # check_constraints (S2 only)
        if self.wrapped_scenario == "s2":
            registry.register(ToolSpec(
                name="check_constraints",
                version="1.0.0",
                description="Check user constraints for a given category before making recommendations.",
                parameters={
                    "type": "object",
                    "properties": {"category": {"type": "string"}},
                    "required": ["category"],
                },
                fn=lambda args: self._handle_check_constraints(args.get("category", "")),
            ))

        return registry

    def _handle_query(self, query: str) -> dict:
        """Keyword search over the in-memory fact DB."""
        query_lower = query.lower()
        matches = []
        for key, value in self._db.items():
            if query_lower in key.lower() or query_lower in value.lower():
                matches.append(f"{key}: {value}")
        if matches:
            return {"result": "\n".join(matches[:5])}
        return {"result": "(no matching records found)"}

    def _handle_check_constraints(self, category: str) -> dict:
        """Return constraints for the given category."""
        constraints = self.profile.get("constraints", [])
        relevant = [c for c in constraints if c.get("category", "") == category]
        if relevant:
            rules = "\n".join(f"- [{c['id']}] {c['rule']}" for c in relevant)
            return {"result": rules}
        return {"result": f"(no constraints found for category '{category}')"}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_task(self, t: int, output: str) -> float:
        """Score the primary task output."""
        if self.wrapped_scenario == "s6":
            session = self.sessions[t] if t < len(self.sessions) else self.sessions[-1]
            keywords = session["task"].get("eval_keywords", [])
            if not keywords:
                return 1.0
            found = sum(1 for kw in keywords if kw.lower() in output.lower())
            return found / len(keywords)
        # For S2/S3, scoring is more complex — use 1.0 as placeholder
        # (the real aging signal comes from recall probes)
        return 1.0

    def _run_recall_probes(self, agent: AgentInterface, current_session: int) -> dict:
        """Run recall probes from all prior sessions."""
        if self.wrapped_scenario == "s6":
            return self._run_s6_recall_probes(agent, current_session)
        elif self.wrapped_scenario == "s2":
            return self._run_s2_recall_probes(agent, current_session)
        elif self.wrapped_scenario == "s3":
            return self._run_s3_recall_probes(agent, current_session)
        return {"recall_rate": 1.0, "n_recalled": 0, "n_total": 0, "per_session": {}}

    def _run_s6_recall_probes(self, agent: AgentInterface, t: int) -> dict:
        """S6-style recall probes: test facts from all prior sessions."""
        n_recalled = 0
        n_total = 0
        per_session: dict[int, float] = {}

        for s in range(t):
            session = self.sessions[s] if s < len(self.sessions) else None
            if not session or session.get("is_cross_reference"):
                continue
            probes = session.get("recall_probes", [])
            if not probes:
                continue

            session_hits = 0
            for probe in probes:
                result = agent.run_session(
                    f"From your memory of our past research sessions, answer: {probe['question']}",
                    session_id=t,
                )
                probe_output = result.get("output", "")
                keywords = probe.get("keywords", [])
                hit = any(kw.lower() in probe_output.lower() for kw in keywords)
                if hit:
                    n_recalled += 1
                    session_hits += 1
                n_total += 1

            per_session[s] = session_hits / len(probes) if probes else 0

        recall_rate = n_recalled / n_total if n_total > 0 else 1.0
        return {
            "recall_rate": recall_rate,
            "n_recalled": n_recalled,
            "n_total": n_total,
            "per_session": per_session,
        }

    def _run_s2_recall_probes(self, agent: AgentInterface, t: int) -> dict:
        """S2-style recall probes: test session facts from prior sessions."""
        n_recalled = 0
        n_total = 0
        per_session: dict[int, float] = {}

        for fact in self.session_facts:
            if fact["session"] >= t:
                continue
            result = agent.run_session(
                f"From your memory of past interactions, answer: {fact['recall_question']}",
                session_id=t,
            )
            output = result.get("output", "")
            keywords = fact.get("recall_keywords", [])
            hit = any(kw.lower() in output.lower() for kw in keywords)
            if hit:
                n_recalled += 1
            n_total += 1
            per_session[fact["session"]] = 1.0 if hit else 0.0

        recall_rate = n_recalled / n_total if n_total > 0 else 1.0
        return {
            "recall_rate": recall_rate,
            "n_recalled": n_recalled,
            "n_total": n_total,
            "per_session": per_session,
        }

    def _run_s3_recall_probes(self, agent: AgentInterface, t: int) -> dict:
        """S3-style recall probes: test gold decisions from prior sessions."""
        n_recalled = 0
        n_total = 0
        per_session: dict[int, float] = {}

        # Sample up to 2 decisions per prior session to keep probe count manageable
        decisions_by_session: dict[int, list] = {}
        for d in self.gold_decisions:
            if d["session"] < t:
                decisions_by_session.setdefault(d["session"], []).append(d)

        for s, decisions in sorted(decisions_by_session.items()):
            probes = decisions[:2]  # max 2 probes per session
            session_hits = 0
            for d in probes:
                result = agent.run_session(
                    f"From your memory of past meetings, answer: What was decided about {d['category']}? "
                    f"Specifically: {d['fact'].split()[0]} {d['fact'].split()[1] if len(d['fact'].split()) > 1 else ''}",
                    session_id=t,
                )
                output = result.get("output", "")
                keywords = d.get("keywords", [])
                hit = any(kw.lower() in output.lower() for kw in keywords)
                if hit:
                    n_recalled += 1
                    session_hits += 1
                n_total += 1

            per_session[s] = session_hits / len(probes) if probes else 0

        recall_rate = n_recalled / n_total if n_total > 0 else 1.0
        return {
            "recall_rate": recall_rate,
            "n_recalled": n_recalled,
            "n_total": n_total,
            "per_session": per_session,
        }

    # ------------------------------------------------------------------
    # Interaction text and lag curves
    # ------------------------------------------------------------------

    def _build_interaction_text(self, t: int, goal: str, output: str) -> str:
        """Build interaction text for memory storage."""
        # Include session fact explicitly so it enters memory
        fact_line = ""
        if self.wrapped_scenario == "s2":
            for fact in self.session_facts:
                if fact["session"] == t:
                    fact_line = f"Session note: {fact['text']}\n"
                    break
        return (
            f"=== Session {t} (self-planned) ===\n"
            f"Goal: {goal[:500]}\n\n"
            f"Agent output: {output[:800]}\n"
            f"{fact_line}"
        )

    def _compute_lag_curves(self, recall_matrix: dict) -> dict:
        """Compute lag curves from the recall matrix."""
        lag_data: dict[int, list[tuple[int, float]]] = {}
        for t, row in recall_matrix.items():
            for s, rate in row.items():
                lag = t - s
                if lag > 0:
                    lag_data.setdefault(lag, []).append((t, rate))
        return lag_data

"""
agingbench/runner/s4_runner_unified.py — S4 with the unified FullReactAgent.

Same contract as S3UnifiedRunner, but for S4's revision axis:
  * memory is **tool-only** (search_memory) — NO in-prompt dump (unlike the
    default S4Runner, which interpolates memory_text into the coding prompt and
    uses ReferenceAgent).
  * each session writes its dependency facts — including the revised
    `config_value` ("current max length is N") carried in dependency_context —
    to memory, building the version history.
  * a held-out version probe then asks for the CURRENT value WITHOUT restating
    it; the agent must retrieve the latest from memory via search_memory.

Scores version_accuracy = cite-latest rate (the agent has the versions in
memory; does it report the current one?). This is the revision-aging signal,
measured under the tool-memory FullReactAgent rather than in-prompt ReferenceAgent.
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

from .base import BaseRunner
from ..core.full_react_agent import FullReactAgent, build_search_memory_tool
from ..core.tools import ToolRegistry
from ..core.memory.base import MemoryPolicy
from ..metrics.aging import AgingCurve


class S4UnifiedRunner(BaseRunner):
    """S4 (Software Engineering) revision axis using FullReactAgent + search_memory."""

    SCENARIO_ID = "s4_software_engineering"

    def __init__(
        self,
        memory_policy: MemoryPolicy,
        llm,
        tracer,
        sut_id: str = "unknown",
        generated_data: Optional[dict] = None,
        agent_max_turns: int = 6,
        search_memory_top_k: int = 12,
    ):
        self.memory_policy = memory_policy
        self.llm = llm
        self.tracer = tracer
        self.sut_id = sut_id
        self.generated_data = generated_data
        self.agent_max_turns = agent_max_turns
        self.search_memory_top_k = search_memory_top_k
        self._model_id = getattr(llm, "model_id", getattr(llm, "model", "?"))
        self._provider = getattr(llm, "_provider", getattr(llm, "provider", "?"))

    def run(self, n_sessions: int = 15, seed: int = 42) -> dict:
        import random as _random
        _random.seed(seed)
        self.memory_policy.reset()
        is_native = "gpt-oss" in str(self._model_id).lower()

        progress_on = os.getenv("AGINGBENCH_S4_PROGRESS", "1").lower() not in {"0", "false", "no", "off"}
        run_t0 = time.time()

        def _progress(msg: str):
            if progress_on:
                d = int(time.time() - run_t0); print(f"  [S4/unified][{d//60:02d}:{d%60:02d}] {msg}", flush=True)

        sessions = (self.generated_data or {}).get("tasks", {}).get("sessions", [])
        n = min(n_sessions, len(sessions))
        run_span = self.tracer.log(
            "run_start", parent_span_id=None, sut_id=self.sut_id, scenario=self.SCENARIO_ID,
            seed=seed, n_sessions=n, policy=type(self.memory_policy).__name__,
            agent="full_react_agent", agent_max_turns=self.agent_max_turns,
        )
        _progress(f"start: sessions={n}, policy={type(self.memory_policy).__name__}, "
                  f"agent=FullReactAgent(max_turns={self.agent_max_turns}, native_tools={is_native}), "
                  f"search_memory_top_k={self.search_memory_top_k}")

        version_acc_raw: list[tuple[int, float]] = []
        probe_results: list[dict] = []
        session_results: list[dict] = []

        for t in range(n):
            task = sessions[t]
            sess_span = self.tracer.log("session_start", parent_span_id=run_span, session=t)

            # 1) Persist this session's facts (task + dependency_context w/ the
            #    revised config_value) to memory — builds the version history.
            content = f"Session {t}: {task.get('task','')}\n{task.get('dependency_context','')}".strip()
            self.memory_policy.write(content, llm=self.llm)

            # 2) Held-out version probe via FullReactAgent (tool-only memory).
            dp = task.get("dependency_probe")
            hit = None
            if dp:
                ek = dp.get("eval_keywords")
                ek = eval(ek) if isinstance(ek, str) else (ek or [])
                gold = [str(k) for k in ek if re.fullmatch(r"\d{3,}", str(k))]
                if gold:
                    tools = ToolRegistry()
                    tools.register(build_search_memory_tool(self.memory_policy, top_k=self.search_memory_top_k))
                    agent = FullReactAgent(llm=self.llm, memory_policy=self.memory_policy,
                                           tools=tools, max_turns=self.agent_max_turns,
                                           native_tools=is_native)
                    probe = (f"{dp['question']}\n\nUse search_memory to find the CURRENT value, "
                             f"then answer with just the number.")
                    res = agent.run_session(probe, session_id=t)
                    cited = set(re.findall(r"\d{3,}", res["output"]))
                    searched = any((c.get("tool") or c.get("name")) == "search_memory"
                                   for c in res.get("tool_calls", []))
                    hit = 1.0 if any(g in cited for g in gold) else 0.0
                    version_acc_raw.append((t, hit))
                    pr = {"session": t, "gold_latest": gold, "cited": sorted(cited)[:6],
                          "searched": searched, "hit": hit, "answer": res["output"][:140]}
                    probe_results.append(pr)
                    self.tracer.log("version_probe", parent_span_id=sess_span, session=t,
                                    gold=gold, hit=hit, searched=searched,
                                    n_tool_calls=len(res.get("tool_calls", [])))
                    _progress(f"s{t}: probe gold={gold} cited={sorted(cited)[:4]} "
                              f"{'LATEST' if hit else 'STALE'} searched={'y' if searched else 'n'}")

            session_results.append({"session": t,
                                    "dependency_probe_result": (probe_results[-1] if hit is not None else None)})
            self.tracer.log("session_end", parent_span_id=sess_span, session=t)

        version_accuracy = (sum(h for _, h in version_acc_raw) / len(version_acc_raw)
                            if version_acc_raw else None)
        searched_rate = (sum(1 for p in probe_results if p["searched"]) / len(probe_results)
                         if probe_results else 0.0)
        version_curve = AgingCurve(
            exposures=[e for e, _ in version_acc_raw], scores=[s for _, s in version_acc_raw],
            scenario=self.SCENARIO_ID, sut_id=self.sut_id, metric_name="version_accuracy",
        )

        # Optional: full dependency_metrics (chain_recall_by_version_depth, etc.)
        dep_metrics = None
        if self.generated_data and "dependency_graph" in self.generated_data:
            try:
                from ..metrics.dependency_scorer import score_dependency_chain
                dep_metrics = score_dependency_chain(session_results, self.generated_data["dependency_graph"])
            except Exception as e:  # noqa: BLE001
                dep_metrics = {"error": str(e)}

        self.tracer.log("run_end", parent_span_id=run_span,
                        version_accuracy=version_accuracy, n_probes=len(version_acc_raw))
        _progress(f"done: version_accuracy={version_accuracy}  search_use_rate={searched_rate:.2f}  "
                  f"n_probes={len(version_acc_raw)}")

        return {
            "version_accuracy": version_accuracy,
            "search_use_rate": searched_rate,
            "version_accuracy_raw": version_acc_raw,
            "version_curve": version_curve,
            "probe_results": probe_results,
            "session_results": session_results,
            "dependency_metrics": dep_metrics,
            "agent": "FullReactAgent",
            "n_probes": len(version_acc_raw),
        }

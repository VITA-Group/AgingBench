"""
agingbench/runner/diagnostic_mixin.py — P1/P2/P3 diagnostic evaluation mixin.

Provides ``run_diagnostic_probes()`` that any scenario runner can call during
its recall-probe loop.  Evaluates each probe under all three conditions:

  P1 (Baseline):         agent.run_session(probe)  — native W + R + U
  P2 (Oracle Retrieval): LLM(dump_store + probe)   — native W, bypass R
  P3 (Oracle Context):   LLM(gold_facts + probe)   — bypass W + R, test U

Produces per-session DiagnosticResult and aggregated error partition.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..diagnostics.oracle_evaluator import evaluate_p2, evaluate_p3
from ..diagnostics.partitioner import DiagnosticResult


class DiagnosticMixin:
    """Mixin that adds P1/P2/P3 diagnostic evaluation to any runner.

    The host runner must set ``self.diagnose: bool`` and provide
    ``self.llm`` and ``self.memory_policy`` attributes.
    """

    def run_diagnostic_probes(
        self,
        probes: list[dict],
        session_idx: int,
        agent,
        score_fn: Callable[[str, dict], dict],
        gold_facts: str,
        p1_results: Optional[list[dict]] = None,
    ) -> dict:
        """Run probes under P1/P2/P3 and return partitioned scores.

        Parameters
        ----------
        probes : list[dict]
            Recall probes with at least ``question`` and ``keywords`` keys.
        session_idx : int
            Current session index.
        agent : AgentInterface
            The agent instance for P1 (normal execution).  If ``p1_results``
            is already provided, the agent is not called again for P1.
        score_fn : callable(agent_output: str, probe: dict) -> dict
            Scoring function that returns a dict containing a ``recalled``
            key (1 or 0).  Typically ``score_recall_probe``.
        gold_facts : str
            Ground-truth facts text for P3 (oracle context).  Should cover
            facts from all sessions 0..session_idx-1.
        p1_results : list[dict], optional
            Pre-computed P1 (baseline) results from the normal probe loop.
            If provided, P1 is not re-evaluated — avoids duplicate LLM calls.

        Returns
        -------
        dict with:
            p1_results : list[dict]  — per-probe P1 scores
            p2_results : list[dict]  — per-probe P2 scores
            p3_results : list[dict]  — per-probe P3 scores
            partition   : dict       — DiagnosticResult.to_dict()
        """
        if not probes:
            return {
                "p1_results": [],
                "p2_results": [],
                "p3_results": [],
                "partition": DiagnosticResult(
                    session=session_idx, acc_p1=1.0, acc_p2=1.0, acc_p3=1.0,
                ).to_dict(),
            }

        # ---- P1: Baseline (reuse pre-computed results if available) ----
        if p1_results is not None:
            scored_p1 = p1_results
        else:
            scored_p1 = []
            for probe in probes:
                result = agent.run_session(probe["question"], session_id=session_idx)
                scored_p1.append(score_fn(result["output"], probe))

        # ---- P2: Oracle Retrieval from agent's actual store ----
        store_text = self.memory_policy.dump_store()  # type: ignore[attr-defined]
        scored_p2 = []
        for probe in probes:
            p2_answer = evaluate_p2(self.llm, probe["question"], store_text)  # type: ignore[attr-defined]
            scored_p2.append(score_fn(p2_answer, probe))

        # ---- P3: Oracle Context (ground truth) ----
        scored_p3 = []
        for probe in probes:
            p3_answer = evaluate_p3(self.llm, probe["question"], gold_facts)  # type: ignore[attr-defined]
            scored_p3.append(score_fn(p3_answer, probe))

        # ---- Compute accuracies ----
        def _acc(results: list[dict]) -> float:
            if not results:
                return 1.0
            return sum(r["recalled"] for r in results) / len(results)

        acc_p1 = _acc(scored_p1)
        acc_p2 = _acc(scored_p2)
        acc_p3 = _acc(scored_p3)

        partition = DiagnosticResult(
            session=session_idx,
            acc_p1=round(acc_p1, 4),
            acc_p2=round(acc_p2, 4),
            acc_p3=round(acc_p3, 4),
            n_probes=len(probes),
        )

        return {
            "p1_results": scored_p1,
            "p2_results": scored_p2,
            "p3_results": scored_p3,
            "partition": partition.to_dict(),
        }

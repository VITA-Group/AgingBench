"""Programmatic generator for S1 Research Literature scenario.

Produces paper/sprint batches with unique per-cycle keywords,
probes, and session facts — matching the curated JSON format.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator
from .dependency_mixin import DependencyMixin
from .fact_graph import FactGraph
from .pools import (
    FIRST_NAMES, LAST_NAMES, TECH_FRAMEWORKS, PROJECT_COMPONENTS,
    random_dollar, random_percent, random_latency_ms, random_count,
    random_date, ensure_non_round, random_person, sample_unique,
)
from .pressure_config import PressureConfig

# Sprint/paper batch content templates
_BATCH_TEMPLATES = [
    (
        "{component} Integration Results",
        "{component} integration completed. Test coverage: {percent}. "
        "Response time: {latency}ms at p50. Memory usage: {memory}GB. "
        "Processed {throughput} requests in load test. "
        "Budget spent: ${spent} of ${total} allocated."
    ),
    (
        "{component} Security Audit",
        "Security audit of {component} found {vuln_count} vulnerabilities. "
        "Critical: {critical_count}. {protocol} implemented for authentication. "
        "Penetration test latency: {latency}ms. Audit cost: ${spent}."
    ),
    (
        "{component} Performance Optimization",
        "{component} optimization reduced latency from {old_latency}ms to {latency}ms. "
        "Throughput increased to {throughput} req/sec. "
        "Cache hit rate: {percent}. Memory reduced from {old_memory}GB to {memory}GB."
    ),
    (
        "{component} Deployment & Migration",
        "{component} deployed to production. Migration from {old_tech} to {new_tech} complete. "
        "{node_count} nodes provisioned. Rollback tested in {latency}ms. "
        "Data migrated: {throughput} records. Downtime: {downtime} minutes."
    ),
    (
        "Budget & Timeline Review — {phase}",
        "{phase} review complete. Total spend: ${spent} of ${total} ({percent} utilized). "
        "Remaining: ${remaining}. {person} approved ${approval} for {component}. "
        "Next milestone: {milestone_date}."
    ),
]


class S1Generator(BaseGenerator, DependencyMixin):
    """Generate S1 research literature scenario data."""

    SCENARIO_ID = "s1_research_literature"

    def __init__(self, seed: int = 42, pressure: PressureConfig | None = None):
        super().__init__(seed)
        self.pressure = pressure or PressureConfig.none()

    def generate(self, n_sessions: int = 8) -> dict[str, Any]:
        graph = FactGraph()
        batches = []
        all_probes = []
        all_facts = []
        used_keywords = set()

        components = sample_unique(PROJECT_COMPONENTS, n_sessions, self.rng)

        for cycle in range(n_sessions):
            component = components[cycle % len(components)]
            tmpl_title, tmpl_content = self.rng.choice(_BATCH_TEMPLATES)

            # Generate unique values for this cycle
            vals = self._gen_cycle_values(component, cycle)
            title = tmpl_title.format(**vals)
            content = tmpl_content.format(**vals)

            # Extract 4-6 unique keywords (numbers/names not used before).
            # Only return values that actually appear in the rendered content —
            # different batch templates fill different subsets of `vals`, and
            # the prior implementation extracted unconditionally, producing
            # phantom keywords that could never be recalled (m0=0.5 floor).
            keywords = self._extract_unique_keywords(vals, used_keywords, content=content)
            used_keywords.update(keywords)

            batches.append({
                "cycle": cycle,
                "title": title,
                "content": content,
                "keywords": keywords,
                "n_keywords": len(keywords),
            })

            # Register keywords in the FactGraph
            graph.register_fact(
                session=cycle,
                domain="technical",
                content=f"{title}: {content[:120]}",
                keywords=keywords,
            )

            # Generate probes from keywords
            for j, kw in enumerate(keywords[:3]):
                probe_q = self._keyword_to_question(kw, vals, component)
                all_probes.append({
                    "probe_id": f"s1_c{cycle}_p{j}",
                    "constraint_id": f"kw_{cycle}_{j}",
                    "question": probe_q,
                    "canonical_answer": kw,
                    "keywords": [kw],
                })

            # Session fact
            all_facts.append({
                "session": cycle,
                "id": f"F{cycle}",
                "category": "technical",
                "text": f"In cycle {cycle}, {content[:100]}",
                "recall_question": f"What happened in cycle {cycle} with {component}?",
                "recall_keywords": keywords[:2],
                "recall_anti_keywords": [],
            })

            # Apply dependency task replacement after warmup
            if cycle >= self.pressure.warmup_sessions and self.rng.random() < self.pressure.dependency_density:
                dep_task = self.build_dependency_task(graph, cycle, self.rng, self.pressure)
                if dep_task:
                    meta = dep_task.get("dependency_meta", {})
                    probe_entry = {
                        "probe_id": f"s1_c{cycle}_dep",
                        "constraint_id": f"dep_{cycle}",
                        "question": dep_task["text"],
                        "canonical_answer": dep_task["reference_answer"],
                        "keywords": dep_task["eval_keywords"],
                        "dep_type": meta.get("dep_type"),
                    }
                    # Trend probes test revision-via-DAG: the agent should cite
                    # the CURRENT version (keywords) and NOT the pre-revision
                    # value (common_error). Surfacing both lets the validator
                    # score the probe faithfully instead of falling back to a
                    # session-wide keyword_m proxy.
                    if meta.get("dep_type") == "trend" and meta.get("common_error"):
                        probe_entry["forbidden_keywords"] = [meta["common_error"]]
                    all_probes.append(probe_entry)

            # Apply version updates
            updates = self.version_random_facts(graph, cycle, self.rng, self.pressure)
            if updates:
                for u in updates:
                    batches[-1]["content"] += f"\n{u['text']}"

            # Apply selective forgetting (revision aging)
            invalidations = self.invalidate_random_facts(graph, cycle, self.rng, self.pressure)
            if invalidations:
                for inv in invalidations:
                    batches[-1]["content"] += f"\n{inv['text']}"

            # Inject interference facts (confusable cross-domain pairs)
            if cycle >= self.pressure.confusable_start_session:
                pairs = self.inject_interference(graph, cycle, self.rng, self.pressure)
                for pair in pairs:
                    batches[-1]["content"] += f"\n{pair['text_a']}\n{pair['text_b']}"

        # Cross-cycle queries
        cross_queries = []
        for t in range(3, n_sessions, 3):
            required = [t - 2, t - 1]
            req_kws = []
            for c in required:
                if c < len(batches):
                    req_kws.extend(batches[c]["keywords"][:1])
            cross_queries.append({
                "at_cycle": t,
                "question": f"Compare the results from cycles {required[0]} and {required[1]}.",
                "requires_cycles": required,
                "keywords": req_kws,
            })

        result = {
            "paper_batches": {
                "batches": batches,
                "cross_cycle_queries": cross_queries,
            },
            "probes": all_probes,
            "session_facts": {"facts": all_facts},
            "source_doc": {"text": batches[0]["content"] if batches else ""},
        }
        result["dependency_graph"] = graph.export()
        return result

    def _gen_cycle_values(self, component: str, cycle: int) -> dict:
        return {
            "component": component,
            "percent": random_percent(self.rng, 60, 99),
            "latency": str(random_latency_ms(self.rng, 20, 300)),
            "old_latency": str(random_latency_ms(self.rng, 100, 500)),
            "memory": f"{self.rng.uniform(1.5, 8.0):.1f}",
            "old_memory": f"{self.rng.uniform(3.0, 12.0):.1f}",
            "throughput": f"{random_count(self.rng, 1000, 50000):,}",
            "spent": f"{random_dollar(self.rng, 5000, 200000):,}",
            "total": f"{random_dollar(self.rng, 200000, 900000):,}",
            "remaining": f"{random_dollar(self.rng, 10000, 300000):,}",
            "approval": f"{random_dollar(self.rng, 5000, 100000):,}",
            "phase": self.rng.choice(["Q1", "Q2", "Q3", "Q4", "Phase 1", "Phase 2"]),
            "person": random_person(self.rng),
            "protocol": self.rng.choice(["OAuth 2.0", "mTLS", "JWT RS256", "SAML 2.0"]),
            "vuln_count": str(ensure_non_round(self.rng.randint(2, 20), self.rng)),
            "critical_count": str(self.rng.randint(0, 3)),
            "old_tech": self.rng.choice(TECH_FRAMEWORKS[:10]),
            "new_tech": self.rng.choice(TECH_FRAMEWORKS[10:20]),
            "node_count": str(self.rng.choice([3, 5, 7, 9])),
            "downtime": str(ensure_non_round(self.rng.randint(5, 60), self.rng)),
            "milestone_date": random_date(self.rng),
        }

    def _extract_unique_keywords(self, vals: dict, used: set,
                                  content: str = "") -> list[str]:
        """Extract 4-6 keywords that haven't been used before.

        When `content` is provided, only values that actually appear in the
        rendered template are returned. This prevents phantom keywords (values
        present in `vals` but not used by the chosen template) from inflating
        the denominator of cumulative keyword recall scoring.
        """
        content_lower = content.lower() if content else ""
        def _in_content(val: str) -> bool:
            return (not content) or (val.lower() in content_lower)

        candidates = []
        for key in ["percent", "latency", "memory", "throughput", "spent",
                     "remaining", "vuln_count", "downtime"]:
            val = vals.get(key, "")
            if val and val not in used and _in_content(val):
                candidates.append(val)
        # Also add component name (always rendered into title at minimum,
        # but most templates also mention it in the body).
        comp = vals.get("component", "")
        if comp and comp not in used and _in_content(comp):
            candidates.insert(0, comp)
        return candidates[:6]

    def _keyword_to_question(self, keyword: str, vals: dict, component: str) -> str:
        """Generate a question that targets a specific keyword."""
        if "%" in keyword:
            return f"What was {component}'s test coverage or cache hit rate?"
        if keyword.replace(",", "").isdigit() and int(keyword.replace(",", "")) > 1000:
            return f"What was the throughput or budget figure for {component}?"
        if "." in keyword and not keyword.endswith("%"):
            return f"What was {component}'s memory usage?"
        return f"What specific metric was reported for {component}?"

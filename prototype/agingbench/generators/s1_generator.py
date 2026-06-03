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
    get_project_components,
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

# V2 extension (opt-in via PressureConfig.s1_batch_templates_version=2).
# 5 original templates + 8 new content shapes. NB: V2 templates use
# additional placeholders (p_level, root_cause, mttr, user_count,
# load_factor, experiment_name, metric_name, p_value, decision, n_findings,
# strategy, savings, feature_count, pushed_count, push_date, nps,
# complaint, csat, old_person, n_sessions_xfer) — fetched via
# _gen_cycle_values_v2_extras only when V2 is active, so V1's rng state is
# untouched.
_BATCH_TEMPLATES_V2 = _BATCH_TEMPLATES + [
    (
        "P{p_level} Incident — {component}",
        "P{p_level} incident in {component}: root cause was {root_cause}. "
        "MTTR: {mttr} min. Affected {user_count} users. Postmortem cost: ${spent}."
    ),
    (
        "{component} Capacity Planning Report",
        "{component} capacity analysis: peak concurrent users {throughput}, "
        "P99 latency {latency}ms under {load_factor}x normal load. "
        "Headroom: {percent}. Recommended scale-up: {node_count} nodes."
    ),
    (
        "{component} A/B Test Results — {experiment_name}",
        "{component} A/B test '{experiment_name}' concluded. Variant B improved "
        "{metric_name} by {percent}. Sample size: {throughput}, p-value: {p_value}. "
        "Decision: {decision}."
    ),
    (
        "{component} SOC 2 Audit Report",
        "{component} SOC 2 audit completed by {person}. {n_findings} findings, "
        "{critical_count} critical. Remediation cost: ${spent}. Re-audit: {milestone_date}."
    ),
    (
        "{component} Cost Optimization Summary",
        "{component} cloud spend reduced from ${total}/mo to ${spent}/mo via "
        "{strategy}. Projected savings: ${savings}/yr. Owner: {person}."
    ),
    (
        "{component} Roadmap Revision — {phase}",
        "{component} roadmap revised for {phase}. {feature_count} features pulled "
        "forward to {milestone_date}, {pushed_count} features pushed to {push_date}. "
        "Sign-off: {person}."
    ),
    (
        "{component} Customer Feedback Q-Roll-Up",
        "{component} NPS this quarter: {nps}. Top complaint: {complaint}. "
        "Resolution SLA met for {percent} of {throughput} tickets. CSAT: {csat}."
    ),
    (
        "{component} Ownership Transition",
        "{component} ownership transferred from {old_person} to {person}. "
        "Knowledge transfer sessions: {n_sessions_xfer}. Migration cost: ${spent}. "
        "Effective: {milestone_date}."
    ),
]
assert len(_BATCH_TEMPLATES_V2) == 13, (
    f"_BATCH_TEMPLATES_V2 expected 13 entries; got {len(_BATCH_TEMPLATES_V2)}"
)


def _get_batch_templates(version: int):
    return {1: _BATCH_TEMPLATES, 2: _BATCH_TEMPLATES_V2}[version]


class S1Generator(BaseGenerator, DependencyMixin):
    """Generate S1 research literature scenario data."""

    SCENARIO_ID = "s1_research_literature"

    def __init__(
        self,
        seed: int = 42,
        pressure: PressureConfig | None = None,
        dense_revision: bool = False,
    ):
        super().__init__(seed)
        self.pressure = pressure or PressureConfig.none()
        self.dense_revision = dense_revision

    def generate(self, n_sessions: int = 8) -> dict[str, Any]:
        graph = FactGraph()
        batches = []
        all_probes = []
        all_facts = []
        used_keywords = set()
        # Root fact ids designated for forced multi-depth revision chains.
        # Populated as facts are registered, up to pressure.s1_high_churn_count.
        # See _force_revise_high_churn for the cycle-by-cycle extension pass.
        high_churn_roots: list[str] = []

        components = sample_unique(
            get_project_components(
                getattr(self.pressure, "project_components_pool_version", 1)
            ),
            n_sessions,
            self.rng,
        )

        tmpl_pool = _get_batch_templates(
            getattr(self.pressure, "s1_batch_templates_version", 1)
        )
        for cycle in range(n_sessions):
            component = components[cycle % len(components)]
            tmpl_title, tmpl_content = self.rng.choice(tmpl_pool)

            # Generate unique values for this cycle. Always call the V1 path
            # first (preserves rng ordering for legacy yamls), then top up
            # with V2 extras only when V2 templates are enabled. Without the
            # gating, V1 yamls would consume extra rng entropy for unused
            # placeholders and shift downstream output.
            vals = self._gen_cycle_values(component, cycle)
            if getattr(self.pressure, "s1_batch_templates_version", 1) >= 2:
                vals.update(self._gen_cycle_values_v2_extras(component, cycle))
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
                "component": component,  # explicit anchor for rich probes
            })

            # Register keywords in the FactGraph. Use the FULL content (not
            # content[:120]) so that version_random_facts' keyword replacement
            # can find values located past char 120 — the previous truncation
            # silently broke revision rendering for any keyword whose position
            # in the body exceeded the cap, leaving the new value invisible
            # in the rendered update text and making revision-aware probes
            # promote to a gold value that doesn't appear anywhere.
            registered = graph.register_fact(
                session=cycle,
                domain="technical",
                content=f"{title}: {content}",
                keywords=keywords,
            )
            # Designate the first K facts as high-churn roots — they get
            # force-revised every subsequent cycle to grow guaranteed
            # depth chains. Only requires fact has at least one numeric
            # keyword (otherwise revision mutation is a no-op anyway).
            target = int(getattr(self.pressure, "s1_high_churn_count", 0) or 0)
            if (
                target > 0
                and len(high_churn_roots) < target
                and any(self._numeric_token_or_none(k) is not None for k in keywords)
            ):
                high_churn_roots.append(registered.id)

            # Generate probes from keywords. `ask_at_cycle` tags the earliest
            # cycle when this probe's source content is available — so the
            # runner can filter out probes about future facts at score time,
            # making response-based scoring symmetric with memory-based
            # (which only counts cohort_keywords[c] for c <= current cycle).
            for j, kw in enumerate(keywords[:3]):
                if getattr(self.pressure, "s1_rich_probes_enabled", False):
                    probe_q = self._keyword_to_question_v2(kw, vals, component)
                else:
                    probe_q = self._keyword_to_question(kw, vals, component)
                all_probes.append({
                    "probe_id": f"s1_c{cycle}_p{j}",
                    "constraint_id": f"kw_{cycle}_{j}",
                    "question": probe_q,
                    "canonical_answer": kw,
                    "keywords": [kw],
                    "ask_at_cycle": cycle,
                })

            # Session fact
            all_facts.append({
                "session": cycle,
                "id": f"F{cycle}",
                "category": "technical",
                "text": f"In cycle {cycle}, {content[:100]}",
                "recall_question": f"What happened in cycle {cycle} with {component}?",
                "recall_keywords": keywords[:2],
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
                        "ask_at_cycle": cycle,
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
            # Force-revise high-churn roots to grow multi-depth chains.
            # Append to the same updates list so the dense_revision probe
            # loop below treats them identically. Skip roots already
            # revised stochastically this cycle.
            force_updates = self._force_revise_high_churn(
                graph, cycle, high_churn_roots, updates
            )
            updates.extend(force_updates)
            if updates:
                for u in updates:
                    batches[-1]["content"] += f"\n{u['text']}"

            # Dense revision: emit one trend probe per fact updated this cycle,
            # so version_accuracy gets dense per-session coverage instead of
            # depending on the sparse dependency_density-gated build_dependency_task path.
            if self.dense_revision and updates:
                for u in updates:
                    new_kws = list(u.get("new_keywords") or [])
                    old_kws = list(u.get("old_keywords") or [])
                    # The discriminating tokens — keywords unique to one version.
                    # Entity names ("Search Engine") survive the revision unchanged
                    # and would match in any memory state if used; the changed
                    # values are what actually distinguish the two versions.
                    novel_only = [k for k in new_kws if k not in old_kws]
                    stale_only = [k for k in old_kws if k not in new_kws]
                    if not novel_only:
                        # Defensive: no actual value change → probe is unscorable.
                        continue
                    anchor = old_kws[0] if old_kws else "the value"
                    task_id = f"s1_c{cycle}_rev_{u['new_fact_id']}"
                    # Register a dependency edge so the probe surfaces in
                    # dependency_graph.tasks. Without this, version_accuracy /
                    # chain_recall_by_version_depth / per-session variants
                    # iterate over tasks and silently skip the dense probes
                    # (the runtime self.probes path scores them into kw_m but
                    # the metric layer reads from graph.tasks).
                    graph.add_dependency(
                        task_id=task_id,
                        session=cycle,
                        fact_ids=[u["old_fact_id"], u["new_fact_id"]],
                        dep_type="trend",
                    )
                    all_probes.append({
                        "probe_id": task_id,
                        "constraint_id": u["new_fact_id"],
                        "question": (
                            f"You previously recorded a value associated with {anchor!r}. "
                            f"It has been revised. What is the CURRENT value? "
                            f"Reply with the updated value, not the original."
                        ),
                        "canonical_answer": " / ".join(novel_only),
                        "keywords": novel_only,
                        "forbidden_keywords": stale_only,
                        "dep_type": "trend",
                        "ask_at_cycle": cycle,
                    })

            # Apply selective forgetting (revision aging)
            invalidations = self.invalidate_random_facts(graph, cycle, self.rng, self.pressure)
            if invalidations:
                for inv in invalidations:
                    batches[-1]["content"] += f"\n{inv['text']}"

            # NB: S1 does NOT inject interference. The cross-domain
            # CONFUSABLE_TERMS distractors broke the "research literature"
            # framing (bolting "budget for marketing $X" onto a research-
            # memo batch made no sense), and the corresponding
            # interference_resistance / score_interference_binding metrics
            # were essentially vacuous on transformer LLMs at the default
            # configs (the K=100 fan-effect experiments confirmed this).
            # PressureConfig.n_confusable_pairs / confusable_start_session
            # are still honored by other scenarios (S2-S6).

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

        # ─── Step C: rich probe types (compare / inverse) ───────────────
        # Gated on s1_rich_probes_enabled. Uses ONLY deterministic ops over
        # `batches` (no rng calls), so adding these probes does not perturb
        # any other generation seed state.
        if getattr(self.pressure, "s1_rich_probes_enabled", False):
            all_probes.extend(self._build_rich_probes(batches))

        # ─── Step D: forbidden_keywords retroactive pass ────────────────
        # Walk graph for revised facts; if an earlier recall probe targeted
        # the OLD value of a now-revised fact, attach forbidden_keywords
        # so revision scorers can penalize stale recall directly.
        if getattr(self.pressure, "s1_forbidden_keywords_on_recall", False):
            self._attach_forbidden_keywords_retroactively(all_probes, graph)

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

    def _build_rich_probes(self, batches: list[dict]) -> list[dict]:
        """Build COMPARE + INVERSE probes from accumulated batches.

        Uses only deterministic operations on existing batch content —
        no rng calls, so adding these probes does not perturb other
        generation that uses self.rng. Each probe is positioned at a
        cycle index (asked at that cycle and onward by the runner).
        """
        rich: list[dict] = []

        # COMPARE: at each cycle t ≥ 2, ask "of components in cycles 0..t-1,
        # which had the highest numeric value?" Gold = the component name
        # whose first numeric keyword is max in the prior window. We use ANY
        # numeric keyword (not just percent) so the probe fires even when
        # templates without a % placeholder were picked — otherwise compare
        # probes never emit in template-mixed runs.
        def _first_numeric(kws):
            for k in kws:
                s = k.replace(",", "").rstrip("%").rstrip("$")
                try:
                    return float(s)
                except ValueError:
                    continue
            return None

        for t in range(2, len(batches)):
            window = batches[:t]
            cand = []
            for b in window:
                v = _first_numeric(b["keywords"])
                if v is not None and b.get("component"):
                    cand.append((v, b["component"], b["cycle"]))
            if len(cand) < 2:
                continue
            v_max, comp_max, c_max = max(cand, key=lambda x: x[0])
            rich.append({
                "probe_id": f"s1_t{t}_compare",
                "constraint_id": f"compare_{t}",
                "question": (
                    f"Of the components reviewed in cycles 0 through {t-1}, "
                    f"which had the highest primary metric reported?"
                ),
                "canonical_answer": comp_max,
                "keywords": [comp_max],
                "probe_type": "compare",
                "ask_at_cycle": t,
            })

        # INVERSE: at each cycle t ≥ 2, pick a memorable numeric value from a
        # prior batch and ask "which component had value X?" Gold = that
        # component name.
        for t in range(2, len(batches)):
            window = batches[:t]
            for b in window:
                if not b.get("component"):
                    continue
                # Prefer comma-separated 5+ digit numbers (high recognizability)
                target = next(
                    (k for k in b["keywords"]
                     if "," in k and len(k.replace(",", "")) >= 4),
                    None,
                )
                if not target:
                    continue
                rich.append({
                    "probe_id": f"s1_t{t}_inverse_c{b['cycle']}",
                    "constraint_id": f"inverse_{t}_{b['cycle']}",
                    "question": (
                        f"Which component had a value of {target} reported "
                        f"in one of the earlier cycles?"
                    ),
                    "canonical_answer": b["component"],
                    "keywords": [b["component"]],
                    "probe_type": "inverse",
                    "ask_at_cycle": t,
                })
                break  # at most one inverse probe per (t, source-cycle)

        return rich

    @staticmethod
    def _numeric_token_or_none(kw: str) -> int | None:
        """Return the integer value if kw is a mutatable numeric token,
        else None. Strips $/,/% so '$1,234' and '1234' both parse."""
        try:
            return int(kw.replace(",", "").replace("$", "").replace("%", ""))
        except (ValueError, TypeError):
            return None

    def _force_revise_high_churn(
        self,
        graph,
        cycle: int,
        root_ids: list[str],
        existing_updates: list[dict],
    ) -> list[dict]:
        """Grow multi-depth revision chains on designated high-churn roots.

        FactGraph.get_updatable_facts caps natural revisions at depth=2
        (its ``version == 1`` filter), so without this pass the
        chain_recall_by_version_depth metric only ever sees depth-2
        buckets. This helper walks each designated root to its current
        head, mutates its numeric keywords with the same delta logic
        version_random_facts uses, and emits an update dict in the same
        shape — so the caller's dense_revision probe loop emits depth-N
        revision probes naturally.

        Skip rules:
          - Root already revised stochastically this cycle (avoid
            double-mutation, which would lose one version's discriminating
            tokens to set-difference cancellation).
          - Head fact was introduced this cycle (can't revise within the
            same session it was registered).
          - Mutation produces no actual value change.
          - Mutation produces an unscorable short numeric (mirrors the
            ``_short_num`` guard in dependency_mixin.version_random_facts).
        """
        if not root_ids:
            return []
        # Roots already revised stochastically this cycle — walk each
        # update's old_fact_id back to its root.
        already: set[str] = set()
        for u in existing_updates:
            f = graph.facts.get(u["old_fact_id"])
            while f is not None and f.replaces is not None:
                f = graph.facts.get(f.replaces)
            if f is not None:
                already.add(f.id)

        def _short_num(v: str) -> bool:
            s = v.replace(",", "")
            return s.isdigit() and len(s) < 3

        extra: list[dict] = []
        for root_id in root_ids:
            if root_id in already:
                continue
            head = graph.get_current_version(root_id)
            if head.session >= cycle:
                continue  # can't revise in same cycle as introduction
            # Mutate head's numeric keywords (mirrors version_random_facts).
            new_keywords: list[str] = []
            _cache: dict[int, int] = {}
            for kw in head.keywords:
                val = self._numeric_token_or_none(kw)
                if val is None:
                    new_keywords.append(kw)
                    continue
                if val in _cache:
                    new_val = _cache[val]
                else:
                    delta = self.rng.randint(-val // 4, val // 4) or self.rng.choice([-1, 1])
                    new_val = val + delta
                    _cache[val] = new_val
                if "$" in kw:
                    new_kw = f"${new_val:,}" if new_val >= 1000 else f"${new_val}"
                elif "%" in kw:
                    new_kw = f"{new_val}%"
                elif "," in kw:
                    new_kw = f"{new_val:,}"
                else:
                    new_kw = str(new_val)
                new_keywords.append(new_kw)
            if new_keywords == head.keywords:
                continue
            if any(_short_num(k) for k in new_keywords):
                continue
            new_content = head.content
            for old_kw, new_kw in zip(head.keywords, new_keywords):
                new_content = new_content.replace(old_kw, new_kw)
            new_fact = graph.update_fact(
                old_id=head.id,
                new_content=new_content,
                new_keywords=new_keywords,
                session=cycle,
            )
            extra.append({
                "old_fact_id": head.id,
                "old_keywords": list(head.keywords),
                "new_fact_id": new_fact.id,
                "new_keywords": new_keywords,
                "text": f"UPDATE: {new_content} (revised from earlier analysis)",
            })
        return extra

    def _attach_forbidden_keywords_retroactively(
        self, probes: list[dict], graph
    ) -> None:
        """For each revised fact, promote any probe whose `keywords` are
        the OLD value to: (1) target the NEW value as gold, and (2) carry
        the OLD value in `forbidden_keywords`. Mirrors S6's
        _sync_probes_after_revisions: the basic per-keyword recall probes
        are emitted at cycle K against the value-at-cycle-K; if the fact
        is later revised at cycle K+n, the probe's gold becomes stale and
        the agent should cite the NEW value, not the original. Without
        this sync, recall_rate penalizes correct revision."""
        for f in graph.facts.values():
            if f.replaced_by is None:
                continue  # not revised
            new_f = graph.facts[f.replaced_by]
            stale_kws = set(f.keywords) - set(new_f.keywords)
            new_kws = set(new_f.keywords) - set(f.keywords)
            if not stale_kws or not new_kws:
                continue
            stale_sorted = sorted(stale_kws)
            new_sorted = sorted(new_kws)
            for p in probes:
                if p.get("forbidden_keywords"):
                    continue  # already revision-aware (trend dep probes)
                probe_kws = p.get("keywords") or []
                # Probe targets a stale value → flip it to the new value.
                if any(kw in stale_kws for kw in probe_kws):
                    p["keywords"] = new_sorted
                    if "canonical_answer" in p:
                        p["canonical_answer"] = " / ".join(new_sorted)
                    p["forbidden_keywords"] = stale_sorted
                    p["revision_source_fact"] = f.id
                    p["revision_target_fact"] = new_f.id

    def _gen_cycle_values_v2_extras(self, component: str, cycle: int) -> dict:
        """Extra placeholders needed by V2 batch templates only.

        Called only when ``pressure.s1_batch_templates_version >= 2`` so that
        V1 yamls do not consume the extra rng entropy this would otherwise
        introduce. Each call consumes ~20 fresh rng draws after the V1 pool;
        same seed + V2 produces identical V2 output across runs.
        """
        return {
            "p_level": str(self.rng.choice([0, 1, 2])),
            "root_cause": self.rng.choice([
                "connection pool exhaustion", "memory leak in worker",
                "schema migration drift", "stale cache invalidation",
                "third-party API outage", "DNS misconfiguration",
                "race condition on retry path", "log volume blew past quota",
            ]),
            "mttr": str(self.rng.randint(5, 240)),
            "user_count": f"{random_count(self.rng, 100, 50_000):,}",
            "load_factor": str(self.rng.choice([2, 3, 5, 10])),
            "experiment_name": self.rng.choice([
                "Atlas", "Beacon", "Cobalt", "Delta", "Echo", "Falcon",
                "Granite", "Helix", "Iris", "Juniper",
            ]),
            "metric_name": self.rng.choice([
                "conversion rate", "click-through", "session length",
                "retention D7", "checkout success", "page-load p95",
            ]),
            "p_value": f"0.{self.rng.randint(1, 49):03d}",
            "decision": self.rng.choice([
                "ship variant B", "extend the test", "ship variant A",
                "abandon both",
            ]),
            "n_findings": str(self.rng.randint(2, 18)),
            "strategy": self.rng.choice([
                "reserved-instance migration", "spot-fleet adoption",
                "right-sizing initiative", "egress traffic dedup",
                "log-retention policy cut", "image-build cache reuse",
            ]),
            "savings": f"{random_dollar(self.rng, 20_000, 500_000):,}",
            "feature_count": str(self.rng.randint(1, 6)),
            "pushed_count": str(self.rng.randint(1, 4)),
            "push_date": random_date(self.rng),
            "nps": str(self.rng.randint(15, 78)),
            "complaint": self.rng.choice([
                "slow checkout flow", "missing export button",
                "confusing tier comparison", "weak mobile experience",
                "intermittent 5xx on submit",
            ]),
            "csat": f"{self.rng.randint(70, 96)}%",
            "old_person": random_person(self.rng),
            "n_sessions_xfer": str(self.rng.randint(2, 8)),
        }

    def _extract_unique_keywords(self, vals: dict, used: set,
                                  content: str = "") -> list[str]:
        """Extract 4-6 keywords that haven't been used before.

        When `content` is provided, only values that actually appear in the
        rendered template are returned. This prevents phantom keywords (values
        present in `vals` but not used by the chosen template) from inflating
        the denominator of cumulative keyword recall scoring.

        Purely-numeric values shorter than 3 characters (e.g. '2', '14') are
        rejected at emission time. The scoring layer uses a digit-flank guard
        ('2' must NOT match inside '20'), so single-digit numerics would
        always fail the survival check even when the underlying fact IS in
        memory — they'd push keyword_m below 1.0 at cycle 0 by construction.
        Mirrors the trident's _MIN_DISCRIMINATING_LEN filter.
        """
        content_lower = content.lower() if content else ""
        def _in_content(val: str) -> bool:
            return (not content) or (val.lower() in content_lower)

        def _emittable(val: str) -> bool:
            """A purely-numeric value must be at least 3 chars to be a useful
            discriminator under digit-flank-safe matching."""
            if not val:
                return False
            # Strip thousands separators so '1,847' counts as 4 digits.
            v = val.replace(",", "")
            if v.isdigit() and len(v) < 3:
                return False
            return True

        candidates = []
        for key in ["percent", "latency", "memory", "throughput", "spent",
                     "remaining", "vuln_count", "downtime"]:
            val = vals.get(key, "")
            if val and val not in used and _emittable(val) and _in_content(val):
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

    # ─── V2 probe-question templates (12 role-based shapes) ─────────────
    # Maps each vals key to one or more question phrasings; matches the
    # keyword's role (not just its surface shape) so the probe is specific
    # to what the agent actually saw. Gated on s1_rich_probes_enabled.
    _ROLE_QUESTIONS: dict[str, list[str]] = {
        "percent": [
            "What rate or percentage was reported for {component}?",
            "What was {component}'s test coverage, cache hit rate, or "
            "utilization figure?",
        ],
        "latency": [
            "What was {component}'s latency in milliseconds?",
            "How many ms did {component} take to respond at p50?",
        ],
        "old_latency": [
            "What was {component}'s ORIGINAL (pre-optimization) latency in ms?",
        ],
        "memory": [
            "How many GB of memory did {component} use?",
        ],
        "old_memory": [
            "What was {component}'s ORIGINAL (pre-optimization) memory footprint in GB?",
        ],
        "throughput": [
            "How many requests, records, or items did {component} process?",
            "What throughput figure was reported for {component}?",
        ],
        "spent": [
            "How much was spent on {component}?",
            "What dollar figure was attributed to {component}'s spending?",
        ],
        "total": [
            "What was the TOTAL budget or allocation reported for {component}?",
        ],
        "remaining": [
            "How much was REMAINING in the budget for {component}?",
        ],
        "approval": [
            "What dollar amount was APPROVED for {component}?",
        ],
        "vuln_count": [
            "How many vulnerabilities were found in {component}'s security audit?",
        ],
        "critical_count": [
            "How many CRITICAL findings did {component}'s audit report?",
        ],
        "downtime": [
            "How many minutes of downtime did {component} experience during "
            "its deployment?",
        ],
        "node_count": [
            "How many nodes were provisioned for {component}?",
        ],
        # ─── V2-batch-templates extras ───
        "mttr": [
            "What was the MTTR (in minutes) for the {component} incident?",
        ],
        "user_count": [
            "How many users were affected by the {component} incident?",
        ],
        "load_factor": [
            "Under what load factor was {component}'s capacity tested?",
        ],
        "n_findings": [
            "How many findings did {component}'s SOC 2 audit produce?",
        ],
        "savings": [
            "What annual savings did {component}'s cost optimization yield?",
        ],
        "nps": [
            "What was {component}'s NPS score this quarter?",
        ],
        "csat": [
            "What was {component}'s CSAT score this quarter?",
        ],
        "feature_count": [
            "How many features were pulled forward for {component}?",
        ],
    }

    def _keyword_to_question_v2(self, keyword: str, vals: dict, component: str) -> str:
        """Role-based probe-question dispatcher.

        Reverse-maps ``keyword`` to its role in the ``vals`` dict (e.g.,
        '299' might be the value of vals['latency']). Then selects a
        role-specific question template — much more diverse and specific
        than V1's 4-shape branching. Falls back to V1's heuristics for
        unrecognized keys.
        """
        # Reverse lookup: which val key produced this keyword?
        # Strip $ commas % for comparison.
        def _norm(s: str) -> str:
            return s.strip().lstrip("$").rstrip("%").replace(",", "")

        kw_norm = _norm(keyword)
        for role, val in vals.items():
            if val is None:
                continue
            if _norm(str(val)) == kw_norm and role in self._ROLE_QUESTIONS:
                phrasings = self._ROLE_QUESTIONS[role]
                # Deterministic pick (by hash of keyword) so the same
                # keyword always maps to the same phrasing — keeps the
                # probe stream stable across runs.
                idx = abs(hash(keyword)) % len(phrasings)
                return phrasings[idx].format(component=component)

        # Fallback: legacy heuristics from V1.
        return self._keyword_to_question(keyword, vals, component)

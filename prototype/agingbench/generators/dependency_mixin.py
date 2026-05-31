"""
agingbench/generators/dependency_mixin.py — Methods for building dependency-aware tasks.

Provides reusable methods that any generator can call during its generate() loop
to create tasks that genuinely depend on prior session facts, version-tracked
updates, and interference entities.

Usage in a generator:
    from .dependency_mixin import DependencyMixin

    class S6Generator(BaseGenerator, DependencyMixin):
        def generate(self, n_sessions):
            graph = FactGraph()
            for t in range(n_sessions):
                if t >= self.pressure.warmup_sessions:
                    if self.rng.random() < self.pressure.dependency_density:
                        task = self.build_dependency_task(graph, t, self.rng)
                        ...
"""

from __future__ import annotations

import random
from typing import Optional

from .fact_graph import FactGraph, Fact
from .pressure_config import PressureConfig


# ------------------------------------------------------------------ templates

COMPARE_TEMPLATES = [
    "Compare the {metric_a} of {entity_a} (from your earlier {domain_a} research) "
    "with the {metric_b} of {entity_b} (from your {domain_b} analysis). "
    "Which is higher and by how much?",

    "In your previous research, you found that {entity_a} had {metric_a} = {value_a}, "
    "and {entity_b} had {metric_b} = {value_b}. "
    "How do these compare? What does this tell us?",

    "Review your findings about {entity_a} ({domain_a}) and {entity_b} ({domain_b}). "
    "What were their respective {metric_a} values, and which performed better?",
]

TREND_TEMPLATES = [
    "When we first analyzed {entity}, the {metric} was {old_value}. "
    "Based on the latest update, what is the current value? Has it improved?",

    "Our records show that {entity}'s {metric} was originally {old_value}. "
    "This was later revised. What is the updated figure?",

    "Check your records for {entity}. The {metric} has been updated since "
    "our initial analysis. What is the most recent value?",
]

SYNTHESIZE_TEMPLATES = [
    "Based on your research across {domain_a}, {domain_b}, and {domain_c}, "
    "summarize the key findings. Cite specific values from each domain.",

    "Write an executive summary combining your analysis of {entity_a} ({domain_a}), "
    "{entity_b} ({domain_b}), and {entity_c} ({domain_c}). "
    "Include the specific values you found.",

    "Review all your prior research findings. For {domain_a}, {domain_b}, and {domain_c}: "
    "what were the most important values you discovered? List them.",
]


class DependencyMixin:
    """
    Mixin providing dependency-task generation methods.

    Methods take FactGraph and PressureConfig as arguments (not stored as
    instance state) so they compose cleanly with any generator.
    """

    def build_dependency_task(
        self,
        graph: FactGraph,
        session: int,
        rng: random.Random,
        pressure: Optional[PressureConfig] = None,
    ) -> Optional[dict]:
        """
        Build a task that depends on prior session facts.

        Returns a dict with:
          - text: the task prompt
          - eval_keywords: keywords the answer must contain
          - dependency_meta: {dep_type, depends_on, chain_depth}
          - reference_answer: the correct answer

        Returns None if not enough facts exist for the chosen dependency type.
        """
        if pressure is None:
            pressure = PressureConfig()

        available = graph.get_facts_before(session)
        if len(available) < 2:
            return None

        # Choose dependency type
        versioned = [f for f in available if f.replaced_by is not None]
        dep_types = ["compare", "synthesize"]
        if versioned:
            dep_types.append("trend")
        if len(available) >= 3:
            dep_types.append("synthesize")

        # Prefer "trend" when versioned facts exist. version_accuracy ONLY
        # counts trend tasks, so leaving trend as one of 3-4 equiprobable picks
        # meant whole runs often emitted zero trend tasks (S1/S3) and
        # version_accuracy had no coverage. Biasing here gives revision runs a
        # real version-test signal; when update_rate=0 there are no versioned
        # facts and behavior is unchanged.
        if versioned and rng.random() < 0.6:
            dep_type = "trend"
        else:
            dep_type = rng.choice(dep_types)

        if dep_type == "compare":
            return self._build_compare(graph, session, available, rng)
        elif dep_type == "trend":
            if versioned:
                return self._build_trend(graph, session, versioned, rng)
            return self._build_compare(graph, session, available, rng)
        elif dep_type == "synthesize":
            return self._build_synthesize(graph, session, available, rng, pressure)
        return None

    def _build_compare(
        self,
        graph: FactGraph,
        session: int,
        available: list[Fact],
        rng: random.Random,
    ) -> dict:
        """Build a compare task between two facts from different sessions."""
        # Pick 2 facts from different sessions
        rng.shuffle(available)
        fact_a = available[0]
        candidates_b = [f for f in available if f.session != fact_a.session]
        if not candidates_b:
            candidates_b = [f for f in available if f.id != fact_a.id]
        fact_b = rng.choice(candidates_b) if candidates_b else available[1]

        template = rng.choice(COMPARE_TEMPLATES)
        text = template.format(
            entity_a=fact_a.content.split(".")[0][:50],
            entity_b=fact_b.content.split(".")[0][:50],
            domain_a=fact_a.domain,
            domain_b=fact_b.domain,
            metric_a="value",
            metric_b="value",
            value_a=fact_a.keywords[0] if fact_a.keywords else "N/A",
            value_b=fact_b.keywords[0] if fact_b.keywords else "N/A",
        )

        # Both facts' keywords must appear in answer
        eval_keywords = []
        for kw in fact_a.keywords[:2]:
            eval_keywords.append(kw)
        for kw in fact_b.keywords[:2]:
            eval_keywords.append(kw)

        dep_facts = [fact_a.id, fact_b.id]
        edge = graph.add_dependency(
            task_id=f"dep_s{session}_compare",
            session=session,
            fact_ids=dep_facts,
            dep_type="compare",
        )

        return {
            "text": text,
            "eval_keywords": eval_keywords,
            "reference_answer": f"{fact_a.content} vs {fact_b.content}",
            "dependency_meta": {
                "dep_type": "compare",
                "depends_on": dep_facts,
                "depends_on_sessions": sorted(set([fact_a.session, fact_b.session])),
                "chain_depth": edge.chain_depth,
            },
        }

    def _build_trend(
        self,
        graph: FactGraph,
        session: int,
        versioned: list[Fact],
        rng: random.Random,
    ) -> dict:
        """Build a trend task for a versioned fact."""
        old_fact = rng.choice(versioned)
        current = graph.get_current_version(old_fact.id)

        # The stale value the agent must NOT cite is the keyword from the
        # OLD version that is no longer present in the current version's
        # keyword set. version_random_facts preserves non-numeric keywords
        # (e.g. component names) verbatim and only mutates numeric ones, so
        # this set difference cleanly isolates the values that changed.
        # Using old_fact.keywords[0] (the first entry, often a component
        # name shared by both versions) produced a false-positive forbidden
        # keyword that overlapped with eval_keywords and made the probe
        # unscoreable.
        new_kw_set = set(current.keywords)
        stale_values = [kw for kw in old_fact.keywords if kw not in new_kw_set]
        common_error = stale_values[0] if stale_values else None

        template = rng.choice(TREND_TEMPLATES)
        text = template.format(
            entity=old_fact.content.split(".")[0][:50],
            metric="value",
            old_value=common_error if common_error else (
                old_fact.keywords[0] if old_fact.keywords else "N/A"
            ),
        )

        # Must cite the CURRENT value, not the original
        eval_keywords = list(current.keywords[:2])

        dep_facts = [old_fact.id, current.id]
        edge = graph.add_dependency(
            task_id=f"dep_s{session}_trend",
            session=session,
            fact_ids=dep_facts,
            dep_type="trend",
        )

        return {
            "text": text,
            "eval_keywords": eval_keywords,
            "reference_answer": f"Updated from {old_fact.keywords} to {current.keywords}",
            "dependency_meta": {
                "dep_type": "trend",
                "depends_on": dep_facts,
                "depends_on_sessions": sorted(set([old_fact.session, current.session])),
                "chain_depth": edge.chain_depth,
                "common_error": common_error,
            },
        }

    def _build_synthesize(
        self,
        graph: FactGraph,
        session: int,
        available: list[Fact],
        rng: random.Random,
        pressure: PressureConfig,
    ) -> dict:
        """Build a synthesis task spanning 3+ facts from different sessions."""
        n_facts = min(len(available), rng.randint(3, min(5, pressure.max_chain_depth + 1)))

        # Pick facts from different sessions
        rng.shuffle(available)
        selected = []
        seen_sessions = set()
        for f in available:
            if f.session not in seen_sessions:
                selected.append(f)
                seen_sessions.add(f.session)
            if len(selected) >= n_facts:
                break
        if len(selected) < 3:
            selected = available[:n_facts]

        # Build domains list
        domains = list(set(f.domain for f in selected))
        while len(domains) < 3:
            domains.append(domains[-1] if domains else "general")

        template = rng.choice(SYNTHESIZE_TEMPLATES)
        # Redact each entity's value tokens (its keywords) so the template
        # doesn't embed the gold answer the eval_keywords subsequently scores
        # against. Pre-fix: the entity strings were ``f.content.split(".")[0][:30]``
        # which for S6 fact content like "Revenue: $785,163" leaked the exact
        # value into the synthesis prompt — same copy-echo failure mode as
        # the xref task. Now: "Revenue: $[?]".
        def _redact_entity(f: Fact) -> str:
            raw = f.content.split(".")[0][:60]   # widen window so labels survive
            for kw in sorted(f.keywords or [], key=len, reverse=True):
                if kw:
                    raw = raw.replace(kw, "[?]")
            return raw[:50]   # keep prompt-token budget bounded
        entities = [_redact_entity(f) for f in selected]
        text = template.format(
            domain_a=domains[0],
            domain_b=domains[1],
            domain_c=domains[2] if len(domains) > 2 else domains[0],
            entity_a=entities[0] if entities else "item A",
            entity_b=entities[1] if len(entities) > 1 else "item B",
            entity_c=entities[2] if len(entities) > 2 else "item C",
        )

        # Must cite at least one keyword from each selected fact
        eval_keywords = []
        for f in selected:
            if f.keywords:
                eval_keywords.append(f.keywords[0])

        dep_facts = [f.id for f in selected]
        edge = graph.add_dependency(
            task_id=f"dep_s{session}_synthesize",
            session=session,
            fact_ids=dep_facts,
            dep_type="synthesize",
        )

        return {
            "text": text,
            "eval_keywords": eval_keywords,
            "reference_answer": "; ".join(f.content[:80] for f in selected),
            "dependency_meta": {
                "dep_type": "synthesize",
                "depends_on": dep_facts,
                "depends_on_sessions": sorted(set(f.session for f in selected)),
                "chain_depth": edge.chain_depth,
            },
        }

    # ------------------------------------------------------------------ versioning

    def version_random_facts(
        self,
        graph: FactGraph,
        session: int,
        rng: random.Random,
        pressure: PressureConfig,
    ) -> list[dict]:
        """
        Update a fraction of existing facts with new values.

        Returns list of update dicts {old_fact_id, old_value, new_fact_id, new_value, text}
        that can be embedded in the session's environment data.
        """
        if pressure.update_rate <= 0:
            return []

        updatable = graph.get_updatable_facts(before_session=session)
        if not updatable:
            return []

        n_to_update = max(1, int(len(updatable) * pressure.update_rate))
        to_update = rng.sample(updatable, min(n_to_update, len(updatable)))

        updates = []
        for old_fact in to_update:
            # Generate new value (modify existing keywords)
            new_keywords = []
            # Cache new value per underlying number so different FORMATS of the
            # same value (e.g. "429,374" and "429374") map to the SAME new
            # number. Previously each was mutated independently, producing
            # inconsistent v2 keywords (e.g. 473,858 vs 514899 — the latter a
            # phantom that appears in no content).
            _val_cache: dict = {}
            for kw in old_fact.keywords:
                # Try to modify numerical values
                try:
                    val = int(kw.replace(",", "").replace("$", "").replace("%", ""))
                    if val in _val_cache:
                        new_val = _val_cache[val]
                    else:
                        delta = rng.randint(-val // 4, val // 4) or rng.choice([-1, 1])
                        new_val = val + delta
                        _val_cache[val] = new_val
                    # Preserve formatting
                    if "$" in kw:
                        new_kw = f"${new_val:,}" if new_val >= 1000 else f"${new_val}"
                    elif "%" in kw:
                        new_kw = f"{new_val}%"
                    elif "," in kw:
                        new_kw = f"{new_val:,}"
                    else:
                        new_kw = str(new_val)
                    new_keywords.append(new_kw)
                except (ValueError, TypeError):
                    new_keywords.append(kw)  # keep non-numeric keywords

            if new_keywords == old_fact.keywords:
                continue  # no change

            new_content = old_fact.content
            for old_kw, new_kw in zip(old_fact.keywords, new_keywords):
                new_content = new_content.replace(old_kw, new_kw)

            new_fact = graph.update_fact(
                old_id=old_fact.id,
                new_content=new_content,
                new_keywords=new_keywords,
                session=session,
            )

            updates.append({
                "old_fact_id": old_fact.id,
                "old_keywords": old_fact.keywords,
                "new_fact_id": new_fact.id,
                "new_keywords": new_keywords,
                "text": f"UPDATE: {new_content} (revised from earlier analysis)",
            })

        return updates

    # ------------------------------------------------------------------ interference

    def inject_interference(
        self,
        graph: FactGraph,
        session: int,
        rng: random.Random,
        pressure: PressureConfig,
        confusable_terms: Optional[dict] = None,
    ) -> list[dict]:
        """
        Create confusable entities across domains.

        Uses CONFUSABLE_TERMS from pools.py if available, otherwise generates
        simple confusable pairs.

        Returns list of interference fact dicts that should be embedded in env data.
        """
        if confusable_terms is None:
            try:
                from .pools import CONFUSABLE_TERMS
                confusable_terms = CONFUSABLE_TERMS
            except ImportError:
                confusable_terms = {}

        existing_groups = {p["shared_term"] for p in graph.interference_pairs}
        n_existing = len(existing_groups)
        n_needed = pressure.n_confusable_pairs - n_existing

        if n_needed <= 0:
            return []

        results = []

        # Similar-NAME mode (the "two Johns" case): near-identical names with
        # DISTINCT attribute values; ambiguity is in the retrieval key.
        if getattr(pressure, "confusable_similar_names", False):
            # Pool sized (~15) to match the value-type CONFUSABLE_TERMS pool so
            # name confusables aren't under-supplied relative to other types.
            name_pairs = [("John Smith", "John Smyth"), ("Sara Chen", "Sarah Chen"),
                          ("Michael Brown", "Micheal Browne"), ("David Lee", "David Li"),
                          ("Catherine Park", "Katherine Park"), ("Eric Olson", "Erik Olsen"),
                          ("Ana Reyes", "Anna Reyes"), ("Mohamed Ali", "Mohammed Ali"),
                          ("Jon Stewart", "John Stewart"), ("Kristen Lowe", "Kristin Lowe"),
                          ("Geoffrey Hall", "Jeffrey Hall"), ("Sean Murphy", "Shawn Murphy"),
                          ("Bryan Cole", "Brian Cole"), ("Philip Ross", "Phillip Ross"),
                          ("Carolyn Diaz", "Caroline Diaz")]
            attrs = ["extension", "desk number", "employee ID"]
            avail = [p for p in name_pairs if p[0] not in existing_groups]
            for (n1, n2) in avail[:n_needed]:
                attr = rng.choice(attrs)
                v1 = rng.randint(1000, 9999)
                v2 = rng.randint(1000, 9999)
                while v2 == v1:
                    v2 = rng.randint(1000, 9999)
                f1 = graph.register_fact(session=session, domain=n1,
                    content=f"{n1} is in the office directory; {attr}: {v1}.", keywords=[str(v1)])
                f2 = graph.register_fact(session=session, domain=n2,
                    content=f"{n2} is in the office directory; {attr}: {v2}.", keywords=[str(v2)])
                graph.add_interference(f1.id, f2.id, n1)
                results.append({
                    "shared_term": n1,
                    "fact_a": {"id": f1.id, "domain": n1, "value": str(v1)},
                    "fact_b": {"id": f2.id, "domain": n2, "value": str(v2)},
                    "text_a": f1.content, "text_b": f2.content,
                    "probe_question": f"What is {n1}'s {attr}? Reply with the exact number only.",
                })
            return results

        # High-similarity mode: near-twin entities (same base, minimal
        # qualifier) with CLOSE values, to actually induce mis-binding.
        if getattr(pressure, "confusable_high_similarity", False):
            bases = ["marketing budget", "engineering budget", "dining budget",
                     "travel budget", "contract deadline", "project deadline",
                     "vendor invoice", "subscription fee"]
            qual_pairs = [("Q3", "Q4"), ("2023", "2024"), ("primary", "secondary"),
                          ("North-region", "South-region"), ("east-team", "west-team"),
                          ("phase-1", "phase-2"), ("January", "February")]
            avail_bases = [b for b in bases if b not in existing_groups]
            for base in avail_bases[:n_needed]:
                q1, q2 = rng.choice(qual_pairs)
                v1 = rng.randint(200, 900)
                while v1 % 5 == 0:
                    v1 += rng.randint(1, 4)
                # v2 within ~5% of v1 (close magnitude → genuinely confusable)
                delta = max(2, rng.randint(2, max(3, v1 // 20)))
                v2 = v1 + (delta if rng.random() < 0.5 else -delta)
                while v2 % 5 == 0 or v2 == v1:
                    v2 += rng.randint(1, 4)
                f1 = graph.register_fact(session=session, domain=q1,
                    content=f"The {q1} {base} is ${v1:,}.", keywords=[f"${v1:,}", str(v1)])
                f2 = graph.register_fact(session=session, domain=q2,
                    content=f"The {q2} {base} is ${v2:,}.", keywords=[f"${v2:,}", str(v2)])
                graph.add_interference(f1.id, f2.id, base)
                results.append({
                    "shared_term": base,
                    "fact_a": {"id": f1.id, "domain": q1, "value": f"${v1:,}"},
                    "fact_b": {"id": f2.id, "domain": q2, "value": f"${v2:,}"},
                    "text_a": f1.content, "text_b": f2.content,
                })
            return results

        available_terms = [t for t in confusable_terms if t not in existing_groups]

        for term in available_terms[:n_needed]:
            term_cfg = confusable_terms[term]
            domains = term_cfg.get("domains", ["domain_a", "domain_b"])
            value_ranges = term_cfg.get("value_ranges", [(100, 500), (10000, 100000)])

            if len(domains) < 2 or len(value_ranges) < 2:
                continue

            # Pick two domains and generate confusable values
            d1, d2 = rng.sample(domains[:len(value_ranges)], 2)
            idx1, idx2 = domains.index(d1), domains.index(d2)
            r1, r2 = value_ranges[min(idx1, len(value_ranges) - 1)], value_ranges[min(idx2, len(value_ranges) - 1)]
            v1 = rng.randint(r1[0], r1[1])
            v2 = rng.randint(r2[0], r2[1])

            # Ensure values are non-round
            while v1 % 5 == 0:
                v1 += rng.randint(1, 4)
            while v2 % 5 == 0:
                v2 += rng.randint(1, 4)

            # Register as facts
            f1 = graph.register_fact(
                session=session,
                domain=d1,
                content=f"The {term} for {d1} is ${v1:,}",
                keywords=[f"${v1:,}", str(v1)],
            )
            f2 = graph.register_fact(
                session=session,
                domain=d2,
                content=f"The {term} for {d2} is ${v2:,}",
                keywords=[f"${v2:,}", str(v2)],
            )

            graph.add_interference(f1.id, f2.id, term)

            results.append({
                "shared_term": term,
                "fact_a": {"id": f1.id, "domain": d1, "value": f"${v1:,}"},
                "fact_b": {"id": f2.id, "domain": d2, "value": f"${v2:,}"},
                "text_a": f1.content,
                "text_b": f2.content,
            })

        return results

    # ------------------------------------------------------------------ invalidation

    FORGET_TEMPLATES = [
        "IMPORTANT: The information about {entity} ({domain}) is no longer accurate. "
        "The value '{old_value}' has been retracted. Do NOT cite this in future analyses.",

        "CORRECTION: Our previous finding that {entity} is now INVALID. "
        "Please disregard the value of {old_value} from session {origin_session}.",

        "UPDATE: The {domain} data point '{old_value}' for {entity} has been "
        "withdrawn due to a data quality issue. Remove it from your records.",
    ]

    def invalidate_random_facts(
        self,
        graph: FactGraph,
        session: int,
        rng: random.Random,
        pressure: PressureConfig,
    ) -> list[dict]:
        """
        Invalidate a fraction of existing facts (selective forgetting).

        Returns list of invalidation dicts that can be embedded in the session's
        environment data or task stream. Each dict has:
          - fact_id: the invalidated fact's ID
          - invalidated_keywords: keywords the agent should NOT cite after this session
          - text: human-readable retraction instruction
        """
        if pressure.forget_rate <= 0:
            return []

        invalidatable = graph.get_invalidatable_facts(before_session=session)
        if not invalidatable:
            return []

        n_to_invalidate = max(1, int(len(invalidatable) * pressure.forget_rate))
        to_invalidate = rng.sample(invalidatable, min(n_to_invalidate, len(invalidatable)))

        results = []
        for fact in to_invalidate:
            graph.invalidate_fact(fact.id, session)
            template = rng.choice(self.FORGET_TEMPLATES)
            text = template.format(
                entity=fact.content.split(".")[0][:50],
                domain=fact.domain,
                old_value=fact.keywords[0] if fact.keywords else "N/A",
                origin_session=fact.session,
            )
            results.append({
                "fact_id": fact.id,
                "invalidated_keywords": list(fact.keywords),
                "text": text,
            })

        return results

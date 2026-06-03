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


# 100 distinct English single-word qualifiers — animals, colors, foods,
# elements, countries, etc. — chosen for token-level distinctness so the
# fan-effect interference test can isolate digit-token confusion (the
# default ``phase-1 ... phase-K`` qualifiers) from generic attention-mass
# dilution. No alphabetic prefix collisions (e.g. "phone"/"phase"); no
# shared suffixes that would re-introduce sub-token overlap.
_WORD_QUALIFIERS: list[str] = [
    # animals (20)
    "tiger", "elephant", "rabbit", "eagle", "wolf", "dolphin", "owl",
    "panda", "kangaroo", "giraffe", "lobster", "crocodile", "hedgehog",
    "raccoon", "ostrich", "platypus", "weasel", "lemur", "bison", "moose",
    # colors / shades (15)
    "crimson", "azure", "magenta", "ochre", "teal", "violet", "scarlet",
    "khaki", "coral", "maroon", "turquoise", "burgundy", "lavender",
    "amber", "olive",
    # foods (15)
    "pizza", "tofu", "ramen", "burrito", "yogurt", "muffin", "lasagna",
    "kimchi", "guacamole", "espresso", "croissant", "biscotti", "falafel",
    "cupcake", "schnitzel",
    # countries / regions (15)
    "Canada", "Brazil", "Egypt", "Vietnam", "Norway", "Argentina", "Kenya",
    "Mongolia", "Iceland", "Thailand", "Morocco", "Portugal", "Finland",
    "Croatia", "Ecuador",
    # elements / minerals (15)
    "carbon", "helium", "uranium", "quartz", "silver", "tungsten",
    "platinum", "bismuth", "obsidian", "chromium", "mercury", "graphite",
    "neon", "cobalt", "iodine",
    # household objects (10)
    "lantern", "anchor", "umbrella", "ladder", "bicycle", "telescope",
    "compass", "violin", "kettle", "harmonica",
    # nature (10)
    "canyon", "glacier", "meadow", "tundra", "savanna", "lagoon",
    "geyser", "fjord", "volcano", "rainforest",
]
assert len(_WORD_QUALIFIERS) >= 100, (
    f"_WORD_QUALIFIERS pool has {len(_WORD_QUALIFIERS)} entries; "
    f"need ≥ 100 for the fan-effect digit-confusion control."
)


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

        # Honor the stochastic draw: a max(1, ...) floor was forcing >=1
        # revision every cycle, which pushed the revision rate to ~100% and
        # left ~1 unrevised fact across the whole run. The trident's
        # revision_fidelity_excess needs a healthy unrevised cohort as
        # baseline to subtract; without it the metric is None and the aging
        # card reports coverage_verdict="underpowered". Removing the floor
        # lets update_rate actually control the revised/unrevised split.
        n_to_update = int(round(len(updatable) * pressure.update_rate))
        if n_to_update == 0:
            return []
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

            # Reject revisions whose mutated value lands in the unscorable
            # short-numeric region. The scoring layer uses a digit-flank
            # guard ('2' must NOT match inside '20'), so a mutation
            # 122 → 99 (delta -23 on val=122 is in-range) produces probes
            # whose gold can never survive at scoring time. Mirrors the
            # _emittable filter in s1_generator._extract_unique_keywords;
            # without this, _attach_forbidden_keywords_retroactively would
            # also propagate the short numeric back to earlier basic probes.
            def _short_num(v: str) -> bool:
                s = v.replace(",", "")
                return s.isdigit() and len(s) < 3
            if any(_short_num(k) for k in new_keywords):
                continue

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
            fan_count = max(2, int(getattr(pressure, "confusable_fan_count", 2)))

            if fan_count == 2:
                # Original 2-way pair behavior — unchanged for backward compat.
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

            # Fan mode (K > 2): K qualifiers per base → K facts → 1 binding
            # probe on the MIDDLE qualifier (avoiding primacy/recency edge
            # advantages). Qualifiers default to programmatic ``phase-1 ...
            # phase-K`` (multi-digit). When confusable_word_qualifiers=True,
            # use a pool of 100 distinct English words instead — tests whether
            # K=100 fan-effect failures are driven by sub-token digit
            # confusion (phase-51 ≈ phase-5 / phase-11 / phase-31) rather than
            # generic attention-mass dilution.
            use_words = bool(getattr(
                pressure, "confusable_word_qualifiers", False
            ))
            for base in avail_bases[:n_needed]:
                if use_words:
                    quals = _WORD_QUALIFIERS[:fan_count]
                    if len(quals) < fan_count:
                        raise ValueError(
                            f"confusable_word_qualifiers pool has only "
                            f"{len(_WORD_QUALIFIERS)} entries; "
                            f"requested fan_count={fan_count}."
                        )
                else:
                    quals = [f"phase-{k}" for k in range(1, fan_count + 1)]
                # Probe position: middle by default; configurable via the
                # confusable_probe_index knob to test "lost in the middle".
                _probe_idx_cfg = getattr(pressure, "confusable_probe_index", None)
                if _probe_idx_cfg is None:
                    probe_idx_override = None
                else:
                    if _probe_idx_cfg < 0:
                        probe_idx_override = fan_count + _probe_idx_cfg
                    else:
                        probe_idx_override = _probe_idx_cfg
                    probe_idx_override = max(0, min(fan_count - 1, probe_idx_override))
                values: list[int] = []
                while len(values) < fan_count:
                    v = rng.randint(200, 900)
                    if v % 5 == 0:
                        v += rng.randint(1, 4)
                    if all(abs(v - u) >= 2 for u in values):
                        values.append(v)
                probe_idx = (
                    fan_count // 2 if probe_idx_override is None
                    else probe_idx_override
                )
                fact_ids: list[str] = []
                contents: list[str] = []
                for q, v in zip(quals, values):
                    f = graph.register_fact(
                        session=session, domain=q,
                        content=f"The {q} {base} is ${v:,}.",
                        keywords=[f"${v:,}", str(v)],
                    )
                    fact_ids.append(f.id)
                    contents.append(f.content)
                gold_id = fact_ids[probe_idx]
                for i, fid in enumerate(fact_ids):
                    if i != probe_idx:
                        graph.add_interference(gold_id, fid, base)
                gold_q, gold_v = quals[probe_idx], values[probe_idx]
                dist_i = 0 if probe_idx != 0 else 1
                results.append({
                    "shared_term": base,
                    "fact_a": {"id": gold_id, "domain": gold_q,
                               "value": f"${gold_v:,}"},
                    "fact_b": {"id": fact_ids[dist_i], "domain": quals[dist_i],
                               "value": f"${values[dist_i]:,}"},
                    "text_a": contents[probe_idx], "text_b": contents[dist_i],
                    "probe_question": (
                        f"What is the exact {gold_q} {base}? "
                        f"Reply with the exact value only."
                    ),
                    # All K qualifier facts for this base. Consumers (e.g.
                    # s6_generator) write the full list into env_data so the
                    # agent sees the entire fan-out, not just gold+distractor.
                    "fan_texts": contents,
                    "fan_distractors": [
                        {"qualifier": q, "value": f"${v:,}"}
                        for i, (q, v) in enumerate(zip(quals, values))
                        if i != probe_idx
                    ],
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
                # Forced-choice binding probe: ask for fact_a's value with
                # fact_b's value as the confusable distractor (shared term, two
                # domains). Lets score_interference_binding classify
                # correct/confused/miss instead of the resistance proxy.
                "probe_question": (
                    f"What is the {term} for {d1}? Reply with the exact value only."
                ),
            })

        return results

    # ------------------------------------------------------------------ unique-singleton controls

    # Pool of "lonely" people — no near-duplicate names — used as within-session
    # controls for the crowding-out test. These names are deliberately distinct
    # from the similar-name pairs in ``inject_interference`` (no John/Jon, Sara/
    # Sarah, etc.) so they probe SAME-shape facts (name + attribute + value)
    # without a confusable competitor in memory.
    _UNIQUE_CONTROL_NAMES = [
        "Priya Iyengar", "Tomohiro Watanabe", "Olusola Adebayo", "Mei-Ling Tan",
        "Aleksandr Volkov", "Beatrice Fournier", "Kwame Asante", "Larissa Petrov",
        "Mateusz Kowalczyk", "Yuki Hashimoto", "Ingrid Bjornsen", "Rafael Quintero",
        "Xiomara Vasquez", "Devanshi Rao", "Hiroko Yamamoto", "Anya Kapoor",
        "Tarek Mansour", "Solange Mbeki", "Naveen Sundaram", "Camila Restrepo",
    ]

    def inject_unique_controls(
        self,
        graph,
        session: int,
        rng: random.Random,
        pressure: PressureConfig,
        existing_control_names: Optional[set] = None,
    ) -> list[dict]:
        """Inject unique-singleton control people (no near-duplicate in memory).

        Mirrors ``inject_interference``'s similar-name branch — same fact shape
        (name + attribute + value), same attribute pool — but each control name
        is a singleton with no confusable partner. Used as the WITHIN-session
        control arm of the crowding-out test: at any given session and bloat
        level, the controls show whether bloat-driven retrieval failure is
        confusable-specific (controls fine, binding probes collapse) or generic
        (both collapse together).

        Returns control probe dicts in the same schema as the binding probes
        emitted by ``inject_interference``, tagged ``probe_type=unique_control``
        with a phantom distractor (a random non-existent number) so the existing
        ``score_interference_binding`` classifies them into correct/miss without
        false confusion hits.
        """
        n_controls = int(getattr(pressure, "n_unique_controls", 0) or 0)
        if n_controls <= 0:
            return []

        attrs = ["extension", "desk number", "employee ID"]
        existing = existing_control_names or set()
        n_existing = len(existing)
        n_needed = n_controls - n_existing
        if n_needed <= 0:
            return []
        avail = [n for n in self._UNIQUE_CONTROL_NAMES if n not in existing]
        results = []
        for name in avail[:n_needed]:
            attr = rng.choice(attrs)
            val = rng.randint(1000, 9999)
            phantom = rng.randint(1000, 9999)
            while phantom == val:
                phantom = rng.randint(1000, 9999)
            f = graph.register_fact(
                session=session, domain=name,
                content=f"{name} is in the office directory; {attr}: {val}.",
                keywords=[str(val)],
            )
            results.append({
                "shared_term": name,
                "fact_a": {"id": f.id, "domain": name, "value": str(val)},
                "fact_b": None,
                "text_a": f.content,
                "text_b": "",
                "probe_question": f"What is {name}'s {attr}? Reply with the exact number only.",
                "phantom_distractor": str(phantom),
                "is_unique_control": True,
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

        # Honor the stochastic draw (see invalidate_random_facts' sibling in
        # version_random_facts above): the max(1, ...) floor previously
        # guaranteed an invalidation every cycle even when the rate said
        # zero, distorting the revised/unrevised ratio used by trident
        # baselines.
        n_to_invalidate = int(round(len(invalidatable) * pressure.forget_rate))
        if n_to_invalidate == 0:
            return []
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

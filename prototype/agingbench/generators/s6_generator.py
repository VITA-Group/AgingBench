"""Programmatic generator for S6 Naturalistic Aging scenario.

Produces multi-domain session data (shopping, travel, project management)
with environment data tables, primary tasks, and recall probes — all in the
exact same JSON format as the curated ``session_tasks.json``.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator
from .dependency_mixin import DependencyMixin
from .fact_graph import FactGraph
from .pools import (
    PRODUCT_NAMES, COMPANY_NAMES, RESTAURANT_NAMES, CITY_NAMES,
    UNIVERSITY_NAMES, PARK_NAMES, PROJECT_COMPONENTS, PROJECT_MILESTONES,
    FIRST_NAMES, LAST_NAMES, CUISINE_TYPES,
    random_dollar, random_count, random_distance_km, random_percent,
    random_date, random_latency_ms, random_person, ensure_non_round,
    sample_unique,
)
from .pressure_config import PressureConfig

# Domain rotation — cycling through these for variety
_DOMAINS = [
    "shopping_admin", "shopping_admin", "shopping", "shopping_admin",  # 0-3
    "map", "map", "map_wikipedia", "map_wikipedia",                    # 5-8
    "gitlab", "gitlab", "reddit", "gitlab",                            # 10-13
]


class S6Generator(BaseGenerator, DependencyMixin):
    """Generate S6 naturalistic aging scenario data."""

    SCENARIO_ID = "s6_naturalistic"

    def __init__(self, seed: int = 42, pressure: PressureConfig | None = None):
        super().__init__(seed)
        self.pressure = pressure or PressureConfig.none()

    def generate(self, n_sessions: int = 15) -> dict[str, Any]:
        graph = FactGraph()
        sessions = []
        # Track all generated facts for cross-reference sessions
        all_facts: list[dict] = []  # [{session_id, key_fact, keywords}, ...]

        xref_interval = max(n_sessions // 3, 4)
        non_xref_idx = 0

        for i in range(n_sessions):
            is_xref = (i > 0 and i % xref_interval == xref_interval - 1)

            if is_xref:
                session = self._generate_xref_session(i, all_facts)
            else:
                domain = _DOMAINS[non_xref_idx % len(_DOMAINS)]
                session, facts = self._generate_data_session(i, domain, non_xref_idx)
                all_facts.extend(facts)
                # Register each fact in the FactGraph
                for fact in facts:
                    graph.register_fact(
                        session=fact["session_id"],
                        domain=domain,
                        content=fact["key_fact"],
                        keywords=fact["keywords"],
                    )
                non_xref_idx += 1

            # Apply dependency task replacement after warmup.
            # Skip on xref sessions: their task is already a carefully-redacted
            # cross-domain synthesis test (see _generate_xref_session). Letting
            # build_dependency_task overwrite it would replace the value-masked
            # xref prompt with a generic synthesize template — losing the
            # multi-session memory-test design.
            if (i >= self.pressure.warmup_sessions
                and not is_xref
                and self.rng.random() < self.pressure.dependency_density):
                dep_task = self.build_dependency_task(graph, i, self.rng, self.pressure)
                if dep_task:
                    session["task"] = {
                        "text": dep_task["text"],
                        "reference_answer": dep_task["reference_answer"],
                        "eval_keywords": dep_task["eval_keywords"],
                    }

            # Apply version updates
            updates = self.version_random_facts(graph, i, self.rng, self.pressure)
            if updates:
                update_text = "\n".join(u["text"] for u in updates)
                session["environment_data"] = session.get("environment_data", "") + "\n\n" + update_text
                self._sync_probes_after_revisions(sessions, all_facts, graph, updates, i)

            # Apply selective forgetting (revision aging)
            invalidations = self.invalidate_random_facts(graph, i, self.rng, self.pressure)
            if invalidations:
                inv_text = "\n".join(inv["text"] for inv in invalidations)
                session["environment_data"] = session.get("environment_data", "") + "\n\n" + inv_text
                # Retire the invalidated fact's recall probe from session `i`
                # onward. Otherwise the headline recall_rate keeps re-asking it
                # with the RETRACTED keywords as gold, rewarding an agent that
                # cites the now-invalid value and penalizing correct forgetting
                # (the "penalizes correct revision" bug). forget_accuracy still
                # measures the retraction separately. Recall before invalidation
                # is preserved — the probe is only skipped for sessions >= i.
                self._sync_probes_after_invalidations(sessions, graph, invalidations, i)

            # Inject interference facts (confusable cross-domain pairs)
            if i >= self.pressure.confusable_start_session:
                pairs = self.inject_interference(graph, i, self.rng, self.pressure)
                if pairs:
                    # When inject_interference produced a fan-out (K > 2
                    # qualifiers per base), emit all K facts. Otherwise the
                    # legacy gold+distractor pair.
                    interf_text = "\n".join(
                        "\n".join(p["fan_texts"]) if p.get("fan_texts")
                        else f"{p['text_a']}\n{p['text_b']}"
                        for p in pairs
                    )
                    session["environment_data"] = session.get("environment_data", "") + "\n\n" + interf_text
                    # Forced binding probes (gold+distractor) are emitted for
                    # EVERY injected pair — including the default value-confusable
                    # pairs — so interference is measured by default, not only in
                    # the explicit similar-name / high-similarity modes. A binding
                    # probe is a legitimate recall probe (citing the gold value =
                    # recalled; citing the distractor = recall failure), so it
                    # counts toward recall_rate AND feeds score_interference_binding
                    # (correct/confused/miss). recall_rate_excl_binding stays as the
                    # binding-free series for apples-to-apples config comparison.
                    # S6 re-asks all prior probes every session → free lag/density
                    # gradient at no extra agent calls.
                    probes = session.setdefault("recall_probes", [])
                    for j, p in enumerate(pairs):
                        q = p.get("probe_question") or (
                            f"What is the exact {p['fact_a']['domain']} "
                            f"{p['shared_term']}? Reply with the exact value only."
                        )
                        probes.append({
                            "probe_id": f"s{i}_interf_{j}",
                            "question": q,
                            "keywords": [str(p["fact_a"]["value"])],
                            "canonical_answer": str(p["fact_a"]["value"]),
                            "gold_value": p["fact_a"]["value"],
                            "distractor_value": p["fact_b"]["value"],
                            "probe_type": "interference_binding",
                        })

            sessions.append(session)

        result = {
            "session_tasks": {
                "benchmark_source": "AgingBench programmatic generator",
                "description": f"Generated S6 naturalistic scenario with {n_sessions} sessions.",
                "system_prompt": (
                    "You are a research analyst assistant. Your job is to analyze "
                    "data from various platforms (e-commerce dashboards, mapping "
                    "services, project management tools), answer questions accurately, "
                    "and remember findings for future reference. When answering, be "
                    "precise with names, numbers, and specific details."
                ),
                "sessions": sessions,
            }
        }
        result["dependency_graph"] = graph.export()
        return result

    # ------------------------------------------------------------------
    # Data session generators (one per domain type)
    # ------------------------------------------------------------------

    def _generate_data_session(
        self, session_id: int, domain: str, seq_idx: int,
    ) -> tuple[dict, list[dict]]:
        """Generate a non-cross-reference session with environment data."""
        generators = {
            "shopping_admin": self._gen_shopping_admin,
            "shopping": self._gen_shopping_reviews,
            "map": self._gen_map,
            "map_wikipedia": self._gen_map_wikipedia,
            "gitlab": self._gen_gitlab,
            "reddit": self._gen_reddit,
        }
        gen_fn = generators.get(domain, self._gen_shopping_admin)
        return gen_fn(session_id, seq_idx)

    def _gen_shopping_admin(self, sid: int, seq: int) -> tuple[dict, list[dict]]:
        """Shopping admin: sales report table."""
        year = self.rng.choice([2022, 2023, 2024])
        products = sample_unique(PRODUCT_NAMES, 8, self.rng)
        units = [random_count(self.rng, 500, 8000) for _ in products]
        revenues = [random_dollar(self.rng, 5000, 200000) for _ in products]
        # Sort by units descending
        ranked = sorted(zip(products, units, revenues), key=lambda x: -x[1])
        total_rev = sum(r for _, _, r in ranked)

        rows = "\n".join(
            f"| {i+1:<4} | {name:<25} | {f'{u:,}':<10} | ${f'{rev:,}':<11} |"
            for i, (name, u, rev) in enumerate(ranked)
        )
        env_data = (
            f"=== E-Commerce Admin Dashboard: Annual Sales Report {year} ===\n\n"
            f"Top {len(ranked)} Best-Selling Products (by units sold):\n\n"
            f"| Rank | Product Name              | Units Sold | Revenue      |\n"
            f"|------|---------------------------|------------|------------- |\n"
            f"{rows}\n\n"
            f"Total revenue: ${total_rev:,}"
        )

        top_name, top_units, top_rev = ranked[0]
        task = {
            "text": f"Based on the annual sales report for {year}, answer: "
                    f"What is the top-1 best-selling product, and how many units were sold?",
            "reference_answer": f"{top_name} with {top_units:,} units sold",
            "eval_keywords": [top_name, f"{top_units:,}"],
        }

        probes = [
            {
                "probe_id": f"s{sid}_p0",
                "question": f"What was the best-selling product in our {year} sales report?",
                "keywords": [top_name],
                "canonical_answer": top_name,
            },
            {
                "probe_id": f"s{sid}_p1",
                "question": f"How much total revenue did our platform generate in {year}?",
                "keywords": [f"{total_rev:,}", str(total_rev)],
                "canonical_answer": f"${total_rev:,}",
            },
        ]

        facts = [
            {"session_id": sid, "key_fact": f"Top product: {top_name}", "keywords": [top_name]},
            {"session_id": sid, "key_fact": f"Revenue: ${total_rev:,}", "keywords": [f"{total_rev:,}"]},
        ]

        return {
            "session_id": sid,
            "source_task_id": f"generated_{sid}",
            "domain": "shopping_admin",
            "environment_data": env_data,
            "task": task,
            "recall_probes": probes,
        }, facts

    def _gen_shopping_reviews(self, sid: int, seq: int) -> tuple[dict, list[dict]]:
        """Shopping: product review analysis."""
        product = self.rng.choice(PRODUCT_NAMES)
        n_reviews = random_count(self.rng, 50, 500)
        issue = self.rng.choice([
            "ear cups being too small", "battery draining quickly",
            "strap breaking after a week", "screen scratching easily",
            "buttons being unresponsive",
        ])
        reviewers = [
            f"{self.rng.choice(FIRST_NAMES)}{self.rng.choice(['B', 'K', 'M', 'S', 'T'])}{self.rng.randint(10,99)}"
            for _ in range(4)
        ]
        review_texts = "\n\n".join(
            f"{i+1}. {'★' * self.rng.randint(2, 4)}{'☆' * (5 - self.rng.randint(2, 4))} "
            f"by {rev} ({random_date(self.rng)}):\n"
            f'   "Mostly good product but {issue}. Would rate higher otherwise."'
            for i, rev in enumerate(reviewers)
        )

        env_data = (
            f"=== Product Reviews: {product} ===\n\n"
            f"Overall Rating: {self.rng.uniform(3.5, 4.5):.1f}/5 ({n_reviews} reviews)\n\n"
            f"Filtered reviews mentioning '{issue.split()[0]}':\n\n"
            f"{review_texts}\n\n"
            f"Total reviews mentioning this issue: {len(reviewers)} out of {n_reviews}"
        )

        task = {
            "text": f"From the product reviews for {product}, list the reviewers who "
                    f"mentioned {issue}.",
            "reference_answer": ", ".join(reviewers),
            "eval_keywords": reviewers[:2],
        }
        probes = [
            {
                "probe_id": f"s{sid}_p0",
                "question": f"Which reviewers complained about {issue} for {product}?",
                "keywords": reviewers[:2],
                "canonical_answer": ", ".join(reviewers),
            },
            {
                "probe_id": f"s{sid}_p1",
                "question": f"How many total reviews does {product} have?",
                "keywords": [str(n_reviews), f"{n_reviews:,}"],
                "canonical_answer": str(n_reviews),
            },
        ]
        facts = [
            {"session_id": sid, "key_fact": f"Reviewers: {', '.join(reviewers[:2])}", "keywords": reviewers[:2]},
        ]
        return {
            "session_id": sid, "source_task_id": f"generated_{sid}",
            "domain": "shopping", "environment_data": env_data,
            "task": task, "recall_probes": probes,
        }, facts

    def _gen_map(self, sid: int, seq: int) -> tuple[dict, list[dict]]:
        """Map: distance/direction queries."""
        city_a, city_b = sample_unique(CITY_NAMES, 2, self.rng)
        dist = random_distance_km(self.rng, 100, 1500)
        hours = dist // 80
        mins = ensure_non_round(self.rng.randint(10, 55), self.rng)
        restaurants = sample_unique(RESTAURANT_NAMES, 5, self.rng)

        env_data = (
            f"=== Map Service: Route from {city_a} to {city_b} ===\n\n"
            f"Driving distance: {dist} km ({dist * 621 // 1000} miles)\n"
            f"Estimated driving time: {hours}h {mins}min\n"
            f"Route: via Interstate Highway\n\n"
            f"Nearby restaurants in {city_b}:\n"
            + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(restaurants))
        )

        task = {
            "text": f"What is the driving distance from {city_a} to {city_b}?",
            "reference_answer": f"{dist} km",
            "eval_keywords": [str(dist)],
        }
        probes = [
            {
                "probe_id": f"s{sid}_p0",
                "question": f"How far is it to drive from {city_a} to {city_b}?",
                "keywords": [str(dist)],
                "canonical_answer": f"{dist} km",
            },
            {
                "probe_id": f"s{sid}_p1",
                "question": f"Name a restaurant near {city_b} from our map search.",
                "keywords": [restaurants[0]],
                "canonical_answer": restaurants[0],
            },
        ]
        facts = [
            {"session_id": sid, "key_fact": f"{city_a}→{city_b}: {dist}km", "keywords": [str(dist)]},
            {"session_id": sid, "key_fact": f"Restaurant: {restaurants[0]}", "keywords": [restaurants[0]]},
        ]
        return {
            "session_id": sid, "source_task_id": f"generated_{sid}",
            "domain": "map", "environment_data": env_data,
            "task": task, "recall_probes": probes,
        }, facts

    def _gen_map_wikipedia(self, sid: int, seq: int) -> tuple[dict, list[dict]]:
        """Map + Wikipedia: university/park cross-reference."""
        uni = self.rng.choice(UNIVERSITY_NAMES)
        park = self.rng.choice(PARK_NAMES)
        dist = random_distance_km(self.rng, 50, 800)
        area = random_count(self.rng, 1000, 50000)

        env_data = (
            f"=== Map + Wikipedia: {park} from {uni} ===\n\n"
            f"Closest national park to {uni}: {park}\n"
            f"Driving distance: {dist} km\n"
            f"Park area: {area:,} acres\n"
            f"Annual visitors: {random_count(self.rng, 500000, 5000000):,}\n"
        )

        task = {
            "text": f"What is the closest national park to {uni}, and how far is the drive?",
            "reference_answer": f"{park}, {dist} km",
            "eval_keywords": [park, str(dist)],
        }
        probes = [
            {
                "probe_id": f"s{sid}_p0",
                "question": f"What national park is closest to {uni}?",
                "keywords": [park],
                "canonical_answer": park,
            },
            {
                "probe_id": f"s{sid}_p1",
                "question": f"How large is {park} in acres?",
                "keywords": [f"{area:,}", str(area)],
                "canonical_answer": f"{area:,} acres",
            },
        ]
        facts = [
            {"session_id": sid, "key_fact": f"Park: {park}, {dist}km from {uni}", "keywords": [park, str(dist)]},
        ]
        return {
            "session_id": sid, "source_task_id": f"generated_{sid}",
            "domain": "map_wikipedia", "environment_data": env_data,
            "task": task, "recall_probes": probes,
        }, facts

    def _gen_gitlab(self, sid: int, seq: int) -> tuple[dict, list[dict]]:
        """GitLab: issue/MR tracking."""
        repo = f"{self.rng.choice(FIRST_NAMES).lower()}-{self.rng.choice(['api', 'sdk', 'cli', 'core', 'lib'])}"
        n_issues = random_count(self.rng, 5, 50)
        n_members = random_count(self.rng, 3, 20)
        author = random_person(self.rng)
        mr_title = self.rng.choice([
            "Fix authentication token refresh",
            "Add rate limiting middleware",
            "Update dependency versions",
            "Refactor database connection pool",
            "Add pagination to list endpoints",
        ])

        env_data = (
            f"=== GitLab: {repo} Repository ===\n\n"
            f"Open issues: {n_issues}\n"
            f"Team members: {n_members}\n\n"
            f"Latest merge request:\n"
            f"  Title: {mr_title}\n"
            f"  Author: {author}\n"
            f"  Status: Merged\n"
            f"  Date: {random_date(self.rng)}\n"
        )

        task = {
            "text": f"What is the latest merge request title in the {repo} repository, "
                    f"and who submitted it?",
            "reference_answer": f"{mr_title} by {author}",
            "eval_keywords": [author.split()[0], mr_title.split()[0]],
        }
        probes = [
            {
                "probe_id": f"s{sid}_p0",
                "question": f"Who submitted the latest MR in the {repo} repo?",
                "keywords": [author.split()[0], author.split()[-1]],
                "canonical_answer": author,
            },
            {
                "probe_id": f"s{sid}_p1",
                "question": f"How many open issues are in the {repo} repo?",
                "keywords": [str(n_issues)],
                "canonical_answer": str(n_issues),
            },
        ]
        facts = [
            {"session_id": sid, "key_fact": f"MR author: {author}", "keywords": [author.split()[-1]]},
            {"session_id": sid, "key_fact": f"Issues: {n_issues}", "keywords": [str(n_issues)]},
        ]
        return {
            "session_id": sid, "source_task_id": f"generated_{sid}",
            "domain": "gitlab", "environment_data": env_data,
            "task": task, "recall_probes": probes,
        }, facts

    def _gen_reddit(self, sid: int, seq: int) -> tuple[dict, list[dict]]:
        """Reddit: forum post analysis."""
        subreddit = self.rng.choice([
            "Showerthoughts", "TodayILearned", "AskScience", "ExplainLikeImFive",
        ])
        username = f"{self.rng.choice(FIRST_NAMES)}{self.rng.randint(100, 999)}"
        n_comments = random_count(self.rng, 10, 200)
        n_downvoted = ensure_non_round(self.rng.randint(1, min(n_comments, 15)), self.rng)

        env_data = (
            f"=== Reddit: r/{subreddit} ===\n\n"
            f"Latest post by u/{username}:\n"
            f"  Title: \"{self.rng.choice(['Why do we', 'How come', 'What if'])} "
            f"{self.rng.choice(['dreams feel real', 'time moves faster when busy', 'mirrors flip left-right'])}\"\n"
            f"  Comments: {n_comments}\n"
            f"  Comments with more downvotes than upvotes: {n_downvoted}\n"
        )

        task = {
            "text": f"In r/{subreddit}, who made the latest post and how many comments "
                    f"received more downvotes than upvotes?",
            "reference_answer": f"u/{username}, {n_downvoted} comments",
            "eval_keywords": [username, str(n_downvoted)],
        }
        probes = [
            {
                "probe_id": f"s{sid}_p0",
                "question": f"Who posted the latest thread in r/{subreddit}?",
                "keywords": [username],
                "canonical_answer": f"u/{username}",
            },
            {
                "probe_id": f"s{sid}_p1",
                "question": f"How many downvoted comments were in that r/{subreddit} thread?",
                "keywords": [str(n_downvoted)],
                "canonical_answer": str(n_downvoted),
            },
        ]
        facts = [
            {"session_id": sid, "key_fact": f"Poster: {username}", "keywords": [username]},
        ]
        return {
            "session_id": sid, "source_task_id": f"generated_{sid}",
            "domain": "reddit", "environment_data": env_data,
            "task": task, "recall_probes": probes,
        }, facts

    # ------------------------------------------------------------------
    # Cross-reference session
    # ------------------------------------------------------------------

    def _sync_probes_after_revisions(
        self,
        sessions: list[dict],
        all_facts: list[dict],
        graph: FactGraph,
        updates: list[dict],
        revision_session: int,
    ) -> None:
        """Propagate `version_random_facts` updates to the originating
        session's `recall_probes` and `all_facts` registry entry, so the
        probe expects the current (post-revision) keywords.

        Mapping is position-aligned with `version_random_facts`. A probe is
        only updated when its keyword set is a subset of the fact's old
        keywords, to avoid cross-fact mutation when two facts share a token.
        """
        for upd in updates:
            old_fact = graph.facts.get(upd["old_fact_id"])
            if old_fact is None:
                continue
            origin = old_fact.session
            if not (0 <= origin < len(sessions)):
                continue
            old_kws = list(upd["old_keywords"])
            new_kws = list(upd["new_keywords"])
            kw_map = {o: n for o, n in zip(old_kws, new_kws) if o != n}
            if not kw_map:
                continue
            old_set = set(old_kws)

            def _remap(seq: list[str]) -> list[str]:
                return [kw_map.get(k, k) for k in seq]

            for probe in sessions[origin].get("recall_probes", []):
                pkws = probe.get("keywords") or []
                if pkws and set(pkws) <= old_set:
                    # Time-versioned gold (mirrors S3/S5): lazily seed history
                    # with the pre-revision value at the fact's origin session,
                    # then append the post-revision value at the revision
                    # session, so the validator can score each re-asked probe
                    # against the value active at THAT eval session (not the
                    # final value). Non-revised probes carry no history and the
                    # validator falls back to probe["keywords"].
                    new_pkws = _remap(pkws)
                    hist = probe.setdefault("keywords_history", [(origin, list(pkws))])
                    hist.append((revision_session, list(new_pkws)))
                    probe["keywords"] = new_pkws
                    # Binding probes carry parallel gold fields for fact_a;
                    # keep them in sync (substring remap) so the binding scorer
                    # and the P3 oracle don't use stale gold and penalize the
                    # agent for citing the correct revised value. (distractor_value
                    # belongs to fact_b and is synced by fact_b's own update.)
                    for _field in ("canonical_answer", "gold_value"):
                        _val = probe.get(_field)
                        if _val is None:
                            continue
                        _sval = str(_val)
                        for _o, _n in zip(old_kws, new_kws):
                            if _o != _n and _o in _sval:
                                _sval = _sval.replace(_o, _n)
                        probe[_field] = _sval
            for fact in all_facts:
                if fact.get("session_id") != origin:
                    continue
                fkws = fact.get("keywords") or []
                if fkws and set(fkws) <= old_set:
                    new_fkws = _remap(fkws)
                    fhist = fact.setdefault("keywords_history", [(origin, list(fkws))])
                    fhist.append((revision_session, list(new_fkws)))
                    fact["keywords"] = new_fkws
                    kf = fact.get("key_fact", "")
                    for old_kw, new_kw in zip(old_kws, new_kws):
                        if old_kw != new_kw and old_kw in kf:
                            kf = kf.replace(old_kw, new_kw)
                    fact["key_fact"] = kf

    def _sync_probes_after_invalidations(
        self,
        sessions: list[dict],
        graph: FactGraph,
        invalidations: list[dict],
        session: int,
    ) -> None:
        """Mark the originating session's recall probes for an invalidated fact
        as retired from ``session`` onward (``invalidated_at_session``). The
        runner skips probes whose ``invalidated_at_session <= current session``,
        so a retracted fact is recalled-scored only while it was still valid and
        is then dropped from the recall pool — instead of the headline rewarding
        citations of the retracted value.

        A probe is matched only when its keyword set is a subset of the
        invalidated keywords, to avoid retiring a sibling probe that shares a
        token. The earliest invalidation session wins (probes are visited in
        increasing session order, so we never overwrite with a later one)."""
        for inv in invalidations:
            fact = graph.facts.get(inv["fact_id"])
            if fact is None:
                continue
            origin = fact.session
            if not (0 <= origin < len(sessions)):
                continue
            inv_set = set(inv.get("invalidated_keywords") or [])
            if not inv_set:
                continue
            for probe in sessions[origin].get("recall_probes", []):
                pkws = probe.get("keywords") or []
                if pkws and set(pkws) <= inv_set:
                    if probe.get("invalidated_at_session") is None:
                        probe["invalidated_at_session"] = session

    def _generate_xref_session(self, sid: int, all_facts: list[dict]) -> dict:
        """Generate a cross-reference session requiring synthesis from memory.

        Earlier the task text literally embedded the fact values
        (``- Revenue: $785,163``) and used those same values as eval_keywords,
        making the xref task a copy-echo of the prompt rather than a memory
        synthesis test. The redaction below preserves the topic structure
        ("Revenue: $[?]") so the agent knows what categories to recall, while
        masking the specific values it must retrieve from memory.
        """
        # Pick 3-5 facts from different prior sessions
        available = [f for f in all_facts if f["session_id"] < sid]
        if len(available) < 3:
            n_pick = len(available)
        else:
            n_pick = min(self.rng.randint(3, 5), len(available))
        selected = sample_unique(available, n_pick, self.rng)

        # Build redacted (value-stripped) fact prompts — preserves the
        # topic/label structure but masks the value the agent must recall.
        redacted_prompts: list[str] = []
        fact_summaries: list[str] = []
        all_kw: list[str] = []
        for f in selected:
            fact_summaries.append(f["key_fact"])
            all_kw.extend(f["keywords"])
            redacted = f["key_fact"]
            # Replace longer keywords first so we don't leave partial residues
            # (e.g. "785,215" masked before "785" inside a different token).
            for kw in sorted(f.get("keywords", []), key=len, reverse=True):
                if kw:
                    redacted = redacted.replace(kw, "[?]")
            redacted_prompts.append(redacted)

        task_text = (
            "Based on your memory of our previous research sessions, recall the "
            "specific details from our past analyses on each of the topics below. "
            "Provide the original values, names, and specifics — don't repeat the "
            "placeholders:\n"
            + "\n".join(f"- {p}" for p in redacted_prompts)
            + "\n\nBe precise with numbers, names, and proper nouns where you can "
            "recall them."
        )

        # Determine domain label
        unique_domains = set()
        for f in selected:
            unique_domains.add(f.get("domain", "mixed"))

        return {
            "session_id": sid,
            "source_task_id": f"cross_reference_{sid}",
            "domain": "all" if len(unique_domains) > 1 else next(iter(unique_domains), "all"),
            "is_cross_reference": True,
            "environment_data": "",
            "task": {
                "text": task_text,
                "reference_answer": "; ".join(fact_summaries),
                "eval_keywords": all_kw[:6],  # cap at 6 keywords
            },
            "recall_probes": [],  # xref tests synthesis, not new-fact recall
        }

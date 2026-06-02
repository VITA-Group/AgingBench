"""
agingbench/generators/s5_generator.py — Task stream generator for S5 (Self-Planning Notebook, Tier 1 with workspace-file access).

Generates 120+ tasks across session blocks with controlled mix of task types
designed to expose aging mechanisms:
  - new_info:        Present facts to remember (targets storage)
  - recall_precise:  Ask for specific value among competitors (targets precision)
  - recall_compare:  Cross-reference across periods/entities (targets consistency)
  - update:          Correct a previous fact (targets propagation)
  - plan:            Recommend based on all accumulated context (targets complexity)
  - repeat_baseline: Same task as session 0, re-asked later (targets fatigue)

Domain-configurable: "assistant", "knowledge_base", "coding"
"""

from __future__ import annotations

import random
from typing import Any

from .base import BaseGenerator
from .dependency_mixin import DependencyMixin
from .fact_graph import FactGraph
from . import pools
from .pressure_config import PressureConfig


# ── Domain-specific templates ──────────────────────────────────────────────

_DOMAINS = {
    "assistant": {
        "info_templates": [
            "Your monthly {category} budget is ${amount}. This was set on {date}.",
            "{person}'s birthday is {date}. They prefer {preference} as gifts.",
            "Your preferred {service} provider is {provider}. Account number: {account}.",
            "You have a recurring {appointment} every {schedule} at {location}.",
            "Your {category} membership is with {provider}, costing ${amount}/month. Member ID: {account}.",
            "Important: you are allergic to {allergen}. This was confirmed by Dr. {doctor} on {date}.",
        ],
        "recall_templates": [
            "What is my {category} budget?",
            "When is {person}'s birthday and what do they prefer as gifts?",
            "What is my account number with {provider}?",
            "Where and when is my recurring {appointment}?",
            "How much does my {category} membership cost per month?",
            "What am I allergic to, and who confirmed it?",
        ],
        "compare_templates": [
            "Compare my {category1} and {category2} budgets. Which is higher?",
            "List all recurring expenses and their total monthly cost.",
            "Which of my appointments are scheduled on weekdays vs weekends?",
        ],
        "plan_templates": [
            "Plan a birthday dinner for {person}, considering all my preferences and constraints.",
            "I have ${amount} to spend this month. Based on all my budgets and subscriptions, how much is left for discretionary spending?",
            "Suggest a weekend activity considering my schedule, allergies, and preferences.",
        ],
        "update_templates": [
            "Update: my {category} budget has changed from ${old_amount} to ${new_amount} effective today.",
            "Correction: {person}'s birthday is actually {new_date}, not {old_date}.",
            "I switched my {service} provider from {old_provider} to {new_provider}. New account: {new_account}.",
        ],
        "categories": ["dining", "entertainment", "groceries", "fitness", "transport", "clothing"],
        "services": ["streaming", "phone", "internet", "insurance", "gym", "cloud storage"],
    },
    "knowledge_base": {
        "info_templates": [
            "Meeting decision ({date}): {decision}. Approved by {person}. Budget impact: ${amount}.",
            "Project milestone: {milestone} deadline is {date}. Owner: {person}. Status: {status}.",
            "Vendor {vendor} was selected for {service}. Contract value: ${amount}. Start date: {date}.",
            "Team update: {person} will lead {project} starting {date}. Previous lead was {old_person}.",
            "Risk identified: {risk}. Severity: {severity}. Mitigation owner: {person}. Due: {date}.",
            "Budget revision: {category} allocation changed to ${amount} (was ${old_amount}). Reason: {reason}.",
        ],
        "recall_templates": [
            "What was the decision made on {date} about {topic}?",
            "Who approved the {decision} and what was the budget impact?",
            "What is the deadline for {milestone} and who owns it?",
            "Which vendor was selected for {service} and what's the contract value?",
            "Who is currently leading {project}?",
            "What is the current budget allocation for {category}?",
        ],
        "compare_templates": [
            "Compare the budget allocations for {category1} and {category2}.",
            "List all decisions made by {person} and their total budget impact.",
            "Which milestones are overdue as of {date}?",
        ],
        "plan_templates": [
            "Given all project decisions so far, summarize the current state of {project}.",
            "Based on all budget revisions, what is the remaining unallocated budget?",
            "Identify any conflicting decisions or timeline overlaps across projects.",
        ],
        "update_templates": [
            "Correction: the {milestone} deadline has moved from {old_date} to {new_date}.",
            "Update: {old_person} is no longer leading {project}. {new_person} takes over.",
            "Budget update: {category} allocation revised from ${old_amount} to ${new_amount}.",
        ],
        "categories": ["infrastructure", "marketing", "R&D", "operations", "security", "hiring"],
        "services": ["cloud hosting", "CI/CD", "monitoring", "design tools", "analytics", "legal review"],
    },
    "coding": {
        "info_templates": [
            "Design decision: {module} uses {technology} for {purpose}. Rationale: {rationale}.",
            "API endpoint {endpoint} accepts {params} and returns {response}. Auth: {auth_method}.",
            "Database schema: {table} has columns {columns}. Primary key: {pk}. Index on: {index}.",
            "Configuration: {config_key} = {config_value}. This affects {affected_modules}.",
            "Bug report #{bug_id}: {description}. Root cause: {root_cause}. Fixed in: {module}.",
            "Dependency: {module} requires {dependency} version {version}. Pinned because: {reason}.",
        ],
        "recall_templates": [
            "What technology does {module} use for {purpose}?",
            "What parameters does the {endpoint} API endpoint accept?",
            "What columns does the {table} table have?",
            "What is the current value of {config_key} and what does it affect?",
            "What was the root cause of bug #{bug_id}?",
            "Why is {dependency} pinned to version {version}?",
        ],
        "compare_templates": [
            "Which modules depend on {dependency} and what versions do they need?",
            "List all API endpoints that require {auth_method} authentication.",
            "Which configuration values affect {module}?",
        ],
        "plan_templates": [
            "If we upgrade {dependency} to {new_version}, which modules would be affected?",
            "Propose a refactoring plan for {module} based on all known design decisions.",
            "What would break if we changed {config_key} from {old_value} to {new_value}?",
        ],
        "update_templates": [
            "Design change: {module} now uses {new_technology} instead of {old_technology}.",
            "Schema migration: {table} column {old_column} renamed to {new_column}.",
            "Config update: {config_key} changed from {old_value} to {new_value}.",
        ],
        "categories": ["auth", "database", "API", "frontend", "caching", "logging"],
        "services": ["PostgreSQL", "Redis", "Elasticsearch", "RabbitMQ", "S3", "Cloudflare"],
    },
}


class S5Generator(BaseGenerator, DependencyMixin):
    """Generate task streams for S5 Self-Planning evaluation.

    Produces a mix of task types designed to expose aging mechanisms:
    planning complexity, recall precision, interference, update propagation,
    decision fragility, and planning fatigue.
    """

    SCENARIO_ID = "s5_self_planning"

    def __init__(self, seed: int = 42, domain: str = "assistant", pressure: PressureConfig | None = None):
        super().__init__(seed)
        if domain not in _DOMAINS:
            raise ValueError(f"Unknown domain: {domain}. Choose from {list(_DOMAINS)}")
        self.domain = domain
        self._templates = _DOMAINS[domain]
        self._facts: list[dict] = []  # accumulated facts for recall probes
        self.pressure = pressure or PressureConfig.none()

    def generate(self, n_sessions: int = 10) -> dict[str, Any]:
        """Generate complete task stream.

        Args:
            n_sessions: number of session blocks.

        Returns:
            dict with keys:
                task_stream: {session_length, tasks: [...]}
                recall_probes: {probes: [...]}
                facts_registry: [{id, session, keywords, content}, ...]
                dependency_graph: {...}
                output_dependency_pairs: [{computation_id, producer_block,
                    consumer_block, distance, computed_value, ...}]
        """
        graph = FactGraph()
        session_length = 12
        tasks = []
        facts_registry = []
        recall_probes = []
        binding_probes = []  # forced-choice gold-vs-distractor probes (interference binding)
        fact_counter = 0
        baseline_tasks = []  # tasks from session 0, re-asked later for fatigue measurement
        output_dependency_pairs: list[dict] = []
        deferred_consumers: dict[int, list[dict]] = {}  # block → [consumer_task, ...]

        for block in range(n_sessions):
            block_tasks = []

            # 5-6 new_info tasks
            n_info = self.rng.randint(5, 6)
            for i in range(n_info):
                # Pull fact_counter ahead of any ids the mixin minted via
                # graph._counter so new_info doesn't overwrite a previously
                # registered fact (e.g., a version-update fact from
                # version_random_facts).
                fact_counter = max(fact_counter, graph._counter)
                fact = self._generate_info_fact(fact_counter, block)
                facts_registry.append(fact)

                # Register each fact in the FactGraph (migrate from facts_registry)
                graph.register_fact(
                    session=block,
                    domain=fact.get("domain", self.domain),
                    content=fact["prompt"],
                    keywords=fact["keywords"],
                    fact_id=fact["id"],
                )
                # Keep graph counter in sync with explicit fact_ids
                graph._counter = max(graph._counter, fact_counter + 1)

                block_tasks.append({
                    "id": f"b{block}_info_{i}",
                    "type": "new_info",
                    "session_block": block,
                    "prompt": fact["prompt"],
                    "eval_keywords": fact["keywords"],
                    "fact_id": fact["id"],
                    "domain": fact.get("domain", self.domain),
                })
                # Create recall probe for this fact
                recall_probes.append({
                    "id": f"rp_{fact['id']}",
                    "available_after_block": block,
                    "question": fact["recall_question"],
                    "keywords": fact["keywords"],
                    "source_fact_id": fact["id"],
                })
                fact_counter += 1

            # 3-4 recall_precise tasks (referencing prior facts)
            if block > 0:
                n_recall = min(self.rng.randint(3, 4), len(facts_registry) - n_info)
                for i in range(n_recall):
                    # Sample from prior blocks with varying lag
                    prior_facts = [f for f in facts_registry if f["session_block"] < block]
                    if prior_facts:
                        fact = self.rng.choice(prior_facts)
                        block_tasks.append({
                            "id": f"b{block}_recall_{i}",
                            "type": "recall_precise",
                            "session_block": block,
                            "prompt": fact["recall_question"],
                            "eval_keywords": fact["keywords"],
                            "references_facts": [fact["id"]],
                            "lag": block - fact["session_block"],
                        })

            # 1 update task (if there are facts to update, starting from block 2)
            if block >= 2 and facts_registry:
                # Exclude facts already invalidated by the DependencyMixin's
                # invalidate_random_facts in a prior block — graph.update_fact
                # now raises ValueError on invalidated facts (silent revival
                # was a bug), so we must skip them here.
                def _still_active(f):
                    fid = f.get("id")
                    if fid in graph.facts and graph.facts[fid].invalidated_at is not None:
                        return False
                    return f["session_block"] < block - 1 and not f.get("updated")

                updatable = [f for f in facts_registry if _still_active(f)]
                if updatable:
                    old_fact = self.rng.choice(updatable)
                    update = self._generate_update(old_fact, block)
                    old_fact["updated"] = True
                    block_tasks.append(update)
                    # Let the FactGraph allocate the new id. Hand-rolling
                    # `fact_{fact_counter}` here collides with ids the mixin
                    # already minted via graph._counter (version_random_facts,
                    # inject_interference), which produces dangling
                    # replaces/replaced_by pointers.
                    if old_fact["id"] in graph.facts:
                        new_fact = graph.update_fact(
                            old_id=old_fact["id"],
                            new_content=update["prompt"],
                            new_keywords=update["new_keywords"],
                            session=block,
                        )
                    else:
                        new_fact = graph.register_fact(
                            session=block,
                            domain=self.domain,
                            content=update["prompt"],
                            keywords=update["new_keywords"],
                        )
                    new_fact_id = new_fact.id
                    facts_registry.append({
                        "id": new_fact_id,
                        "session_block": block,
                        "keywords": update["new_keywords"],
                        "recall_question": old_fact["recall_question"],
                        "prompt": update["prompt"],
                        "replaces": old_fact["id"],
                    })
                    # --- Revision sync (faithfulness) ---
                    # Flip the superseded fact's recall probe to the NEW value
                    # from this block onward via keywords_history. Without this
                    # the probe keeps the OLD keywords and recall_accuracy
                    # rewards citing the STALE value (and penalizes a correct
                    # update) — the S5 revision gap. Mirrors S3's keywords_history
                    # and S6's _sync_probes_after_revisions; the runner scores
                    # the active value for the block (old before update, new after).
                    for rp in recall_probes:
                        if rp.get("source_fact_id") == old_fact["id"]:
                            hist = rp.get("keywords_history") or [
                                (rp["available_after_block"], list(rp["keywords"]))
                            ]
                            hist.append((block, list(update["new_keywords"])))
                            rp["keywords_history"] = hist
                            rp["keywords"] = list(update["new_keywords"])
                            break
                    fact_counter = max(fact_counter, graph._counter)

            # 1 cross_reference task (if enough facts)
            if block >= 1 and len(facts_registry) >= 4:
                cross_ref = self._generate_cross_reference(facts_registry, block)
                if cross_ref:
                    block_tasks.append(cross_ref)

            # 1 plan task (if enough context)
            if block >= 2:
                plan_task = self._generate_plan_task(facts_registry, block)
                if plan_task:
                    block_tasks.append(plan_task)

            # 1 interference probe (from block 3+, when similar-category facts exist)
            if block >= 3:
                interference = self._generate_interference_probe(facts_registry, block)
                if interference:
                    block_tasks.append(interference)

            # Apply dependency task replacement after warmup
            if block >= self.pressure.warmup_sessions and self.rng.random() < self.pressure.dependency_density:
                dep_task = self.build_dependency_task(graph, block, self.rng, self.pressure)
                if dep_task:
                    block_tasks.append({
                        "id": f"b{block}_dep_0",
                        "type": "dependency",
                        "session_block": block,
                        "prompt": dep_task["text"],
                        "eval_keywords": dep_task["eval_keywords"],
                        "reference_answer": dep_task["reference_answer"],
                    })

            # Apply version updates
            updates = self.version_random_facts(graph, block, self.rng, self.pressure)
            if updates:
                for ui, u in enumerate(updates):
                    block_tasks.append({
                        "id": f"b{block}_ver_update_{ui}",
                        "type": "update",
                        "session_block": block,
                        "prompt": u["text"],
                        "eval_keywords": u["new_keywords"],
                        "old_keywords": u["old_keywords"],
                        "new_keywords": u["new_keywords"],
                        "replaces_fact": u["old_fact_id"],
                    })

            # Apply selective forgetting (revision aging)
            invalidations = self.invalidate_random_facts(graph, block, self.rng, self.pressure)
            if invalidations:
                for ii, inv in enumerate(invalidations):
                    block_tasks.append({
                        "id": f"b{block}_forget_{ii}",
                        "type": "invalidation",
                        "session_block": block,
                        "prompt": inv["text"],
                        "eval_keywords": [],  # no keywords expected — agent should NOT cite these
                        "invalidated_fact_id": inv["fact_id"],
                        "invalidated_keywords": inv["invalidated_keywords"],
                    })

            # Inject interference facts (confusable cross-domain pairs)
            if block >= self.pressure.confusable_start_session:
                pairs = self.inject_interference(graph, block, self.rng, self.pressure)
                for pi, pair in enumerate(pairs):
                    block_tasks.append({
                        "id": f"b{block}_interf_{pi}",
                        "type": "new_info",
                        "session_block": block,
                        "prompt": f"Please remember: {pair['text_a']} Also: {pair['text_b']}",
                        "eval_keywords": [pair["fact_a"]["value"], pair["fact_b"]["value"]],
                    })
                    # Forced binding probe (gold = fact_a, distractor = fact_b),
                    # emitted by default for EVERY injected pair so interference
                    # is measured under medium pressure (not only the explicit
                    # similar-name / high-similarity modes). The runner re-asks
                    # these from a later block so the agent must recover the gold
                    # value from its own workspace files; citing the distractor
                    # is a binding failure. These do NOT touch recall_accuracy —
                    # they live in a separate `binding_probes` list and feed only
                    # score_interference_binding.
                    gold = pair["fact_a"]["value"]
                    distractor = pair["fact_b"]["value"]
                    question = pair.get("probe_question") or (
                        f"What is the exact {pair['fact_a']['domain']} "
                        f"{pair.get('shared_term', 'value')}? "
                        f"Reply with the exact value only."
                    )
                    binding_probes.append({
                        "probe_id": f"b{block}_binding_{pi}",
                        "available_after_block": block,
                        "question": question,
                        "keywords": [str(gold)],
                        "gold_value": gold,
                        "distractor_value": distractor,
                    })

            # Schedule output dependency producer — tests plan-execution drift
            # across blocks (producer writes a computed value; consumer reads it
            # distance blocks later, testing whether the value survived in workspace files)
            if (block >= self.pressure.warmup_sessions
                    and self.rng.random() < self.pressure.dependency_density
                    and block + 1 < n_sessions):
                max_dist = min(3, n_sessions - 1 - block)
                distance = self.rng.randint(1, max(1, max_dist))
                consumer_block = block + distance
                pair_idx = len(output_dependency_pairs)
                producer, consumer = self._generate_output_dependency_pair(
                    block, consumer_block, pair_idx
                )
                block_tasks.append(producer)
                deferred_consumers.setdefault(consumer_block, []).append(consumer)
                output_dependency_pairs.append({
                    "computation_id": producer["computation_id"],
                    "producer_block": block,
                    "consumer_block": consumer_block,
                    "distance": distance,
                    "computation_type": producer["computation_type"],
                    "computed_value": producer["computed_value"],
                    "producer_task_id": producer["id"],
                    "consumer_task_id": consumer["id"],
                })

            # Save session-0 tasks as baseline for fatigue measurement
            if block == 0:
                baseline_tasks = [t for t in block_tasks if t["type"] in ("recall_precise", "plan")]

            # 1 repeat_baseline task (re-ask a session-0 task at later sessions)
            if block >= 5 and baseline_tasks:
                baseline = self.rng.choice(baseline_tasks)
                block_tasks.append({
                    "id": f"b{block}_fatigue_0",
                    "type": "repeat_baseline",
                    "session_block": block,
                    "prompt": baseline["prompt"],
                    "eval_keywords": baseline.get("eval_keywords", []),
                    "original_block": 0,
                    "original_task_id": baseline["id"],
                })

            # Inject deferred consumers from earlier producers (protected from trim)
            deferred = deferred_consumers.pop(block, [])

            # Shuffle within block (keep new_info and producers first so agent stores them)
            info_tasks = [t for t in block_tasks
                          if t["type"] in ("new_info", "output_dependency_producer")]
            other_tasks = [t for t in block_tasks
                           if t["type"] not in ("new_info", "output_dependency_producer")]
            self.rng.shuffle(other_tasks)
            block_tasks = info_tasks + other_tasks

            # Trim normal tasks to session_length, then append consumers (never trimmed)
            if len(block_tasks) > session_length:
                block_tasks = block_tasks[:session_length]
            block_tasks.extend(deferred)

            tasks.extend(block_tasks)

        result = {
            "task_stream": {
                "description": f"S5 self-planning task stream ({self.domain} domain)",
                "domain": self.domain,
                "session_length": session_length,
                "n_sessions": n_sessions,
                "tasks": tasks,
            },
            "recall_probes": {
                "probes": recall_probes,
            },
            "binding_probes": binding_probes,
            "facts_registry": facts_registry,
        }
        # Interference-binding source for the runner. "topic" reuses the
        # same-category competitor probe (S5's own personal facts) the runner
        # already asks each block — semantically coherent and per-block dense,
        # at no extra agent cost. "generic" (default) keeps the business-pool
        # binding probes for bit-for-bit reproducibility of prior runs.
        result["interference_binding_source"] = (
            "topic" if getattr(self.pressure, "confusable_topic_matched", False)
            else "generic"
        )
        result["dependency_graph"] = graph.export()
        result["output_dependency_pairs"] = output_dependency_pairs
        return result

    def _generate_info_fact(self, fact_id: int, block: int) -> dict:
        """Generate a single new_info fact with recall question and keywords."""
        templates = self._templates["info_templates"]
        recall_templates = self._templates["recall_templates"]
        idx = fact_id % len(templates)

        # Generate values
        person = f"{self.rng.choice(pools.FIRST_NAMES)} {self.rng.choice(pools.LAST_NAMES)}"
        amount = self.rng.choice([127, 234, 389, 456, 578, 612, 743, 891, 1023, 1567])
        date = f"202{self.rng.randint(4,6)}-{self.rng.randint(1,12):02d}-{self.rng.randint(1,28):02d}"
        categories = self._templates["categories"]
        services = self._templates["services"]
        category = self.rng.choice(categories)
        service = self.rng.choice(services)

        # Fill template
        fill = {
            "category": category, "amount": amount, "date": date,
            "person": person, "preference": self.rng.choice(["books", "orchids", "tech gadgets", "handmade crafts", "wine"]),
            "service": service, "provider": self.rng.choice(pools.LAST_NAMES) + " Corp",
            "account": f"{self.rng.randint(1000,9999)}-{self.rng.randint(100,999)}",
            "appointment": self.rng.choice(["dentist", "therapist", "trainer", "mentor meeting"]),
            "schedule": self.rng.choice(["Monday", "Wednesday", "first Friday of the month"]),
            "location": f"{self.rng.randint(100,999)} {self.rng.choice(pools.LAST_NAMES)} Street",
            "allergen": self.rng.choice(["shellfish", "peanuts", "latex", "penicillin"]),
            "doctor": self.rng.choice(pools.LAST_NAMES),
            # coding domain
            "module": self.rng.choice(["auth", "payments", "notifications", "analytics", "search"]),
            "technology": self.rng.choice(["JWT", "OAuth2", "gRPC", "GraphQL", "WebSocket"]),
            "purpose": self.rng.choice(["authentication", "data streaming", "caching", "logging"]),
            "rationale": self.rng.choice(["performance", "security", "simplicity", "team expertise"]),
            "endpoint": f"/api/v{self.rng.randint(1,3)}/{self.rng.choice(['users', 'orders', 'products'])}",
            "params": self.rng.choice(["user_id, page, limit", "query, filters, sort", "start_date, end_date"]),
            "response": self.rng.choice(["JSON array", "paginated object", "stream"]),
            "auth_method": self.rng.choice(["Bearer token", "API key", "OAuth2"]),
            "table": self.rng.choice(["users", "orders", "products", "sessions", "audit_log"]),
            "columns": self.rng.choice(["id, name, email, created_at", "id, user_id, total, status", "id, title, price, category"]),
            "pk": "id", "index": self.rng.choice(["email", "user_id", "created_at"]),
            "config_key": self.rng.choice(["MAX_CONNECTIONS", "CACHE_TTL", "LOG_LEVEL", "RATE_LIMIT"]),
            "config_value": str(self.rng.choice([100, 300, 3600, 50])),
            "affected_modules": self.rng.choice(["database, API", "auth, sessions", "all services"]),
            "bug_id": self.rng.randint(100, 999),
            "description": self.rng.choice(["timeout on large queries", "auth token not refreshing", "race condition in cache"]),
            "root_cause": self.rng.choice(["missing index", "stale token cache", "unlocked shared state"]),
            "dependency": self.rng.choice(["numpy", "fastapi", "sqlalchemy", "redis-py", "pydantic"]),
            "version": f"{self.rng.randint(1,5)}.{self.rng.randint(0,9)}.{self.rng.randint(0,9)}",
            "reason": self.rng.choice(["breaking change in newer version", "security patch", "API stability"]),
            # knowledge_base domain
            "decision": self.rng.choice(["approved Q3 budget", "selected vendor", "delayed launch", "hired contractor"]),
            "milestone": self.rng.choice(["Phase 1 delivery", "Beta launch", "Security audit", "Load testing"]),
            "status": self.rng.choice(["on track", "at risk", "completed", "blocked"]),
            "vendor": self.rng.choice(pools.LAST_NAMES) + " Solutions",
            "project": self.rng.choice(["Project Atlas", "Initiative Horizon", "Platform Nexus"]),
            "old_person": f"{self.rng.choice(pools.FIRST_NAMES)} {self.rng.choice(pools.LAST_NAMES)}",
            "risk": self.rng.choice(["vendor lock-in", "key person dependency", "scope creep", "data migration failure"]),
            "severity": self.rng.choice(["high", "medium", "critical"]),
            "old_amount": self.rng.choice([200, 350, 500, 750]),
            "topic": self.rng.choice(["budget allocation", "vendor selection", "timeline", "staffing", "architecture", "security policy"]),
            "new_person": f"{self.rng.choice(pools.FIRST_NAMES)} {self.rng.choice(pools.LAST_NAMES)}",
            "new_date": pools.random_date(self.rng),
            "old_date": pools.random_date(self.rng),
            "new_version": f"{self.rng.randint(2,6)}.{self.rng.randint(0,9)}.0",
            "new_amount": pools.random_dollar(self.rng, 100, 5000),
        }

        try:
            prompt = templates[idx].format(**fill)
        except (KeyError, IndexError):
            prompt = templates[0].format(**fill)

        try:
            recall_q = recall_templates[idx].format(**fill)
        except (KeyError, IndexError):
            recall_q = recall_templates[0].format(**fill)

        # Extract keywords — the specific values that must be recalled.
        # Collect candidate fill values across all templates, then keep only
        # those that appear in the rendered prompt so the gold matches what
        # the agent actually saw.
        candidate_keys = [
            "amount", "person", "category", "date", "account", "provider",
            "service", "appointment", "schedule", "location", "allergen",
            "doctor", "preference",
            # coding domain
            "module", "technology", "purpose", "endpoint", "auth_method",
            "table", "config_key", "dependency", "version",
            # knowledge_base domain
            "vendor", "project", "milestone", "topic", "new_version",
            "new_amount", "old_amount", "bug_id",
        ]
        prompt_lower = prompt.lower()
        seen: set[str] = set()
        keywords: list[str] = []
        for k in candidate_keys:
            val = fill.get(k)
            if val is None:
                continue
            sval = str(val)
            # Multi-word person/doctor/vendor: use first token only.
            if k in ("person", "doctor", "vendor", "old_person", "new_person") and " " in sval:
                sval = sval.split()[0]
            if not sval or sval.lower() in seen:
                continue
            if sval.lower() in prompt_lower:
                keywords.append(sval)
                seen.add(sval.lower())
        if not keywords:
            keywords = [str(amount)]

        return {
            "id": f"fact_{fact_id}",
            "session_block": block,
            "keywords": keywords,
            "recall_question": recall_q,
            "prompt": f"Please remember this information: {prompt}",
            # Use category as sub-domain for domain diversity (enables COMPARE tasks)
            "domain": category if self.domain != "coding" else fill.get("module", self.domain),
        }

    def _generate_update(self, old_fact: dict, block: int) -> dict:
        """Generate an update task that corrects a previous fact."""
        old_keywords = old_fact["keywords"]
        # Change the numerical keyword
        new_amount = self.rng.choice([129, 237, 391, 458, 582, 619, 747, 893])
        new_keywords = [str(new_amount)] + old_keywords[1:]

        prompt = (
            f"IMPORTANT CORRECTION: The information from '{old_fact['id']}' needs updating. "
            f"The value '{old_keywords[0]}' should be '{new_amount}'. "
            f"Please update your records accordingly."
        )
        # eval_keywords scores the acknowledgement of THIS correction (only
        # the new value can reasonably appear). new_keywords carries the
        # full post-update gold for downstream recall probes.
        return {
            "id": f"b{block}_update_0",
            "type": "update",
            "session_block": block,
            "prompt": prompt,
            "eval_keywords": [str(new_amount)],
            "old_keywords": old_keywords,
            "new_keywords": new_keywords,
            "replaces_fact": old_fact["id"],
        }

    def _generate_cross_reference(self, facts: list[dict], block: int) -> dict | None:
        """Generate a task requiring cross-referencing multiple facts."""
        templates = self._templates.get("compare_templates", [])
        if not templates:
            return None

        categories = list(set(f.get("domain", "") for f in facts if f.get("domain")))
        if len(categories) < 2:
            categories = self._templates["categories"][:2]

        fill = {
            "category1": categories[0] if categories else "A",
            "category2": categories[1] if len(categories) > 1 else "B",
            "person": facts[0]["keywords"][1] if len(facts[0]["keywords"]) > 1 else "someone",
            "date": "today",
            "dependency": "the system",
            "auth_method": "authentication",
            "module": "the main module",
        }

        template = self.rng.choice(templates)
        try:
            prompt = template.format(**fill)
        except KeyError:
            prompt = templates[0].format(**fill)

        return {
            "id": f"b{block}_crossref_0",
            "type": "recall_compare",
            "session_block": block,
            "prompt": prompt,
            "eval_keywords": [fill["category1"], fill["category2"]],
            "references_facts": [f["id"] for f in facts[:4]],
        }

    def _generate_plan_task(self, facts: list[dict], block: int) -> dict | None:
        """Generate a planning task requiring synthesis of accumulated context."""
        templates = self._templates.get("plan_templates", [])
        if not templates:
            return None

        person = "someone"
        for f in facts:
            if len(f.get("keywords", [])) > 1:
                person = f["keywords"][1]
                break

        fill = {
            "person": person,
            "amount": self.rng.choice([500, 1000, 2000]),
            "project": "the current project",
            "dependency": "core framework",
            "new_version": "2.0",
            "module": "main module",
            "config_key": "KEY",
            "old_value": "old",
            "new_value": "new",
        }

        template = self.rng.choice(templates)
        try:
            prompt = template.format(**fill)
        except KeyError:
            prompt = templates[0].format(**fill)

        return {
            "id": f"b{block}_plan_0",
            "type": "plan",
            "session_block": block,
            "prompt": prompt,
            "eval_keywords": [],  # scored qualitatively or by keyword coverage
        }

    def _generate_output_dependency_pair(
        self,
        producer_block: int,
        consumer_block: int,
        pair_index: int,
    ) -> tuple[dict, dict]:
        """Generate a producer/consumer task pair for plan-execution drift measurement.

        The producer presents two numbers and asks the agent to compute a result
        and save it to a specific file.  The consumer (distance blocks later) asks
        the agent to read that file and report the value.

        Drift = producer_score - consumer_score grouped by (consumer_block - producer_block).
        This isolates whether computed outputs survive across block resets in workspace files.
        """
        computation_id = f"odep_b{producer_block}_{pair_index}"
        save_file = f"notes/computed_{computation_id}.md"
        distance = consumer_block - producer_block

        # Use two different amount pools so a + b is always unique-looking
        pool_a = [234, 389, 456, 578, 612, 743, 891, 127, 345, 512]
        pool_b = [167, 223, 301, 445, 534, 689, 178, 267, 390, 481]
        a = self.rng.choice(pool_a)
        b = self.rng.choice(pool_b)

        comp_type = self.rng.choice(["sum", "percentage", "difference"])
        cats = self._templates["categories"]

        if comp_type == "sum":
            computed_value = a + b
            cat_a, cat_b = cats[0], cats[1]
            producer_prompt = (
                f"Please remember: the {cat_a} allocation is ${a} "
                f"and the {cat_b} allocation is ${b}. "
                f"Calculate the combined total (${a} + ${b} = ?) and save "
                f"the exact integer result to {save_file}. "
                f"The file should contain only the number."
            )
            consumer_prompt = (
                f"Read {save_file} and report the exact combined total you "
                f"computed for the {cat_a} and {cat_b} allocations. "
                f"Give me only the number."
            )

        elif comp_type == "percentage":
            computed_value = round(a * 0.15)
            cat = cats[2]
            producer_prompt = (
                f"Please remember: the {cat} budget is ${a}. "
                f"Calculate exactly 15% of this (0.15 × {a} = ?) and save "
                f"the exact integer result to {save_file}. "
                f"The file should contain only the number."
            )
            consumer_prompt = (
                f"Read {save_file} and report the exact 15% target you "
                f"computed for the {cat} budget. "
                f"Give me only the number."
            )

        else:  # difference
            if b >= a:
                a, b = b, a   # ensure positive result
            computed_value = a - b
            cat = cats[3]
            producer_prompt = (
                f"Please remember: the original {cat} budget was ${a}, "
                f"then a ${b} reduction was applied. "
                f"Calculate the remaining budget (${a} - ${b} = ?) and save "
                f"the exact integer result to {save_file}. "
                f"The file should contain only the number."
            )
            consumer_prompt = (
                f"Read {save_file} and report the exact remaining budget you "
                f"computed for the {cat} allocation after the reduction. "
                f"Give me only the number."
            )

        producer_id = f"b{producer_block}_odep_{pair_index}_producer"
        consumer_id = f"b{consumer_block}_odep_{pair_index}_consumer"

        producer = {
            "id": producer_id,
            "type": "output_dependency_producer",
            "session_block": producer_block,
            "computation_id": computation_id,
            "computation_type": comp_type,
            "computed_value": computed_value,
            "save_target": save_file,
            "prompt": producer_prompt,
            "eval_keywords": [str(computed_value)],
        }
        consumer = {
            "id": consumer_id,
            "type": "output_dependency_consumer",
            "session_block": consumer_block,
            "computation_id": computation_id,
            "computation_type": comp_type,
            "computed_value": computed_value,
            "distance": distance,
            "save_target": save_file,
            "prompt": consumer_prompt,
            "eval_keywords": [str(computed_value)],
        }
        return producer, consumer

    def _generate_interference_probe(self, facts: list[dict], block: int) -> dict | None:
        """Generate a probe targeting a specific fact among same-category competitors.

        Finds 2+ facts in the same category, picks one as the target, and asks
        a question that could be answered with any of them. Tests whether the
        agent retrieves the CORRECT one vs a competitor.
        """
        # Group facts by category (first keyword is typically the category or amount)
        from collections import defaultdict
        by_category = defaultdict(list)
        for f in facts:
            if f.get("session_block", 99) >= block:
                continue
            kws = f.get("keywords", [])
            # Use the category keyword (last in the list) for grouping
            cat = kws[-1] if kws else "unknown"
            by_category[cat].append(f)

        # Find a category with 2+ facts (competition exists)
        candidates = [(cat, fs) for cat, fs in by_category.items() if len(fs) >= 2]
        if not candidates:
            return None

        cat, competing_facts = self.rng.choice(candidates)
        target = self.rng.choice(competing_facts)
        competitors = [f for f in competing_facts if f["id"] != target["id"]]

        # Build the probe question — ask specifically about the target
        target_kws = target.get("keywords", [])
        # Use a distinguishing keyword (person name, typically the second keyword)
        person = target_kws[1] if len(target_kws) > 1 else "this entry"
        amount = target_kws[0] if target_kws else "?"

        question = (
            f"What is the specific value or detail associated with {person} "
            f"in the {cat} category? Give me the exact number or value."
        )

        # Competitor keywords (values that should NOT appear if correctly retrieved)
        competitor_kws = []
        for comp in competitors:
            comp_kws = comp.get("keywords", [])
            if comp_kws:
                competitor_kws.append(comp_kws[0])  # the specific value

        return {
            "id": f"b{block}_interference_0",
            "type": "interference",
            "session_block": block,
            "prompt": question,
            "eval_keywords": target_kws[:1],  # the correct value
            "target_keywords": target_kws,
            "competitor_keywords": competitor_kws,
            "target_fact_id": target["id"],
            "competitor_fact_ids": [c["id"] for c in competitors],
            "category": cat,
        }

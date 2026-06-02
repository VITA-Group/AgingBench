"""Programmatic generator for S2 Lifestyle Assistant scenario.

Produces user profiles with arbitrary constraints, lifestyle tasks,
eval probes with violation/anti-violation regex, session facts, and
compounding probes — all matching the curated JSON format.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator
from .dependency_mixin import DependencyMixin
from .fact_graph import FactGraph
from .pools import (
    FIRST_NAMES, LAST_NAMES, RESTAURANT_NAMES, CUISINE_TYPES, FOOD_ITEMS,
    DIETARY_RESTRICTIONS, OCCASIONS, BUDGET_CATEGORIES,
    random_dollar, random_count, random_person, ensure_non_round,
    sample_unique,
)
from .pressure_config import PressureConfig

# ---------------------------------------------------------------------------
# Constraint templates — each produces an arbitrary, non-inferable rule
# ---------------------------------------------------------------------------

_CONSTRAINT_TEMPLATES = [
    # (category, rule_template, keywords_fn, probe_template, viol_patterns, anti_patterns, fragility)
    {
        "category": "dining",
        "fragility": "high",
        "gen": lambda rng: _gen_budget_constraint(rng, "Dining", "dining prepaid card"),
    },
    {
        "category": "shopping",
        "fragility": "high",
        "gen": lambda rng: _gen_boycott_constraint(rng),
    },
    {
        "category": "subscriptions",
        "fragility": "high",
        "gen": lambda rng: _gen_subscription_limit(rng),
    },
    {
        "category": "dietary",
        "fragility": "medium",
        "gen": lambda rng: _gen_dietary_constraint(rng),
    },
    {
        "category": "dining",
        "fragility": "medium",
        "gen": lambda rng: _gen_favorite_restaurant(rng),
    },
    {
        "category": "scheduling",
        "fragility": "medium",
        "gen": lambda rng: _gen_scheduling_constraint(rng),
    },
    {
        "category": "transport",
        "fragility": "medium",
        "gen": lambda rng: _gen_transport_constraint(rng),
    },
    {
        "category": "communication",
        "fragility": "low",
        "gen": lambda rng: _gen_communication_constraint(rng),
    },
    {
        "category": "financial",
        "fragility": "low",
        "gen": lambda rng: _gen_financial_constraint(rng),
    },
    {
        "category": "gifting",
        "fragility": "low",
        "gen": lambda rng: _gen_gifting_constraint(rng),
    },
]


def _gen_budget_constraint(rng, category_label, card_label):
    amount = ensure_non_round(rng.randint(120, 350), rng)
    return {
        "rule": f"{category_label} budget: exactly ${amount}/month maximum. "
                f"This is the amount on the {card_label}. Do not exceed ${amount}.",
        "keywords": [str(amount)],
        "test_value": amount,
        "test_unit": "USD/month",
        "probe": {
            "text": f"I want to try that new place tonight. The bill would be "
                    f"${amount + rng.randint(3, 20)} total. Should I go?",
            "gold": f"Agent should flag that this exceeds the ${amount}/month budget.",
            "violation_patterns": ["go for it|sounds great|enjoy|book|reserve|yes.*go"],
            "anti_patterns": [f"{amount}|budget|exceeds|over.*budget|too much"],
            "precision_targets": [str(amount)],
        },
    }


def _gen_boycott_constraint(rng):
    stores = ["Amazon", "Walmart", "Target", "Best Buy", "Costco", "eBay"]
    boycotted = rng.choice(stores)
    alternatives = [s for s in stores if s != boycotted][:3]
    alt_str = ", ".join(alternatives)
    return {
        "rule": f"Never buy from {boycotted}. Personal boycott. "
                f"Always suggest alternatives: {alt_str}.",
        "keywords": [boycotted, "boycott"],
        "test_value": None,
        "test_unit": None,
        "probe": {
            "text": f"I found a great deal on {boycotted} for $199. Buy it for me.",
            "gold": f"Agent should refuse {boycotted} and suggest {alt_str}.",
            "violation_patterns": [f"ordered|purchased|bought.*{boycotted.lower()}|placed.*order"],
            "anti_patterns": [
                f"{boycotted.lower()}.*boycott|don't.*{boycotted.lower()}|never.*{boycotted.lower()}"
                f"|{'|'.join(a.lower() for a in alternatives)}"
            ],
            # Precision is met if the agent demonstrates constraint knowledge
            # by EITHER naming the policy ("boycott") OR naming the boycotted
            # store while refusing. Sole "boycott" target was too jargon-y:
            # agents that say "I will not buy from Walmart" demonstrate the
            # same knowledge in plain language and were scored 0.
            "precision_targets": ["boycott", boycotted],
        },
    }


def _gen_subscription_limit(rng):
    max_subs = rng.choice([3, 4, 5])
    services = sample_unique(
        ["Spotify", "Netflix", "NYT", "iCloud", "Hulu", "Disney+",
         "HBO Max", "YouTube Premium", "Audible", "Apple Music"],
        max_subs - 1, rng,
    )
    svc_str = ", ".join(services)
    return {
        "rule": f"Maximum {max_subs} active subscriptions at any time. "
                f"Current: {svc_str}. Only 1 slot remaining.",
        "keywords": [str(max_subs)] + services[:2],
        "test_value": max_subs,
        "test_unit": "subscriptions",
        "probe": {
            "text": "I want to subscribe to two new streaming services. Sign me up for both.",
            "gold": f"Agent should warn 2 subs would exceed the {max_subs}-max.",
            "violation_patterns": ["signed up.*both|subscribed.*both|enrolled.*both"],
            "anti_patterns": [f"{max_subs}|maximum|limit|slot|only.*one|{services[0].lower()}"],
            "precision_targets": services[:2],
        },
    }


def _gen_dietary_constraint(rng):
    restriction = rng.choice(DIETARY_RESTRICTIONS[:7])  # allergy types
    return {
        "rule": f"Severe {restriction} (life-threatening, not a preference). "
                f"Always check menus and ingredients.",
        "keywords": [restriction.split()[0], "allergy" if "allergy" in restriction else restriction.split()[-1]],
        "test_value": None,
        "test_unit": None,
        "probe": {
            "text": f"I'm thinking of trying a dish that might contain "
                    f"{restriction.split()[0]}. What do you think?",
            "gold": f"Agent should warn about the {restriction}.",
            "violation_patterns": ["go ahead|sounds.*fine|enjoy|try it|sure"],
            "anti_patterns": [f"{restriction.split()[0]}|allergy|allergic|dangerous|avoid"],
            "precision_targets": [restriction.split()[0]],
        },
    }


def _gen_favorite_restaurant(rng):
    restaurant = rng.choice(RESTAURANT_NAMES)
    address = f"{ensure_non_round(rng.randint(100, 999), rng)} {rng.choice(['Oak', 'Elm', 'Main', 'Pine', 'Maple'])} Street"
    backup = rng.choice([r for r in RESTAURANT_NAMES if r != restaurant])
    return {
        "rule": f"Favorite restaurant is '{restaurant}' at {address}. "
                f"Always suggest it first for dining. Backup: '{backup}'.",
        "keywords": [restaurant, address.split()[0], backup],
        "test_value": None,
        "test_unit": None,
        "probe": {
            "text": "Suggest a good restaurant for dinner tonight.",
            "gold": f"Agent should suggest {restaurant} at {address} first.",
            "violation_patterns": [],
            "anti_patterns": [restaurant.lower().split()[0]],
            "precision_targets": [restaurant],
        },
    }


def _gen_scheduling_constraint(rng):
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    reason = rng.choice(["deep focus", "remote work", "family time", "personal projects"])
    # Other weekdays the agent might reschedule TO (any of these in the
    # response demonstrates that the agent understood the constraint).
    other_days = [d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                  if d != day]
    return {
        "rule": f"No meetings or appointments on {day}s — reserved for {reason}. "
                f"Reschedule anything proposed for {day}.",
        "keywords": [day, reason.split()[0]],
        "test_value": None,
        "test_unit": None,
        "probe": {
            "text": (
                f"Schedule a team meeting for next {day} at 2pm. "
                f"If that doesn't work, tell me which day you scheduled it for instead."
            ),
            "gold": f"Agent should refuse {day} and suggest an alternative day.",
            "violation_patterns": [f"scheduled|booked|confirmed.*{day.lower()}"],
            "anti_patterns": [f"{day.lower()}|reschedule|not available|{reason.split()[0]}"],
            # Precision satisfied if the agent EITHER names the forbidden day
            # ({day}), uses scheduling language showing they recognized the
            # conflict ("rescheduled"), OR names a different weekday they
            # rescheduled TO. Previously only {day} matched, so agents that
            # correctly rescheduled (e.g., "scheduled for Wednesday") and
            # never mentioned the original Thursday scored 0 despite having
            # solved the task perfectly.
            "precision_targets": [day, "rescheduled"] + other_days,
        },
    }


def _gen_transport_constraint(rng):
    preferred, avoided = rng.sample(["Lyft", "Uber", "Via", "Curb"], 2)
    reason = rng.choice(["billing dispute", "bad experience", "privacy concerns"])
    return {
        "rule": f"Always book {preferred}, never {avoided}. "
                f"Had a {reason} with {avoided}.",
        "keywords": [preferred, f"never {avoided}"],
        "test_value": None,
        "test_unit": None,
        "probe": {
            # Force the agent to NAME the service it used. Previously the
            # probe just said "Book me a ride with {avoided}", and an agent
            # could complete the task by silently using {preferred} but
            # respond with "I have completed the task." That vacuous response
            # makes the probe untestable (we can't tell whether the agent
            # knew the preferred service). The added instruction surfaces
            # the agent's choice in the response text.
            "text": (
                f"Book me a ride to the airport tomorrow at 6 AM. "
                f"Tell me which ride service you booked."
            ),
            "gold": f"Agent should refuse {avoided} and use {preferred} instead.",
            "violation_patterns": [f"booked.*{avoided.lower()}|ordered.*{avoided.lower()}"],
            "anti_patterns": [f"{preferred.lower()}|never.*{avoided.lower()}|{reason.split()[0]}"],
            "precision_targets": [preferred],
        },
    }


def _gen_communication_constraint(rng):
    title = rng.choice(["Dr.", "Prof.", "Mx."])
    last = rng.choice(LAST_NAMES)
    return {
        "rule": f"Always address user as '{title} {last}' in formal communications. "
                f"Use first name only in casual contexts.",
        "keywords": [f"{title} {last}"],
        "test_value": None,
        "test_unit": None,
        "probe": {
            # Disambiguate sender vs recipient. Previously this read
            # "Draft a formal email to my colleague", which agents
            # interpreted in conflicting ways — some drafted email TO the
            # user using the formal name as recipient, others addressed it
            # FROM the user with no formal signature. Both directionalities
            # are technically reasonable but produce uninterpretable scoring.
            # The reworded probe forces the formal name into the sender
            # signature, which is the actual constraint test.
            "text": (
                "Draft a formal email from me to my colleague Alex about "
                "the meeting next week. Sign it with my formal name."
            ),
            "gold": f"Agent should use '{title} {last}' as the sender name.",
            "violation_patterns": [],
            "anti_patterns": [f"{title.lower().replace('.', '')}|{last.lower()}"],
            # Accept both "Dr. Volkov" and bare "Volkov" — the surname alone
            # in the signature still demonstrates the constraint is being
            # applied; the title prefix is conventional but not load-bearing.
            "precision_targets": [f"{title} {last}", last],
        },
    }


def _gen_financial_constraint(rng):
    threshold = ensure_non_round(rng.randint(30, 150), rng)
    card_name = rng.choice(["Chase Sapphire", "Amex Gold", "Capital One Venture", "Citi Premier"])
    last4 = rng.randint(1000, 9999)
    return {
        "rule": f"All purchases over ${threshold} must use the {card_name} card "
                f"(ending {last4}) for reward points.",
        "keywords": [card_name.split()[0], str(last4)],
        "test_value": threshold,
        "test_unit": "USD",
        "probe": {
            "text": f"Buy this ${threshold + rng.randint(10, 80)} item with my debit card.",
            "gold": f"Agent should suggest using {card_name} ending {last4} instead.",
            "violation_patterns": ["debit.*purchased|bought.*debit|ordered.*debit"],
            "anti_patterns": [f"{card_name.split()[0].lower()}|{last4}|reward"],
            "precision_targets": [str(last4)],
        },
    }


def _gen_gifting_constraint(rng):
    partner = rng.choice(FIRST_NAMES)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    date_str = f"{'January February March April May June July August September October November December'.split()[month-1]} {day}"
    likes = rng.choice(["hiking gear", "cooking gadgets", "vinyl records", "board games"])
    hates = rng.choice(["lavender scent", "sports memorabilia", "anything pink", "scented candles"])
    return {
        "rule": f"Partner {partner}'s birthday is on {date_str}. "
                f"{partner} loves {likes} and hates {hates}.",
        "keywords": [partner, date_str.split()[0], likes.split()[0], hates.split()[0]],
        "test_value": None,
        "test_unit": None,
        "probe": {
            "text": f"I need a birthday gift idea for {partner}.",
            "gold": f"Agent should suggest {likes}, avoid {hates}.",
            "violation_patterns": [hates.split()[0].lower()],
            "anti_patterns": [likes.split()[0].lower()],
            "precision_targets": [partner, likes.split()[0]],
        },
    }


# ---------------------------------------------------------------------------
# Accumulator probe phrasings — rotated to discourage pattern-matching when
# probes fire every session (dense_accumulator=True). All variants must elicit
# a single numeric answer for the current remaining balance.
# ---------------------------------------------------------------------------

_ACCUMULATOR_PROBE_TEMPLATES = [
    "What is my remaining {category} budget this month? Give me the exact number.",
    "How much do I have left in my {category} budget? Reply with the exact remaining dollar amount.",
    "What's the current balance on my {category} budget? Just the number, please.",
    "Tell me the exact remaining amount in my {category} budget right now.",
    "If I checked my {category} budget today, what number would I see? Exact value only.",
]


# ---------------------------------------------------------------------------
# Task templates per category
# ---------------------------------------------------------------------------

_TASK_TEMPLATES = {
    "dining": [
        "Find a restaurant for {occasion} dinner with {person}. We want {cuisine} food.",
        "Order {food} delivery for lunch today. Something under ${budget}.",
        "Plan a brunch for {count} people this weekend. Budget-friendly please.",
        "What's a good place for a {occasion} celebration? Preferably {cuisine}.",
        "I'm craving {food} tonight. Where should I go?",
    ],
    "shopping": [
        "I need new {item}. Find me something under ${price}.",
        "There's a sale on {item} at {store}. Should I buy?",
        "Order a {item} for me online. Best deal you can find.",
        "Compare prices for {item} across different stores.",
    ],
    "subscriptions": [
        "I found a {service} subscription for ${price}/month. Should I sign up?",
        "What subscriptions am I currently paying for? Should I cancel any?",
        "There's a free trial for {service}. Sign me up.",
    ],
    "scheduling": [
        "Schedule a meeting for next {day} at {time}.",
        "Book a {activity} class for {day} morning.",
        "Set up a weekly recurring event on {day}s.",
    ],
    "transport": [
        "Book me a ride to the airport for tomorrow at {time}.",
        "I need a car to {destination}. What's the best option?",
    ],
    "financial": [
        "Buy this ${price} item. Use whatever card is fine.",
        "Set up auto-pay for my ${price}/month phone bill.",
    ],
    "privacy": [
        "This app wants access to my location and contacts. Should I allow it?",
        "A store wants to send me marketing emails. Sign me up.",
    ],
    "communication": [
        "Draft a formal email to my colleague about the project update.",
        "Send a message to {person} about the meeting tomorrow.",
    ],
    "gifting": [
        "I need a gift for {person}'s birthday. Any ideas?",
        "Find a nice {category} gift under ${price}.",
    ],
}


class S2Generator(BaseGenerator, DependencyMixin):
    """Generate S2 lifestyle assistant scenario data."""

    SCENARIO_ID = "s2_lifestyle_assistant"

    def __init__(
        self,
        seed: int = 42,
        pressure: PressureConfig | None = None,
        dense_accumulator: bool = False,
    ):
        super().__init__(seed)
        self.pressure = pressure or PressureConfig.none()
        self.dense_accumulator = dense_accumulator

    def generate(self, n_sessions: int = 10) -> dict[str, Any]:
        graph = FactGraph()

        # Generate user profile with 10 constraints
        profile = self._generate_profile()
        constraints = profile["constraints"]

        # Register constraints in the FactGraph
        for c in constraints:
            graph.register_fact(
                session=0,
                domain=c["category"],
                content=c["rule"],
                keywords=c["keywords"],
            )

        # Generate session tasks
        sessions = self._generate_sessions(n_sessions, constraints)

        # Generate eval probes (one per constraint)
        eval_probes = self._generate_eval_probes(constraints)

        # Generate constraint updates (1-2 mid-lifetime)
        updates = self._generate_constraint_updates(n_sessions, constraints)

        # Refresh precision_targets for "relax" updates that change a constraint's
        # numeric value. Without this, an agent that learns the new value (e.g.
        # $400) is penalized because the eval probe's frozen target is still the
        # OLD value (e.g. $309) — the metric perversely rewards staleness. The
        # `precision_target_change` field records the session at which the new
        # target becomes active so the scorer can pick the correct gold per session.
        for upd in updates:
            if upd.get("type") != "relax":
                continue
            new_kws = upd.get("keywords_added") or []
            if not new_kws:
                continue
            for probe in eval_probes:
                if probe["constraint_id"] == upd["constraint_id"]:
                    probe["precision_target_change"] = {
                        "session": upd["session"],
                        "new_targets": new_kws,
                    }
                    break

        # Generate session facts (1 per session)
        facts = self._generate_session_facts(n_sessions, sessions)

        # Register session facts in the FactGraph
        for f in facts:
            graph.register_fact(
                session=f["session"],
                domain=f["category"],
                content=f["text"],
                keywords=f["recall_keywords"],
            )

        # Generate compounding probes
        compounding = self._generate_compounding_probes(facts)

        # Apply dependency tasks and version updates per session
        for t in range(n_sessions):
            if t >= self.pressure.warmup_sessions and self.rng.random() < self.pressure.dependency_density:
                dep_task = self.build_dependency_task(graph, t, self.rng, self.pressure)
                if dep_task and t < len(sessions):
                    sessions[t]["tasks"].append({
                        "id": f"s{t}_dep",
                        "text": dep_task["text"],
                        "constraints_tested": [],
                        "category": "dependency",
                    })

            updates_for_session = self.version_random_facts(graph, t, self.rng, self.pressure)
            if updates_for_session and t < len(sessions):
                for u in updates_for_session:
                    sessions[t]["tasks"].append({
                        "id": f"s{t}_update",
                        "text": u["text"],
                        "constraints_tested": [],
                        "category": "update",
                    })

            # Apply selective forgetting (revision aging)
            invalidations = self.invalidate_random_facts(graph, t, self.rng, self.pressure)
            if invalidations and t < len(sessions):
                for inv in invalidations:
                    sessions[t]["tasks"].append({
                        "id": f"s{t}_forget",
                        "text": inv["text"],
                        "constraints_tested": [],
                        "category": "invalidation",
                    })

            # Inject interference facts (confusable cross-domain pairs)
            if t >= self.pressure.confusable_start_session and t < len(sessions):
                pairs = self.inject_interference(graph, t, self.rng, self.pressure)
                for pair in pairs:
                    sessions[t]["tasks"].append({
                        "id": f"s{t}_interf",
                        "text": f"{pair['text_a']} {pair['text_b']}",
                        "constraints_tested": [],
                        "category": "interference",
                    })

        # Generate accumulator track (Ledger-QA pattern for revision aging).
        # NB: this MUTATES the chosen budget constraint's rule + _probe_data
        # when the initial value needs scaling to keep the ledger positive
        # over long horizons (n_sessions=10+). The mutation flips e.g. $309
        # → $579 in the constraint, but eval_probes were already snapshotted
        # above with the OLD value, so the post-mutation re-sync below is
        # essential — without it, the eval probe targets a value the agent
        # never sees in memory (since memory carries the mutated $579 rule).
        accumulator_probes = self._generate_accumulator_track(
            n_sessions, constraints, sessions, graph
        )

        # Re-sync eval_probes against any post-mutation constraint state.
        # _generate_accumulator_track may have rewritten budget_constraint's
        # rule, precision_targets, anti_patterns, etc. Pull the latest values
        # back into the matching eval_probe so the scorer's gold matches the
        # constraint the agent actually reads.
        for ep in eval_probes:
            cid = ep["constraint_id"]
            c = next((c for c in constraints if c["id"] == cid), None)
            if c is None:
                continue
            pd = c.get("_probe_data") or {}
            # Resync the fields the accumulator-track mutation touches.
            ep["constraint_rule"] = c["rule"][:80]
            if isinstance(pd.get("precision_targets"), list):
                ep["precision_targets"] = list(pd["precision_targets"])
            if isinstance(pd.get("anti_patterns"), list):
                ep["violation_anti_patterns"] = list(pd["anti_patterns"])
            if isinstance(pd.get("text"), str):
                ep["text"] = pd["text"]
            if isinstance(pd.get("gold"), str):
                ep["gold_answer"] = pd["gold"]

        # Inject accumulator probes into session tasks so the runner executes them
        for probe in accumulator_probes:
            t = probe["session"]
            if t < len(sessions):
                sessions[t]["tasks"].append({
                    "id": f"s{t}_acc_probe",
                    "text": probe["question"],
                    "constraints_tested": [],
                    "category": "accumulator_probe",
                    "gold_value": probe["gold_value"],
                    "eval_keywords": probe["eval_keywords"],
                })

        result = {
            "source_profile": profile,
            "session_tasks": {"description": f"{n_sessions} sessions generated", "sessions": sessions},
            "eval_probes": {"description": "Generated eval probes", "probes": eval_probes},
            "constraint_updates": {"description": "Generated updates", "updates": updates},
            "session_facts": {"description": "Generated facts", "facts": facts},
            "compounding_probes": {"description": "Generated compounding", "probes": compounding},
            "accumulator_probes": accumulator_probes,
        }
        result["dependency_graph"] = graph.export()
        return result

    def _generate_profile(self) -> dict:
        first = self.rng.choice(FIRST_NAMES)
        last = self.rng.choice(LAST_NAMES)
        user_name = f"{first} {last}"

        constraints = []
        for i, tmpl in enumerate(_CONSTRAINT_TEMPLATES):
            c_data = tmpl["gen"](self.rng)
            constraints.append({
                "id": f"C{i+1}",
                "category": tmpl["category"],
                "fragility": tmpl["fragility"],
                "rule": c_data["rule"],
                "keywords": c_data["keywords"],
                "test_value": c_data.get("test_value"),
                "test_unit": c_data.get("test_unit"),
                "_probe_data": c_data["probe"],  # internal, used for eval_probes
            })

        # Build profile_text from constraints
        lines = [f"User Profile — {user_name}\n"]
        for c in constraints:
            lines.append(f"- [{c['id']}] {c['rule']}")
        profile_text = "\n".join(lines)

        return {
            "user_name": user_name,
            "profile_version": "generated",
            "constraints": constraints,
            "profile_text": profile_text,
        }

    def _generate_sessions(self, n: int, constraints: list) -> list[dict]:
        # sorted(), not list(set()): set iteration order over strings is
        # PYTHONHASHSEED-dependent, which made the same seed produce different
        # scenarios across processes (non-reproducible). Sorting fixes the order
        # so a given seed is deterministic.
        categories = sorted(set(c["category"] for c in constraints))
        sessions = []
        for t in range(n):
            tasks = []
            # Pick 5 categories for this session's tasks
            session_cats = [self.rng.choice(categories) for _ in range(5)]
            for j, cat in enumerate(session_cats):
                templates = _TASK_TEMPLATES.get(cat, _TASK_TEMPLATES["dining"])
                template = self.rng.choice(templates)
                # Fill template slots
                text = template.format(
                    occasion=self.rng.choice(OCCASIONS),
                    person=self.rng.choice(FIRST_NAMES),
                    cuisine=self.rng.choice(CUISINE_TYPES),
                    food=self.rng.choice(FOOD_ITEMS),
                    budget=ensure_non_round(self.rng.randint(20, 200), self.rng),
                    count=self.rng.randint(3, 8),
                    item=self.rng.choice(["running shoes", "headphones", "jacket", "backpack", "laptop stand"]),
                    price=ensure_non_round(self.rng.randint(30, 300), self.rng),
                    store=self.rng.choice(["Nordstrom", "REI", "Best Buy", "Target"]),
                    service=self.rng.choice(["Netflix", "Hulu", "Disney+", "HBO Max"]),
                    day=self.rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
                    time=f"{self.rng.randint(6, 18)}:{self.rng.choice(['00', '30'])}",
                    activity=self.rng.choice(["yoga", "spinning", "boxing", "pilates"]),
                    destination=self.rng.choice(["downtown", "the airport", "the office"]),
                    category=self.rng.choice(["tech", "outdoor", "kitchen"]),
                )
                # Find which constraints this category matches
                tested = [c["id"] for c in constraints if c["category"] == cat]
                tasks.append({
                    "id": f"s{t}_t{j+1}",
                    "text": text,
                    "constraints_tested": tested[:3],
                    "category": cat,
                })
            sessions.append({"session": t, "tasks": tasks})
        return sessions

    def _generate_eval_probes(self, constraints: list) -> list[dict]:
        probes = []
        for c in constraints:
            pd = c.get("_probe_data", {})
            probes.append({
                "id": f"eval_{c['id']}",
                "constraint_id": c["id"],
                "constraint_rule": c["rule"][:80],
                "text": pd.get("text", f"Test probe for {c['id']}"),
                "gold_answer": pd.get("gold", ""),
                "violation_patterns": pd.get("violation_patterns", []),
                "violation_anti_patterns": pd.get("anti_patterns", []),
                "precision_targets": pd.get("precision_targets", c["keywords"][:2]),
            })
        return probes

    def _generate_constraint_updates(self, n: int, constraints: list) -> list[dict]:
        updates = []
        if n < 4:
            return updates
        # Update 1: strengthen a medium-fragility constraint at ~session n//3
        medium = [c for c in constraints if c["fragility"] == "medium"]
        if medium:
            c = self.rng.choice(medium)
            session = max(2, n // 3)
            updates.append({
                "session": session,
                "constraint_id": c["id"],
                "type": "strengthen",
                "old_rule": c["rule"],
                "new_rule": c["rule"] + " (STRICTER enforcement required)",
                "update_text": f"Update: my {c['category']} constraint is now stricter.",
                "keywords_added": ["stricter"],
            })
        # Update 2: relax a high-fragility constraint at ~2*n//3
        high = [c for c in constraints if c["fragility"] == "high" and c.get("test_value")]
        if high and n >= 6:
            c = self.rng.choice(high)
            session = max(4, 2 * n // 3)
            old_val = c["test_value"]
            new_val = old_val + self.rng.randint(20, 60)
            updates.append({
                "session": session,
                "constraint_id": c["id"],
                "type": "relax",
                "old_rule": c["rule"],
                "new_rule": c["rule"].replace(str(old_val), str(new_val)),
                "update_text": f"Good news! My {c['category']} limit increased to ${new_val}.",
                "keywords_added": [str(new_val)],
            })
        return updates

    def _generate_session_facts(self, n: int, sessions: list) -> list[dict]:
        facts = []
        # Event-fact text embeds {cat} so the recall_question's
        # "{cat} activity" matches a token present in the fact.
        fact_templates = [
            ("preference", "The user mentioned their favorite {cat} spot is {entity}.",
             "What is the user's favorite {cat} spot?"),
            ("event", "The user {action} costing ${amount} for {cat}.",
             "What recent {cat} activity did the user do?"),
            ("preference", "The user said they prefer {entity} for {cat}.",
             "What does the user prefer for {cat}?"),
        ]
        for t in range(n):
            tmpl_type, tmpl_text, tmpl_q = self.rng.choice(fact_templates)
            entity = self.rng.choice(RESTAURANT_NAMES + list(FIRST_NAMES))
            cat = self.rng.choice(BUDGET_CATEGORIES[:6])
            amount = ensure_non_round(self.rng.randint(20, 300), self.rng)
            action = self.rng.choice(["bought something", "booked a service", "made a purchase"])
            text = tmpl_text.format(cat=cat, entity=entity, action=action, amount=amount)
            question = tmpl_q.format(cat=cat)
            # Two keywords (entity/amount + category) so paraphrased recall
            # that captures either still scores via the
            # "hits >= max(1, n//2)" rule in score_recall.
            if tmpl_type == "preference":
                kws = [entity.lower().split()[0], cat.lower()]
            else:
                kws = [str(amount), cat.lower()]
            facts.append({
                "session": t,
                "id": f"F{t}",
                "category": tmpl_type,
                "text": text,
                "recall_question": question,
                "recall_keywords": kws,
                "recall_anti_keywords": [],
            })
        return facts

    def _generate_accumulator_track(
        self,
        n_sessions: int,
        constraints: list[dict],
        sessions: list[dict],
        graph: "FactGraph",
    ) -> list[dict]:
        """Generate a budget accumulator with per-session deltas (Ledger-QA pattern).

        Picks a budget constraint, registers an accumulator in the FactGraph,
        injects delta events into session tasks, and creates interspersed probes.
        Returns list of accumulator probe dicts.
        """
        # Find a budget constraint (has a dollar amount keyword)
        budget_constraint = None
        for c in constraints:
            if c["category"] in ("dining", "financial", "subscriptions"):
                try:
                    int(c["keywords"][0])
                    budget_constraint = c
                    break
                except (ValueError, IndexError):
                    continue
        if budget_constraint is None:
            return []

        original_initial = int(budget_constraint["keywords"][0])
        category = budget_constraint["category"]
        acc_name = f"{category}_budget"

        # Scale the initial budget so the ledger stays positive across the
        # horizon. Per-session net ≈ -$35; per-session σ ≈ 34. We scale when
        # the worst-case (mean drift + 2σ noise) would breach zero.
        per_session_drain = 35
        expected_drain = per_session_drain * max(0, n_sessions - 1)
        two_sigma = int(34 * (n_sessions ** 0.5))
        if expected_drain + two_sigma > original_initial:
            initial = int(expected_drain * 1.5) + two_sigma
        else:
            initial = original_initial

        # Keep the source constraint coherent with the scaled accumulator: a
        # "remaining $X budget?" probe is meaningless if the constraint cap and
        # the accumulator's starting balance disagree.
        if initial != original_initial:
            old_s, new_s = str(original_initial), str(initial)
            budget_constraint["keywords"] = [new_s] + budget_constraint["keywords"][1:]
            budget_constraint["test_value"] = initial
            budget_constraint["rule"] = budget_constraint["rule"].replace(
                f"${old_s}", f"${new_s}"
            )
            pd = budget_constraint.get("_probe_data") or {}
            for k in ("text", "gold"):
                if isinstance(pd.get(k), str):
                    pd[k] = pd[k].replace(f"${old_s}", f"${new_s}")
            if isinstance(pd.get("anti_patterns"), list):
                pd["anti_patterns"] = [p.replace(old_s, new_s) for p in pd["anti_patterns"]]
            if isinstance(pd.get("precision_targets"), list):
                pd["precision_targets"] = [t.replace(old_s, new_s) for t in pd["precision_targets"]]

        graph.register_accumulator(acc_name, float(initial), session=0, domain=category)

        # Generate delta events: 1 per session starting from session 1
        delta_descriptions = [
            "spent ${amount} at {place}",
            "paid ${amount} for {place} delivery",
            "received ${amount} {place} credit",
            "subscription charge ${amount} from {place}",
        ]
        places = ["La Bella Notte", "Tokyo Sushi", "Corner Bakery", "Fresh Market",
                   "Cloud Kitchen", "Green Bowl", "Pizza Palace", "Café Metro"]

        probes = []
        for t in range(1, n_sessions):
            # 80% spend, 20% credit
            if self.rng.random() < 0.8:
                amount = -self.rng.randint(20, min(80, max(20, initial // 4)))
            else:
                amount = self.rng.randint(10, 40)

            template = self.rng.choice(delta_descriptions)
            place = self.rng.choice(places)
            desc = template.format(amount=abs(amount), place=place)
            graph.add_delta(acc_name, float(amount), session=t, description=desc)

            # Inject delta as task text into the session
            if t < len(sessions):
                if amount < 0:
                    delta_text = f"Note: you {desc}. This comes from your {category} budget."
                else:
                    delta_text = f"Note: you {desc}. This adds back to your {category} budget."
                sessions[t]["tasks"].append({
                    "id": f"s{t}_delta",
                    "text": delta_text,
                    "constraints_tested": [],
                    "category": "accumulator_delta",
                })

            # Probe schedule. Gold reflects state at the end of session t
            # (delta_t already applied); probe is placed at session t+1 so the
            # agent reads memory written + compressed at session t's boundary
            # (avoids within-session snapshot lag).
            #   default: every 2-3 sessions (sparse, ~5 probes per 12-session run)
            #   dense_accumulator=True: every session ≥ 2 (dense, ~10 probes,
            #     for smooth per-session accumulator_error curves)
            if self.dense_accumulator:
                fires = t >= 2 and t + 1 < n_sessions
            else:
                fires = t >= 2 and t % self.rng.randint(2, 3) == 0 and t + 1 < n_sessions
            if fires:
                gold = graph.get_accumulator_value(acc_name, at_session=t)
                question = self.rng.choice(_ACCUMULATOR_PROBE_TEMPLATES).format(
                    category=category
                )
                probes.append({
                    "session": t + 1,
                    "gold_at_session": t,
                    "question": question,
                    "gold_value": gold,
                    "accumulator": acc_name,
                    "eval_keywords": [str(int(gold))],
                })

        return probes

    def _generate_compounding_probes(self, facts: list) -> list[dict]:
        """
        Compounding probes: each tests multi-session context synthesis by
        requiring recall of facts from two distinct prior sessions.

        Two parallel probe schedules are emitted so downstream scorers can
        report both a gradual decay curve and the legacy cliff signal:

          - `comp_sparse_<t>` (legacy): one probe per 3 sessions, targets
            facts from t-2 and t-1. Cumulatively re-scored each session.
            Produces the binary cliff documented in earlier paper runs.

          - `comp_fresh_<t>` (new): one probe per session t>=2, targets
            facts from t-2 and t-1 (same dependency structure as legacy).
            Scored ONLY at its cohort session so per-session accuracy
            tracks rate-of-decay, not cumulative failure.
        """
        probes: list[dict] = []

        # Legacy sparse schedule (preserves comparability with old runs).
        for t in range(3, len(facts), 3):
            deps = [facts[t - 2], facts[t - 1]]
            dep_ids = [d["id"] for d in deps]
            probes.append({
                "id": f"comp_sparse_{t}",
                "available_from_session": t,
                "cohort_session": t,
                "schedule": "sparse",
                "text": "Considering our past interactions, what do you remember about "
                        "the user's recent activities from a few sessions ago?",
                "dependencies": dep_ids,
                "dependency_description": f"Must remember facts from {', '.join(dep_ids)}.",
                "scoring": {
                    "required_keywords": [d["recall_keywords"] for d in deps],
                    "fail_if_missing_any": True,
                },
                "gold_answer": "; ".join(d["text"] for d in deps),
            })

        # Dense per-session schedule (new; gives compounding decay rate).
        for t in range(2, len(facts)):
            deps = [facts[t - 2], facts[t - 1]]
            dep_ids = [d["id"] for d in deps]
            probes.append({
                "id": f"comp_fresh_{t}",
                "available_from_session": t,
                "cohort_session": t,
                "schedule": "fresh",
                "text": "Considering our past interactions, what do you remember about "
                        "the user's recent activities from a few sessions ago?",
                "dependencies": dep_ids,
                "dependency_description": f"Must remember facts from {', '.join(dep_ids)}.",
                "scoring": {
                    "required_keywords": [d["recall_keywords"] for d in deps],
                    "fail_if_missing_any": True,
                },
                "gold_answer": "; ".join(d["text"] for d in deps),
            })
        return probes

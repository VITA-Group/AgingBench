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
            "precision_targets": ["boycott"],
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
    return {
        "rule": f"No meetings or appointments on {day}s — reserved for {reason}. "
                f"Reschedule anything proposed for {day}.",
        "keywords": [day, reason.split()[0]],
        "test_value": None,
        "test_unit": None,
        "probe": {
            "text": f"Schedule a team meeting for next {day} at 2pm.",
            "gold": f"Agent should refuse {day} and suggest an alternative day.",
            "violation_patterns": [f"scheduled|booked|confirmed.*{day.lower()}"],
            "anti_patterns": [f"{day.lower()}|reschedule|not available|{reason.split()[0]}"],
            "precision_targets": [day],
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
            "text": f"Book me a ride with {avoided} to the airport.",
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
            "text": "Draft a formal email to my colleague about the meeting next week.",
            "gold": f"Agent should use '{title} {last}' as the sender name.",
            "violation_patterns": [],
            "anti_patterns": [f"{title.lower().replace('.', '')}|{last.lower()}"],
            "precision_targets": [f"{title} {last}"],
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

    def __init__(self, seed: int = 42, pressure: PressureConfig | None = None):
        super().__init__(seed)
        self.pressure = pressure or PressureConfig.none()

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

        # Generate accumulator track (Ledger-QA pattern for revision aging)
        accumulator_probes = self._generate_accumulator_track(
            n_sessions, constraints, sessions, graph
        )

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
        categories = list(set(c["category"] for c in constraints))
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
        # Event-fact text now embeds the {cat} token so the recall_question's
        # "{cat} activity" reference points at content actually present in the
        # fact. Pre-fix the recall_question's category was sampled independently
        # from the text, so a fact like "user booked a service costing $42"
        # was paired with "What recent groceries activity?" — even with
        # perfect memory the agent had no semantic anchor and the metric
        # could not score the event facts above zero.
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
            # Multi-keyword gold: entity/amount + category. Aligns with the
            # curated session_facts.json which already uses 2-3 keywords per
            # fact. score_recall's "hits >= max(1, n//2)" rule means at 2
            # keywords either token can satisfy recall — so paraphrased
            # responses that capture the category (or the value) but not both
            # still register, giving lag_recall a graded signal instead of
            # the all-or-nothing single-token collapse.
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

        initial = int(budget_constraint["keywords"][0])
        category = budget_constraint["category"]
        acc_name = f"{category}_budget"
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

            # Generate probe every 2-3 sessions. Gold reflects state at the
            # end of session t (delta_t already applied), but the probe is
            # placed at session t+1 so the agent reads memory that has been
            # written + compressed at session t's boundary. Without this
            # +1 offset, the probe in session t can never see the delta from
            # the same session (within-session memory snapshot lag), giving
            # a systematic ~$20-80 error baked into every affected probe.
            if t >= 2 and t % self.rng.randint(2, 3) == 0 and t + 1 < n_sessions:
                gold = graph.get_accumulator_value(acc_name, at_session=t)
                probes.append({
                    "session": t + 1,   # moved from t to t+1 (Option A fix)
                    "gold_at_session": t,   # records the state the gold corresponds to
                    "question": f"What is my remaining {category} budget this month? "
                                f"Give me the exact number.",
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

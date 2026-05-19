"""Programmatic generator for S3 Knowledge Base scenario.

Produces meeting transcripts, gold timeline decisions, and team queries
— all matching the curated JSON format.
"""

from __future__ import annotations

from typing import Any

from .base import BaseGenerator
from .dependency_mixin import DependencyMixin
from .fact_graph import FactGraph
from .pools import (
    FIRST_NAMES, LAST_NAMES, COMPANY_NAMES, TECH_FRAMEWORKS,
    PROJECT_COMPONENTS, PROJECT_MILESTONES,
    random_dollar, random_count, random_date, random_percent,
    random_latency_ms, ensure_non_round, random_person, sample_unique,
)
from .pressure_config import PressureConfig

# Decision category templates
_DECISION_TEMPLATES = {
    "budget": [
        ("{phase} budget is ${amount}",
         ["amount"], "{person} (Finance)"),
        ("Contingency reserve set at ${amount}",
         ["amount"], "{person} (Finance)"),
        ("{phase} spending: ${spent} of ${total} used ({pct})",
         ["spent", "pct"], "{person} (Finance)"),
    ],
    "tech": [
        ("{framework} selected as {role}",
         ["framework"], "{person} (Tech Lead)"),
        ("{component} connection pool limit set to {count} concurrent",
         ["count"], "{person} (DBA)"),
        ("Rate limiting: {count_a} req/min standard, {count_b} req/min premium",
         ["count_a", "count_b"], "{person} (API Lead)"),
    ],
    "vendor": [
        ("{vendor} is the {service} provider at ${amount}/month",
         ["vendor", "amount"], "{person} (PM)"),
        ("Contract with {vendor} renewed for {months} months at ${amount}/month",
         ["vendor", "amount", "months"], "{person} (PM)"),
    ],
    "timeline": [
        ("{milestone} target is {date}",
         ["milestone", "date"], "{person} (PM)"),
        ("{milestone} moved from {old_date} to {new_date}",
         ["new_date"], "{person} (PM)"),
    ],
    "security": [
        ("{protocol} required for all {scope}",
         ["protocol"], "{person} (Security)"),
        ("Session tokens expire after {minutes} minutes of inactivity",
         ["minutes"], "{person} (Security)"),
    ],
    "hiring": [
        ("{count} {role} engineers hired at ${amount}/month combined",
         ["count", "amount", "role"], "{person} (PM)"),
    ],
    "infra": [
        ("{platform} deployment on {count} nodes, {vcpu} vCPUs each",
         ["count", "vcpu"], "{person} (DevOps)"),
    ],
}

_CATEGORIES = list(_DECISION_TEMPLATES.keys())


class S3Generator(BaseGenerator, DependencyMixin):
    """Generate S3 knowledge base scenario data."""

    SCENARIO_ID = "s3_knowledge_base"

    def __init__(self, seed: int = 42, pressure: PressureConfig | None = None):
        super().__init__(seed)
        self.pressure = pressure or PressureConfig.none()

    def generate(self, n_sessions: int = 12) -> dict[str, Any]:
        graph = FactGraph()
        # Build team roster
        team = [random_person(self.rng) for _ in range(8)]
        project_name = f"Project {self.rng.choice(['Nexus', 'Atlas', 'Horizon', 'Beacon', 'Catalyst'])}"

        all_decisions = []
        all_transcripts = []
        all_queries = []
        decision_idx = 0

        for t in range(n_sessions):
            # 3-5 decisions per session
            n_decisions = self.rng.randint(3, 5)
            session_decisions = []
            cats = sample_unique(_CATEGORIES, min(n_decisions, len(_CATEGORIES)), self.rng)
            # pad if needed
            while len(cats) < n_decisions:
                cats.append(self.rng.choice(_CATEGORIES))

            for cat in cats:
                templates = _DECISION_TEMPLATES[cat]
                tmpl_text, kw_fields, role_tmpl = self.rng.choice(templates)
                person = self.rng.choice(team)

                # Generate values for template slots
                vals = self._gen_values(cat)
                fact_text = tmpl_text.format(**vals)
                keywords = self._extract_keywords(vals, kw_fields)

                decision = {
                    "id": f"D{decision_idx + 1:02d}",
                    "session": t,
                    "category": cat,
                    "fact": fact_text,
                    "keywords": keywords,
                    "person": person,
                }
                session_decisions.append(decision)
                all_decisions.append(decision)
                decision_idx += 1

                # Register each decision in the FactGraph
                graph.register_fact(
                    session=t,
                    domain=cat,
                    content=fact_text,
                    keywords=keywords,
                )

            # Generate transcript
            transcript = self._generate_transcript(t, session_decisions, team, project_name)
            all_transcripts.append(transcript)

            # Generate queries (3 per session: 1-2 from this session + 1 from earlier)
            session_queries = self._generate_queries(t, session_decisions, all_decisions)

            # Apply dependency task replacement after warmup
            if t >= self.pressure.warmup_sessions and self.rng.random() < self.pressure.dependency_density:
                dep_task = self.build_dependency_task(graph, t, self.rng, self.pressure)
                if dep_task:
                    session_queries.append({
                        "query_id": f"q{t}_dep",
                        "question": dep_task["text"],
                        "gold_decision_ids": dep_task.get("dependency_meta", {}).get("depends_on", []),
                        "keywords": dep_task["eval_keywords"],
                    })

            # Apply version updates
            updates = self.version_random_facts(graph, t, self.rng, self.pressure)
            if updates:
                update_lines = [u["text"] for u in updates]
                transcript["transcript"] += "\n\n" + "\n".join(update_lines)

            # Apply selective forgetting (revision aging)
            invalidations = self.invalidate_random_facts(graph, t, self.rng, self.pressure)
            if invalidations:
                inv_lines = [inv["text"] for inv in invalidations]
                transcript["transcript"] += "\n\n" + "\n".join(inv_lines)

            # Inject interference facts (confusable cross-domain pairs)
            if t >= self.pressure.confusable_start_session:
                pairs = self.inject_interference(graph, t, self.rng, self.pressure)
                if pairs:
                    interf_lines = [f"{p['text_a']} {p['text_b']}" for p in pairs]
                    transcript["transcript"] += "\n\nNote: " + " ".join(interf_lines)

            all_queries.append({"session": t, "queries": session_queries})

        result = {
            "transcripts": {"sessions": all_transcripts},
            "gold_timeline": {"decisions": all_decisions},
            "queries": {"sessions": all_queries},
        }
        result["dependency_graph"] = graph.export()
        return result

    def _gen_values(self, cat: str) -> dict:
        """Generate random values for a decision template."""
        return {
            "amount": f"{random_dollar(self.rng, 5000, 500000):,}",
            "spent": f"{random_dollar(self.rng, 10000, 200000):,}",
            "total": f"{random_dollar(self.rng, 100000, 500000):,}",
            "pct": random_percent(self.rng, 20, 80),
            "framework": self.rng.choice(TECH_FRAMEWORKS),
            "role": self.rng.choice(["backend framework", "frontend framework", "data store", "cache layer"]),
            "component": self.rng.choice(PROJECT_COMPONENTS),
            "count": str(random_count(self.rng, 2, 100)),
            "count_a": str(random_count(self.rng, 100, 2000)),
            "count_b": str(random_count(self.rng, 2000, 10000)),
            "vendor": self.rng.choice(COMPANY_NAMES),
            "service": self.rng.choice(["cloud hosting", "monitoring", "CI/CD", "security scanning"]),
            "months": str(self.rng.choice([6, 12, 18, 24])),
            "milestone": self.rng.choice(PROJECT_MILESTONES),
            "date": random_date(self.rng),
            "old_date": random_date(self.rng),
            "new_date": random_date(self.rng),
            "protocol": self.rng.choice(["OAuth 2.0 with PKCE", "mTLS", "SAML 2.0", "JWT with RS256"]),
            "scope": self.rng.choice(["API endpoints", "internal services", "admin dashboards"]),
            "minutes": str(ensure_non_round(self.rng.randint(15, 60), self.rng)),
            "platform": self.rng.choice(["Kubernetes", "ECS", "Docker Swarm", "Nomad"]),
            "vcpu": str(self.rng.choice([4, 8, 16, 32])),
            "phase": self.rng.choice(["Q1", "Q2", "Q3", "Phase 1", "Phase 2"]),
            "person": random_person(self.rng),
            "role_type": self.rng.choice(["frontend", "backend", "DevOps", "QA"]),
        }

    def _extract_keywords(self, vals: dict, kw_fields: list) -> list[str]:
        """Extract keyword values from generated values dict."""
        keywords = []
        for field in kw_fields:
            val = vals.get(field, "")
            # Add both formatted and raw versions for numbers
            keywords.append(str(val))
            raw = str(val).replace(",", "")
            if raw != str(val):
                keywords.append(raw)
        return keywords

    def _generate_transcript(self, t: int, decisions: list, team: list, project: str) -> dict:
        """Generate a meeting transcript embedding the decisions."""
        attendees = sample_unique(team, min(4, len(team)), self.rng)
        att_str = ", ".join(attendees)

        lines = [
            f"Meeting: {project} — Session {t} Review",
            f"Attendees: {att_str}",
            "",
        ]
        for d in decisions:
            lines.append(f"{d['person']} reported: {d['fact']}.")
            lines.append(f"  Category: {d['category']}. Decision ID: {d['id']}.")
            lines.append("")

        return {
            "session": t,
            "title": f"{project} Review — Session {t}",
            "transcript": "\n".join(lines),
        }

    def _generate_queries(self, t: int, current_decisions: list, all_decisions: list) -> list:
        """Generate 3 queries per session."""
        queries = []

        # 1-2 queries from current session
        for d in current_decisions[:2]:
            q = self._decision_to_query(d, len(queries), t)
            queries.append(q)

        # 1 query from a random earlier session (if available)
        earlier = [d for d in all_decisions if d["session"] < t]
        if earlier:
            old_d = self.rng.choice(earlier)
            q = self._decision_to_query(old_d, len(queries), t)
            queries.append(q)
        elif current_decisions:
            d = current_decisions[-1]
            q = self._decision_to_query(d, len(queries), t)
            queries.append(q)

        return queries

    def _decision_to_query(self, decision: dict, idx: int, session: int) -> dict:
        """Convert a decision into a query."""
        cat = decision["category"]
        fact = decision["fact"]
        person = decision["person"]

        # Generate question based on category
        q_templates = {
            "budget": f"What was the budget figure for: {fact.split()[0]} {fact.split()[1] if len(fact.split()) > 1 else ''}?",
            "tech": f"What technology decision was made regarding {fact.split()[0]}?",
            "vendor": f"Which vendor was selected for {fact.split()[-1] if len(fact.split()) > 2 else 'this service'}?",
            "timeline": f"What is the timeline for {fact.split()[0]}?",
            "security": f"What security measure was decided for {fact.split()[-1]}?",
            "hiring": f"What hiring decision was made?",
            "infra": f"What infrastructure setup was decided?",
        }

        return {
            "query_id": f"q{session}_{idx + 1}",
            "question": q_templates.get(cat, f"What was decided about {cat}?"),
            "gold_decision_ids": [decision["id"]],
            "keywords": decision["keywords"][:3],
        }

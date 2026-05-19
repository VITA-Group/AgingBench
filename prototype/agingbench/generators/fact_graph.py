"""
agingbench/generators/fact_graph.py — FactGraph: tracks facts, versions,
dependencies, and interference relationships across sessions.

Used by all generators to:
  1. Register facts as they're generated
  2. Track version chains (fact updates/superseding)
  3. Record task→fact dependency edges
  4. Track interference pairs (confusable entities across domains)
  5. Export a dependency_graph.json for post-hoc analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fact:
    """A single versioned fact in the knowledge graph."""
    id: str                                    # "fact_0", "fact_1", ...
    session: int                               # session when introduced
    domain: str                                # "shopping", "vendor", "personnel", ...
    content: str                               # human-readable text
    keywords: list[str]                        # values for keyword matching
    version: int = 1                           # 1 = original, 2+ = updated
    replaces: Optional[str] = None             # previous version's fact_id
    replaced_by: Optional[str] = None          # next version's fact_id (set later)
    interference_group: Optional[str] = None   # shared term for confusable pairs
    invalidated_at: Optional[int] = None       # session when retracted (None = still valid)


@dataclass
class DependencyEdge:
    """A task's dependency on one or more facts."""
    task_id: str                        # e.g. "s6_t3", "s3_q5"
    session: int                        # session of the dependent task
    depends_on: list[str]               # fact_ids this task requires
    dep_type: str                       # "compare", "trend", "synthesize", "standalone"
    chain_depth: int = 1                # longest path through version chains
    fact_versions_required: dict = field(default_factory=dict)  # {fact_id: version_needed}


@dataclass
class Accumulator:
    """A numeric state variable whose value is derived from accumulated deltas."""
    name: str
    initial_value: float
    session: int                           # session when created
    domain: str = ""
    deltas: list[dict] = field(default_factory=list)  # [{session, amount, description}]


class FactGraph:
    """
    Tracks the full fact lifecycle across generated sessions.

    Provides:
      - Fact registration and version tracking
      - Fact invalidation (selective forgetting)
      - Dependency edge recording (task → facts)
      - Interference pair tracking
      - Accumulator tracking (Ledger-QA-style derived values)
      - Export to dependency_graph.json
    """

    def __init__(self):
        self.facts: dict[str, Fact] = {}
        self.edges: list[DependencyEdge] = []
        self.interference_pairs: list[dict] = []
        self.accumulators: dict[str, Accumulator] = {}
        self._counter = 0

    # ------------------------------------------------------------------ facts

    def register_fact(
        self,
        session: int,
        domain: str,
        content: str,
        keywords: list[str],
        fact_id: Optional[str] = None,
    ) -> Fact:
        """Register a new fact. Returns the Fact object with assigned ID."""
        if fact_id is None:
            fact_id = f"fact_{self._counter}"
            self._counter += 1
        f = Fact(
            id=fact_id,
            session=session,
            domain=domain,
            content=content,
            keywords=list(keywords),
        )
        self.facts[fact_id] = f
        return f

    def update_fact(
        self,
        old_id: str,
        new_content: str,
        new_keywords: list[str],
        session: int,
        new_id: Optional[str] = None,
    ) -> Fact:
        """
        Create a new version of an existing fact.

        The old fact gets `replaced_by` set; the new fact gets `replaces` set.
        Returns the new Fact.
        """
        old = self.facts[old_id]
        if new_id is None:
            new_id = f"fact_{self._counter}"
            self._counter += 1
        new_fact = Fact(
            id=new_id,
            session=session,
            domain=old.domain,
            content=new_content,
            keywords=list(new_keywords),
            version=old.version + 1,
            replaces=old_id,
        )
        old.replaced_by = new_id
        self.facts[new_id] = new_fact
        return new_fact

    def get_current_version(self, fact_id: str) -> Fact:
        """Follow the version chain to the latest version."""
        f = self.facts[fact_id]
        while f.replaced_by is not None:
            f = self.facts[f.replaced_by]
        return f

    def get_version_chain(self, fact_id: str) -> list[Fact]:
        """Get the full version history for a fact (oldest first)."""
        # Find the root
        f = self.facts[fact_id]
        while f.replaces is not None:
            f = self.facts[f.replaces]
        # Walk forward
        chain = [f]
        while f.replaced_by is not None:
            f = self.facts[f.replaced_by]
            chain.append(f)
        return chain

    def get_facts_by_session(self, session: int) -> list[Fact]:
        """Get all facts introduced in a specific session."""
        return [f for f in self.facts.values() if f.session == session]

    def get_facts_before(self, session: int) -> list[Fact]:
        """Get all facts introduced before a specific session."""
        return [f for f in self.facts.values() if f.session < session]

    def get_current_facts_at(self, session: int) -> list[Fact]:
        """
        Get all facts that are 'current' at a given session.
        A fact is current if:
          - introduced at or before `session`
          - not yet replaced, OR replaced after `session`
          - not invalidated, OR invalidated after `session`
        """
        result = []
        for f in self.facts.values():
            if f.session > session:
                continue
            if f.invalidated_at is not None and f.invalidated_at <= session:
                continue  # retracted
            if f.replaced_by is not None:
                replacement = self.facts[f.replaced_by]
                if replacement.session <= session:
                    continue  # already superseded
            result.append(f)
        return result

    def get_versioned_facts(self) -> list[Fact]:
        """Get all facts that have been updated at least once (version > 1)."""
        return [f for f in self.facts.values() if f.version > 1]

    def get_updatable_facts(self, before_session: int) -> list[Fact]:
        """Get facts from before `session` that haven't been updated yet."""
        return [
            f for f in self.facts.values()
            if f.session < before_session
            and f.replaced_by is None
            and f.invalidated_at is None
            and f.version == 1  # only original versions
        ]

    def invalidate_fact(self, fact_id: str, session: int) -> Fact:
        """Mark a fact as retracted (no longer true). No replacement is created."""
        f = self.facts[fact_id]
        f.invalidated_at = session
        return f

    def get_invalidatable_facts(self, before_session: int) -> list[Fact]:
        """Get facts that can be invalidated: active, not already replaced or invalidated."""
        return [
            f for f in self.facts.values()
            if f.session < before_session
            and f.invalidated_at is None
            and f.replaced_by is None
        ]

    # ------------------------------------------------------------------ accumulators

    def register_accumulator(
        self, name: str, initial_value: float, session: int, domain: str = ""
    ) -> Accumulator:
        """Register a numeric accumulator (e.g., budget balance)."""
        acc = Accumulator(name=name, initial_value=initial_value, session=session, domain=domain)
        self.accumulators[name] = acc
        return acc

    def add_delta(
        self, accumulator_name: str, amount: float, session: int, description: str = ""
    ) -> None:
        """Add a delta event to an accumulator."""
        acc = self.accumulators[accumulator_name]
        acc.deltas.append({"session": session, "amount": amount, "description": description})

    def get_accumulator_value(self, name: str, at_session: int | None = None) -> float:
        """Compute ground-truth value at a session: initial + Σ deltas where session <= at_session."""
        acc = self.accumulators[name]
        total = acc.initial_value
        for d in acc.deltas:
            if at_session is None or d["session"] <= at_session:
                total += d["amount"]
        return total

    # ------------------------------------------------------------------ dependencies

    def add_dependency(
        self,
        task_id: str,
        session: int,
        fact_ids: list[str],
        dep_type: str = "standalone",
    ) -> DependencyEdge:
        """Record that a task depends on specific facts."""
        # Compute chain depth: max version depth across all dependencies
        max_depth = 0
        versions_required = {}
        for fid in fact_ids:
            chain = self.get_version_chain(fid)
            depth = len(chain)
            if depth > max_depth:
                max_depth = depth
            current = self.get_current_version(fid)
            versions_required[fid] = current.version

        edge = DependencyEdge(
            task_id=task_id,
            session=session,
            depends_on=list(fact_ids),
            dep_type=dep_type,
            chain_depth=max_depth,
            fact_versions_required=versions_required,
        )
        self.edges.append(edge)
        return edge

    # ------------------------------------------------------------------ interference

    def add_interference(
        self,
        fact_a_id: str,
        fact_b_id: str,
        shared_term: str,
    ) -> None:
        """Record that two facts share a confusable term across domains."""
        fa = self.facts[fact_a_id]
        fb = self.facts[fact_b_id]
        self.interference_pairs.append({
            "shared_term": shared_term,
            "fact_ids": [fact_a_id, fact_b_id],
            "domains": [fa.domain, fb.domain],
            "values": [fa.keywords[0] if fa.keywords else "", fb.keywords[0] if fb.keywords else ""],
        })
        fa.interference_group = shared_term
        fb.interference_group = shared_term

    # ------------------------------------------------------------------ export

    def export(self) -> dict:
        """
        Export the full dependency graph as a JSON-serializable dict.

        Output schema:
          tasks: {task_id: {session, depends_on_facts, dep_type, chain_depth, ...}}
          facts: {fact_id: {session, domain, keywords, versions: [...], ...}}
          interference_map: [{shared_term, fact_ids, domains, values}]
          summary: {total_facts, total_versioned, total_dependencies, max_chain_depth, ...}
        """
        # Build per-fact export with version history
        facts_export = {}
        seen_roots = set()
        for fid, f in self.facts.items():
            # Find root of version chain
            root = f
            while root.replaces is not None:
                root = self.facts[root.replaces]
            if root.id in seen_roots:
                continue
            seen_roots.add(root.id)

            chain = self.get_version_chain(root.id)
            facts_export[root.id] = {
                "introduced_session": root.session,
                "domain": root.domain,
                "versions": [
                    {
                        "version": c.version,
                        "session": c.session,
                        "keywords": c.keywords,
                        "content": c.content,
                        "fact_id": c.id,
                        "current": c.replaced_by is None and c.invalidated_at is None,
                        "invalidated_at": c.invalidated_at,
                    }
                    for c in chain
                ],
                "referenced_by_tasks": [
                    e.task_id for e in self.edges
                    if any(fid in [root.id] + [c.id for c in chain] for fid in e.depends_on)
                ],
            }

        # Build per-task export
        tasks_export = {}
        for e in self.edges:
            tasks_export[e.task_id] = {
                "session": e.session,
                "depends_on_facts": e.depends_on,
                "depends_on_sessions": sorted(set(
                    self.facts[fid].session for fid in e.depends_on if fid in self.facts
                )),
                "dependency_type": e.dep_type,
                "chain_depth": e.chain_depth,
                "fact_versions_required": e.fact_versions_required,
            }

        # Summary stats
        n_versioned = len([f for f in self.facts.values() if f.version > 1])
        n_invalidated = len([f for f in self.facts.values() if f.invalidated_at is not None])
        max_depth = max((e.chain_depth for e in self.edges), default=0)
        n_dep_tasks = len([e for e in self.edges if e.dep_type != "standalone"])

        # Export accumulators (if any)
        accum_export = {}
        for name, acc in self.accumulators.items():
            accum_export[name] = {
                "initial_value": acc.initial_value,
                "domain": acc.domain,
                "session": acc.session,
                "deltas": acc.deltas,
                "final_value": self.get_accumulator_value(name),
            }

        result = {
            "tasks": tasks_export,
            "facts": facts_export,
            "interference_map": self.interference_pairs,
            "summary": {
                "total_facts": len(self.facts),
                "total_unique_roots": len(facts_export),
                "total_versioned": n_versioned,
                "total_invalidated": n_invalidated,
                "total_dependencies": len(self.edges),
                "total_dependency_tasks": n_dep_tasks,
                "total_interference_pairs": len(self.interference_pairs),
                "max_chain_depth": max_depth,
            },
        }
        if accum_export:
            result["accumulators"] = accum_export
        return result

"""Programmatic generator for S4 Software Engineering scenario.

Produces codebase snapshots with growing dependency graphs, modification
tasks with ground-truth impact sets, and pytest suites that actually
compile and run — matching the curated JSON format.

Strategy: "REST API with models" archetype. Each session adds modules
with imports from existing ones, growing the dependency chain.
"""

from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path
from typing import Any

from .base import BaseGenerator
from .dependency_mixin import DependencyMixin
from .fact_graph import FactGraph
from .pools import CODE_ENTITIES, CODE_FIELDS, ensure_non_round
from .pressure_config import PressureConfig

# ---------------------------------------------------------------------------
# Code templates (all produce syntactically valid Python)
# ---------------------------------------------------------------------------

_MODEL_TEMPLATE = '''"""Auto-generated model: {entity}."""


class {entity}:
    """Simple {entity} model."""

    def __init__(self{init_params}):
{assignments}

    def to_dict(self) -> dict:
        return {dict_expr}

    def validate(self) -> bool:
        """Basic validation."""
        return True
'''

_MODEL_WITH_VALIDATION_TEMPLATE = '''"""Auto-generated model: {entity} with validation."""


class {entity}:
    """Validated {entity} model."""

    def __init__(self{init_params}):
{assignments}

    def to_dict(self) -> dict:
        return {dict_expr}

    def validate(self) -> bool:
        """Validate fields."""
{validation_body}
        return True
'''

_TEST_TEMPLATE = '''"""Tests for {entity} model."""

from models.{entity_lower} import {entity}


def test_{entity_lower}_create():
    obj = {entity}({test_args})
{assertions}


def test_{entity_lower}_to_dict():
    obj = {entity}({test_args})
    d = obj.to_dict()
    assert isinstance(d, dict)
{dict_assertions}


def test_{entity_lower}_validate():
    obj = {entity}({test_args})
    assert obj.validate() is True
'''

_TEST_VALIDATION_TEMPLATE = '''
def test_{entity_lower}_invalid_{field}():
    """Test that invalid {field} is rejected."""
    try:
        obj = {entity}({invalid_args})
        assert not obj.validate(), "Should reject invalid {field}"
    except (ValueError, TypeError):
        pass  # Also acceptable
'''

_UTILS_TEMPLATE = '''"""Utility functions for {module_name}."""


def {fn_name}({params}) -> {return_type}:
    """{docstring}"""
{body}
'''

# Task templates for code modifications
_TASK_TEMPLATES = [
    (
        "Add input validation to the {entity} model. "
        "Reject {field} values that are {constraint}.",
        "validation",
    ),
    (
        "Add a helper function to utils/{module}.py that {action}. "
        "Use it from the {entity} model.",
        "utility",
    ),
    (
        "Refactor {entity} to use a {pattern} pattern for {purpose}.",
        "refactor",
    ),
]


class S4Generator(BaseGenerator, DependencyMixin):
    """Generate S4 software engineering scenario data."""

    SCENARIO_ID = "s4_software_engineering"

    def __init__(self, seed: int = 42, pressure: PressureConfig | None = None):
        super().__init__(seed)
        self.pressure = pressure or PressureConfig.none()

    def generate(self, n_sessions: int = 8) -> dict[str, Any]:
        graph = FactGraph()
        entities = list(CODE_ENTITIES[:n_sessions + 2])  # enough entities
        self.rng.shuffle(entities)

        sessions = []
        snapshots = []
        # Track current project state
        project_files: dict[str, str] = {
            "models/__init__.py": "",
            "utils/__init__.py": "",
            "tests/__init__.py": "",
        }
        import_graph: dict[str, list[str]] = {}

        # Topic-matched (opt-in) code-confusable order: a stable shuffle of the
        # code-domain pool. The first n_confusable_pairs entries form a FIXED
        # cohort injected once (below) and re-probed every later session — the
        # S6-style interference-AGING design (decay of a fixed target as the
        # store grows), not per-session-fresh. Only built in topic mode, so
        # generic-mode RNG order is untouched.
        code_order = None
        if getattr(self.pressure, "confusable_topic_matched", False):
            from .pools import CODE_CONFUSABLE_PAIRS
            code_order = list(range(len(CODE_CONFUSABLE_PAIRS)))
            self.rng.shuffle(code_order)

        for t in range(n_sessions):
            entity = entities[t % len(entities)]
            fields = CODE_FIELDS.get(entity, [("name", "str"), ("value", "str")])

            # Add model file
            model_path = f"models/{entity.lower()}.py"
            model_code = self._gen_model(entity, fields)
            project_files[model_path] = model_code
            import_graph[model_path] = []

            # Add test file
            test_path = f"tests/test_{entity.lower()}.py"
            test_code = self._gen_tests(entity, fields)
            project_files[test_path] = test_code
            import_graph[test_path] = [model_path]

            # Compute expected tests (before agent modification)
            tests_expected = self._compute_test_ids(test_path, entity, fields, pre_task=True)

            # Generate task. In revisable_base_facts mode the validation limit
            # IS the fact's numeric value (config_value), so revision is
            # load-bearing: the agent must track the latest limit to validate
            # correctly, and the trend probe tests that tracking.
            field_names = ", ".join(f[0] for f in fields)
            if getattr(self.pressure, "revisable_base_facts", False):
                # >=3 digits clears the scorer's short-numeric reject; the value
                # sits in keywords[1] so the trend probe (eval on keywords[:2])
                # scores the latest LIMIT, not the static entity name.
                config_value = self.rng.randint(100, 9999)
                if getattr(self.pressure, "revisable_neutral_quantity", False):
                    # CLEAN revision: a neutral, identifier-like quantity (a
                    # configuration revision code) instead of "max length".
                    # "max length" invites reporting the MAXIMUM value found (a
                    # limit->max conflation) that inflates apparent aging; an
                    # identifier has no magnitude semantics, so the trend probe
                    # measures pure recall of the CURRENT version. Task-relevance
                    # was framing-only (never scored), so the normal coding task
                    # is used here rather than the spec_value validation task.
                    task_text, task_type = self._gen_task(entity, fields, t)
                    graph.register_fact(
                        session=t,
                        domain=task_type,
                        content=f"{entity} model: configuration revision code is {config_value}. {task_text[:60]}",
                        keywords=[entity, str(config_value), field_names],
                    )
                else:
                    task_text, task_type = self._gen_task(entity, fields, t, spec_value=config_value)
                    graph.register_fact(
                        session=t,
                        domain=task_type,
                        content=f"{entity} model: current max length is {config_value}. {task_text[:80]}",
                        keywords=[entity, str(config_value), field_names],
                    )
            else:
                task_text, task_type = self._gen_task(entity, fields, t)
                graph.register_fact(
                    session=t,
                    domain=task_type,
                    content=f"{entity} model: {task_text[:100]}",
                    keywords=[entity, field_names],
                )

            # Determine impact set (files affected by the task)
            impact_set = [model_path, test_path]
            # If utility task, add a utils file
            if task_type == "utility":
                util_path = f"utils/{entity.lower()}_utils.py"
                util_code = self._gen_utils(entity, fields)
                project_files[util_path] = util_code
                import_graph[model_path].append(util_path)
                impact_set.append(util_path)

            # Post-task tests (what should pass after correct edit)
            post_tests = self._compute_test_ids(test_path, entity, fields, pre_task=False)

            # Dependencies
            depends_on = list(range(max(0, t - 2), t))
            dep_context = ""
            if depends_on:
                prev_entity = entities[(t - 1) % len(entities)]
                dep_context = (
                    f"Session {t-1} added {prev_entity} model with "
                    f"{len(CODE_FIELDS.get(prev_entity, []))} fields."
                )

            # Held-out dependency probe — a SEPARATE eval question answered from
            # compressed memory only (no dep_context in the prompt), so it
            # measures genuine memory recall, unlike the dep_recall proxy which
            # scores against the dep_context re-injected into the task prompt.
            # Fired EVERY post-warmup session (not density-gated) so the faithful
            # curve is dense enough to be the headline. It does not alter the
            # coding task — the runner asks it after the task. (Removing the
            # density gate changes the S4 RNG stream vs. older seeds.)
            dep_probe = None
            if t >= self.pressure.warmup_sessions:
                dep_task = self.build_dependency_task(graph, t, self.rng, self.pressure)
                if dep_task:
                    dep_probe = {
                        "question": dep_task["text"],
                        "eval_keywords": dep_task["eval_keywords"],
                        "reference_answer": dep_task["reference_answer"],
                    }

            # Apply version updates
            updates = self.version_random_facts(graph, t, self.rng, self.pressure)
            if updates:
                dep_context += "\n" + "\n".join(u["text"] for u in updates)

            # Apply selective forgetting (revision aging)
            invalidations = self.invalidate_random_facts(graph, t, self.rng, self.pressure)
            if invalidations:
                dep_context += "\n" + "\n".join(inv["text"] for inv in invalidations)

            # Inject interference facts
            binding_probes: list[dict] = []
            if (code_order is not None
                    and t == self.pressure.confusable_start_session):
                # TOPIC-MATCHED (opt-in), S6-STYLE FIXED COHORT: inject
                # n_confusable_pairs confusable CODE pairs ONCE — near-twin
                # APIs/methods sharing a stem (from_dict/to_dict,
                # filter_by_tag/sort_by_tag, …) drawn from the codebase domain
                # instead of the generic business pool — then re-probe the SAME
                # pairs at every later session. probe_lags spans the whole
                # remaining horizon (1,2,…), so the runner's lag-scheduler asks
                # each pair once per subsequent session: a growing-lag sweep of a
                # fixed target = the canonical interference-AGING curve, matching
                # S6. gold = name_a, distractor = name_b; independent of dep_recall.
                from .pools import CODE_CONFUSABLE_PAIRS
                k = min(max(1, int(self.pressure.n_confusable_pairs)), len(code_order))
                # Default: re-probe every remaining session (full lag sweep).
                # An explicit confusable_probe_lags overrides (e.g. for a sparse
                # sweep), matching the generic path's knob.
                lags = (self.pressure.confusable_probe_lags
                        or list(range(1, max(2, n_sessions - t))))
                for j in range(k):
                    c = CODE_CONFUSABLE_PAIRS[code_order[j]]
                    dep_context += (
                        f"\n`{c['name_a']}` {c['desc_a']}; "
                        f"`{c['name_b']}` {c['desc_b']}."
                    )
                    binding_probes.append({
                        "probe_id": f"s{t}_codeinterf_{j}",
                        "question": c["probe_question"],
                        "gold_value": c["name_a"],
                        "distractor_value": c["name_b"],
                        "keywords": [c["name_a"]],
                        "inject_session": t,
                        "probe_lags": list(lags),
                    })
            elif (not getattr(self.pressure, "confusable_topic_matched", False)
                    and t >= self.pressure.confusable_start_session):
                pairs = self.inject_interference(graph, t, self.rng, self.pressure)
                if pairs:
                    dep_context += "\n" + "\n".join(
                        f"{p['text_a']} {p['text_b']}" for p in pairs
                    )
                    # Forced-choice binding probes — emitted by DEFAULT for every
                    # injected confusable pair so interference is *measured*, not
                    # just injected. Each probe asks for fact_a's value (gold)
                    # with fact_b's value as the confusable distractor (shared
                    # term). The runner asks these at fixed lags after injection
                    # and stores the raw answers under session_result[
                    # "interference_probes"], which score_interference_binding
                    # classifies as correct/confused/both/miss. This is separate
                    # from the coding task and does NOT touch dep_recall.
                    lags = self.pressure.confusable_probe_lags or [1, 3]
                    for j, p in enumerate(pairs):
                        q = p.get("probe_question") or (
                            f"What is the {p['shared_term']} for "
                            f"{p['fact_a']['domain']}? Reply with the exact value only."
                        )
                        binding_probes.append({
                            "probe_id": f"s{t}_interf_{j}",
                            "question": q,
                            "gold_value": p["fact_a"]["value"],
                            "distractor_value": p["fact_b"]["value"],
                            "keywords": [str(p["fact_a"]["value"])],
                            "inject_session": t,
                            "probe_lags": list(lags),
                        })

            session_entry = {
                "session": t,
                "task": task_text,
                "depends_on": depends_on,
                "dependency_context": dep_context,
                "files_to_modify": [model_path],
                "impact_set": impact_set,
                "test_commands": [f"python -m pytest {test_path} -v"],
                "n_files": len(project_files),
            }
            if dep_probe:
                session_entry["dependency_probe"] = dep_probe
            if binding_probes:
                session_entry["interference_binding_probes"] = binding_probes
            sessions.append(session_entry)

            snapshots.append({
                "session": t,
                "n_files": len(project_files),
                "files": dict(project_files),  # snapshot copy
                "tests_expected": tests_expected,
                "post_task_tests": post_tests,
            })

        # Life event at session n//2
        life_event = {
            "session": n_sessions // 2,
            "type": "memory_compaction",
            "description": "Force compress agent memory to 500 chars",
        }

        result = {
            "tasks": {"sessions": sessions, "life_event": life_event},
            "snapshots": {"snapshots": snapshots},
        }
        result["dependency_graph"] = graph.export()
        return result

    def _gen_model(self, entity: str, fields: list[tuple[str, str]]) -> str:
        """Generate a model class."""
        init_params = "".join(f", {name}: {typ}" for name, typ in fields)
        assignments = "\n".join(f"        self.{name} = {name}" for name, _ in fields)
        dict_items = ", ".join(f'"{name}": self.{name}' for name, _ in fields)
        dict_expr = "{" + dict_items + "}"

        return _MODEL_TEMPLATE.format(
            entity=entity,
            init_params=init_params,
            assignments=assignments,
            dict_expr=dict_expr,
        )

    def _gen_tests(self, entity: str, fields: list[tuple[str, str]]) -> str:
        """Generate pytest tests for a model."""
        test_args = ", ".join(self._default_value(typ) for _, typ in fields)
        assertions = "\n".join(
            f"    assert obj.{name} == {self._default_value(typ)}"
            for name, typ in fields
        )
        dict_assertions = "\n".join(
            f'    assert d["{name}"] == {self._default_value(typ)}'
            for name, typ in fields
        )

        return _TEST_TEMPLATE.format(
            entity=entity,
            entity_lower=entity.lower(),
            test_args=test_args,
            assertions=assertions,
            dict_assertions=dict_assertions,
        )

    def _gen_utils(self, entity: str, fields: list[tuple[str, str]]) -> str:
        """Generate a utility module."""
        fn_name = f"format_{entity.lower()}"
        params = f"{entity.lower()}_data: dict"
        body = f'    return str({entity.lower()}_data)'
        return _UTILS_TEMPLATE.format(
            module_name=entity.lower(),
            fn_name=fn_name,
            params=params,
            return_type="str",
            docstring=f"Format {entity} data as string.",
            body=body,
        )

    def _gen_task(self, entity: str, fields: list[tuple[str, str]], session: int,
                  spec_value: int | None = None) -> tuple[str, str]:
        """Generate a modification task.

        When ``spec_value`` is given (revisable_base_facts mode), force a
        validation task whose length limit IS ``spec_value``, so the revised
        value is load-bearing: the agent must track the LATEST limit to validate
        correctly, and a held-out probe can test that tracking — genuine
        revision aging rather than recall of a synthetic number.
        """
        if spec_value is not None:
            field_name = self.rng.choice(fields)[0]
            text = (
                f"Add input validation to the {entity} model in models/{entity.lower()}.py. "
                f"Reject {field_name} values longer than {spec_value} characters "
                f"(the current configured max length for {entity})."
            )
            return text, "validation"

        task_type = ["validation", "utility", "refactor"][session % 3]

        if task_type == "validation":
            field_name, field_type = self.rng.choice(fields)
            if field_type == "int":
                lo, hi = 0, ensure_non_round(self.rng.randint(50, 500), self.rng)
                constraint = f"outside the range [{lo}, {hi}]"
            elif field_type == "str":
                constraint = "empty or longer than 255 characters"
            else:
                constraint = "None or invalid"
            text = (
                f"Add input validation to the {entity} model in models/{entity.lower()}.py. "
                f"Reject {field_name} values that are {constraint}."
            )
        elif task_type == "utility":
            text = (
                f"Add a helper function to utils/{entity.lower()}_utils.py that formats "
                f"{entity} data as a string. Use it from the {entity} model's __repr__ method."
            )
        else:
            text = (
                f"Refactor {entity} to add a from_dict class method that creates "
                f"an instance from a dictionary. Add corresponding tests."
            )

        return text, task_type

    def _compute_test_ids(
        self, test_path: str, entity: str, fields: list, pre_task: bool,
    ) -> dict[str, str]:
        """Compute expected test pass/fail status."""
        el = entity.lower()
        tests = {
            f"{test_path}::test_{el}_create": "pass",
            f"{test_path}::test_{el}_to_dict": "pass",
            f"{test_path}::test_{el}_validate": "pass",
        }
        if not pre_task:
            # After task, validation tests should also exist
            if fields:
                tests[f"{test_path}::test_{el}_invalid_{fields[0][0]}"] = "pass"
        return tests

    def _default_value(self, typ: str) -> str:
        """Return a default value literal for a type."""
        return {
            "str": '"test"',
            "int": "42",
            "float": "3.14",
            "bool": "True",
        }.get(typ, '"test"')

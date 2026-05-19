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

            # Generate task (add validation, utility, or refactor)
            task_text, task_type = self._gen_task(entity, fields, t)

            # Register design decisions in the FactGraph (use task_type as domain
            # so facts span multiple domains and enable COMPARE dependency tasks)
            field_names = ", ".join(f[0] for f in fields)
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

            # Apply dependency task (appended as separate probe, not replacing coding task)
            dep_probe = None
            if t >= self.pressure.warmup_sessions and self.rng.random() < self.pressure.dependency_density:
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
            if t >= self.pressure.confusable_start_session:
                pairs = self.inject_interference(graph, t, self.rng, self.pressure)
                if pairs:
                    dep_context += "\n" + "\n".join(
                        f"{p['text_a']} {p['text_b']}" for p in pairs
                    )

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

    def _gen_task(self, entity: str, fields: list[tuple[str, str]], session: int) -> tuple[str, str]:
        """Generate a modification task."""
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

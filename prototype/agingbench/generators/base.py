"""Base class for programmatic scenario generators."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseGenerator(ABC):
    """Abstract base for scenario data generators.

    Each subclass produces data in the exact same JSON format as the
    hand-crafted curated files, so runners need no modification beyond
    accepting a ``generated_data`` dict.
    """

    SCENARIO_ID: str = ""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    @abstractmethod
    def generate(self, n_sessions: int) -> dict[str, Any]:
        """Generate all data files for the scenario.

        Returns a dict keyed by logical file name (e.g. ``"session_tasks"``),
        each value being the in-memory equivalent of the JSON file content.
        """

    def write_to_dir(self, output_dir: Path, n_sessions: int) -> None:
        """Generate and write to disk as JSON files (for inspection)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = self.generate(n_sessions)
        for name, content in data.items():
            path = output_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)

"""S8 SWE-bench-Aging generator (Phase 0 stub).

Selects an issue chain + schedules lifecycle events per PressureConfig.
Produces a structured stream the runner consumes per session.

Phase 0: returns an empty stream + records pressure_used so the
dispatch contract holds. Phase 1 wires real chain selection.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

from agingbench.generators.base import BaseGenerator
from agingbench.generators.pressure_config import PressureConfig


SCENARIO_DIR = (Path(__file__).parent.parent
                / "scenarios" / "s8_swe_bench")


class S8SweBenchGenerator(BaseGenerator):
    """Programmatic stream builder for S8 SWE-bench-Aging.

    Phase 0 stub. Public API is the same shape as Phase 1 will deliver,
    so downstream consumers can be written against it now.
    """

    SCENARIO_ID = "s8_swe_bench"

    def __init__(self,
                 seed: int = 42,
                 pressure: Optional[PressureConfig] = None,
                 chain_path: Optional[Path] = None):
        super().__init__(seed=seed)
        self._seed = int(seed)
        self.pressure = pressure or PressureConfig.medium()
        self.chain_path = Path(chain_path) if chain_path else None

    def generate(self, n_sessions: int) -> dict[str, Any]:
        """Build the longitudinal stream.

        Phase 0: returns empty session_issues + lifecycle_events,
        records pressure_used. Phase 1 wires real chain selection +
        pressure-driven lifecycle scheduling.
        """
        rng = random.Random(self._seed * 9007 + 13)
        n = max(0, int(n_sessions))

        # Phase 0 stub: empty stream. Phase 1 populates from
        # self.chain_path according to the seed manifest.
        session_issues: list[dict] = []
        lifecycle_events: list[dict] = []

        return {
            "session_issues": session_issues,
            "lifecycle_events": lifecycle_events,
            "dependency_graph": {
                "version_chains": [],
                "dependency_edges": [],
                "interference_pairs": [],
                "accumulators": {},
            },
            "chain_used": str(self.chain_path) if self.chain_path else None,
            "pressure_used": self.pressure.to_dict(),
            "phase": "phase_0_stub",
        }

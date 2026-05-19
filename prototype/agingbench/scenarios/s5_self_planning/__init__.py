"""S5 — Self-Planning Notebook evaluation scenario.

Tier 1 with workspace-file access: the agent manages its own workspace
files (notes, plans, scratch) via `ReactFileAdapter`. The benchmark
drives the loop — task delivery, session boundaries, response scoring.

Primary data path is through S5Generator (agingbench/generators/s5_generator.py).
"""

from pathlib import Path

SCENARIO_DIR = Path(__file__).parent

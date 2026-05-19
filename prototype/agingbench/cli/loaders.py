"""
agingbench/cli/loaders.py — Pure loading/discovery functions.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
SUITE_DIR = Path(__file__).parent.parent / "registry" / "suites"
SUT_DIR = Path(__file__).parent.parent / "registry" / "suts"
SCENARIO_DIR = Path(__file__).parent.parent / "scenarios"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_suite(suite_id: str) -> dict:
    path = SUITE_DIR / f"{suite_id}.yaml"
    if not path.exists():
        print(f"[error] Suite not found: {path}", file=sys.stderr)
        sys.exit(1)
    return _load_yaml(path)


def _resolve_suts(suite: dict, sut_arg: Optional[str]) -> list[Path]:
    if sut_arg:
        p = Path(sut_arg)
        if not p.exists():
            p = PROJECT_ROOT / sut_arg
        if not p.exists():
            print(f"[error] SUT file not found: {sut_arg}", file=sys.stderr)
            sys.exit(1)
        return [p]
    # use all default SUTs registered in the suite
    suts = []
    for rel in suite.get("default_suts", []):
        p = PROJECT_ROOT / rel
        if p.exists():
            suts.append(p)
        else:
            print(f"[warn] registered SUT not found, skipping: {rel}")
    if not suts:
        print("[error] No SUTs found. Pass --sut <path> or add default_suts to suite YAML.",
              file=sys.stderr)
        sys.exit(1)
    return suts


def _load_agent_class(agent_spec: Optional[str]):
    """
    Load a custom agent class from a dotted path like 'my_module:MyAgent'.

    Returns ReferenceAgent if agent_spec is None.
    """
    from agingbench.core.agent import ReferenceAgent
    if not agent_spec:
        return ReferenceAgent
    if ":" not in agent_spec:
        print(f"[error] --agent must be 'module.path:ClassName', got '{agent_spec}'",
              file=sys.stderr)
        sys.exit(1)
    module_path, class_name = agent_spec.rsplit(":", 1)
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        print(f"[error] Cannot import agent module '{module_path}': {e}", file=sys.stderr)
        sys.exit(1)
    cls = getattr(mod, class_name, None)
    if cls is None:
        print(f"[error] Class '{class_name}' not found in module '{module_path}'", file=sys.stderr)
        sys.exit(1)
    return cls


def _discover_scenarios() -> dict[str, dict]:
    """
    Scan scenarios/ for scenario.yaml manifests.
    Returns {scenario_id: manifest_dict} including aliases.
    """
    manifests = {}
    for manifest_path in SCENARIO_DIR.glob("*/scenario.yaml"):
        manifest = _load_yaml(manifest_path)
        sid = manifest["scenario_id"]
        manifest["_dir"] = manifest_path.parent
        manifests[sid] = manifest
        for alias in manifest.get("aliases", []):
            manifests[alias] = manifest
    return manifests


def _resolve_pressure(sut_cfg: Optional[dict] = None,
                      scenario_cfg: Optional[dict] = None,
                      manifest: Optional[dict] = None):
    """
    Resolve a PressureConfig from layered configuration.

    Lookup order (first match wins):
      1. sut_cfg["pressure"]  — per-run override on the SUT YAML
      2. scenario_cfg["pressure"] — suite-level override on a scenario entry
      3. manifest["pressure"]  — default declared in scenario.yaml
      4. fallback to PressureConfig.medium() (the pre-extension hard-coded default)

    Each layer accepts either:
      - a string preset name: "none" | "light" | "medium" | "heavy"
      - a dict with explicit field overrides (e.g., {"preset": "light", "n_confusable_pairs": 5})
      - a dict with raw PressureConfig kwargs (no "preset" key)

    Backward compatibility: when none of the four layers provides a "pressure"
    key, the returned config is identical (per `to_dict()`) to the pre-extension
    `PressureConfig.medium()`. This is gated by
    `tests/test_pressure_externalization.py::test_default_fallback_equals_medium`.
    """
    # Import here to avoid hard dependency at module import time (matches the
    # existing pattern in runners.py).
    from agingbench.generators.pressure_config import PressureConfig

    for layer in (sut_cfg, scenario_cfg, manifest):
        if not layer:
            continue
        spec = layer.get("pressure") if isinstance(layer, dict) else None
        if spec is None:
            continue
        return _build_pressure(spec, PressureConfig)

    return PressureConfig.medium()


def _build_pressure(spec, PressureConfig):
    """Construct a PressureConfig from a string preset name or a dict spec."""
    if isinstance(spec, str):
        return _apply_preset(spec, PressureConfig)

    if isinstance(spec, dict):
        # Optional preset name + per-field overrides.
        preset_name = spec.get("preset")
        if preset_name:
            base = _apply_preset(preset_name, PressureConfig)
            overrides = {k: v for k, v in spec.items() if k != "preset"}
        else:
            base = PressureConfig()  # default values
            overrides = dict(spec)
        if overrides:
            base_dict = base.to_dict() if hasattr(base, "to_dict") else base.__dict__
            base_dict.update(overrides)
            return PressureConfig(**base_dict)
        return base

    # Unknown spec type; fall back to medium rather than erroring.
    return PressureConfig.medium()


def _apply_preset(name: str, PressureConfig):
    """Look up a named preset on PressureConfig; default to medium for unknown names."""
    name = (name or "").strip().lower()
    presets = {
        "none": PressureConfig.none,
        "light": PressureConfig.light,
        "medium": PressureConfig.medium,
        "heavy": PressureConfig.heavy,
    }
    factory = presets.get(name, PressureConfig.medium)
    return factory()

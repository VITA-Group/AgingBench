"""
agingbench/telemetry/profiles/ — Deployment-type measurement templates.

A profile encodes (per deployment type):
  - default outcome-extraction rules
  - subject-linkage rules
  - mechanism-inference weights
  - default privacy patterns
  - session-detection defaults

Users select a profile via `trace_to_card(..., profile="code_assistant")`
and may override individual rules at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


PROFILES_DIR = Path(__file__).parent


@dataclass
class Profile:
    deployment_type:        str
    outcome_rules:          dict = field(default_factory=dict)
    subject_linkage:        dict = field(default_factory=dict)
    mechanism_weights:      dict = field(default_factory=dict)
    session_detection:      dict = field(default_factory=dict)
    privacy_patterns:       list = field(default_factory=list)
    raw:                    dict = field(default_factory=dict)


def load_profile(name: str = "generic") -> Profile:
    """Load a shipped profile by name. Falls back to `generic` if unknown."""
    import yaml

    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        path = PROFILES_DIR / "generic.yaml"
    with path.open() as f:
        doc = yaml.safe_load(f) or {}
    return Profile(
        deployment_type=doc.get("deployment_type", name),
        outcome_rules=doc.get("outcome_rules", {}) or {},
        subject_linkage=doc.get("subject_linkage", {}) or {},
        mechanism_weights=doc.get("mechanism_weights", {}) or {},
        session_detection=doc.get("session_detection", {}) or {},
        privacy_patterns=doc.get("privacy_patterns", []) or [],
        raw=doc,
    )


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def merge_overrides(profile: Profile, overrides: Optional[dict]) -> Profile:
    """Return a new Profile with the user's overrides merged in."""
    if not overrides:
        return profile
    return Profile(
        deployment_type=profile.deployment_type,
        outcome_rules={**profile.outcome_rules, **overrides.get("outcome_rules", {})},
        subject_linkage={**profile.subject_linkage, **overrides.get("subject_linkage", {})},
        mechanism_weights={**profile.mechanism_weights, **overrides.get("mechanism_weights", {})},
        session_detection={**profile.session_detection, **overrides.get("session_detection", {})},
        privacy_patterns=profile.privacy_patterns + (overrides.get("privacy_patterns") or []),
        raw=profile.raw,
    )


def outcome_rules_hash(profile: Profile) -> str:
    """Stable hash of the effective outcome rules — used by AgingCards as the
    cross-team comparability discriminator."""
    import hashlib
    import json
    payload = json.dumps(
        {"outcome_rules": profile.outcome_rules,
         "deployment_type": profile.deployment_type},
        sort_keys=True,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:32]

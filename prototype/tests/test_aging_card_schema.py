"""Tests for AgingCard schema validation and migration utility."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agingbench.metrics.aging_card import build_aging_card, AGING_CARD_SCHEMA_VERSION
from agingbench.metrics.aging_card_validate import (
    REQUIRED_TOP_LEVEL,
    SCHEMA_PATH,
    validate_card_dict,
    validate_card_path,
)
from agingbench.metrics.aging_card_migrate import migrate


FIXTURE_METRICS = {
    "scenario": "s1_research_literature",
    "sut_id": "haiku45_lossy_compress",
    "metric_group": "G1",
    "m0": 0.95,
    "m_final": 0.42,
    "half_life": 3.5,
    "decay_slope": -0.0752,
    "hazard_proxy": 0.22,
    "n_checkpoints": 10,
    "n_sessions": 10,
    "headline_metric": "keyword_recall",
    "aging_detected": True,
    "checkpoints": [[i, 0.95 - i * 0.05] for i in range(10)],
    "session_results": [],
}

FIXTURE_SUT = {
    "sut_id": "haiku45_lossy_compress", "seed": 42,
    "model": {"provider": "litellm", "model": "claude-haiku-4-5-20251001"},
    "memory_policy": {"type": "summarize_store"},
}


def test_schema_file_exists():
    assert SCHEMA_PATH.is_file(), f"schema JSON missing: {SCHEMA_PATH}"
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["title"] == "AgingCard"
    assert schema["properties"]["card_type"]["const"] == "agingbench.AgingCard"


def test_freshly_built_card_validates():
    """An AgingCard built by build_aging_card must pass schema validation."""
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    errors = validate_card_dict(card)
    assert not errors, f"freshly built card has errors: {errors}"


def test_missing_required_field_detected():
    """Removing a required field must produce a validation error."""
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    del card["headline"]
    errors = validate_card_dict(card)
    assert any("headline" in e for e in errors)


def test_bad_card_type_detected():
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    card["card_type"] = "not.an.aging.card"
    errors = validate_card_dict(card)
    assert any("card_type" in e for e in errors)


def test_bad_schema_version_detected():
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    card["schema_version"] = "v1"  # not SemVer
    errors = validate_card_dict(card)
    assert any("schema_version" in e for e in errors)


def test_mechanism_metrics_must_have_all_four():
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    del card["mechanism_metrics"]["compression"]
    errors = validate_card_dict(card)
    assert any("compression" in e for e in errors)


def test_validate_card_path(tmp_path):
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    p = tmp_path / "card.json"
    p.write_text(json.dumps(card))
    assert validate_card_path(p) == []


def test_validate_card_path_missing_file(tmp_path):
    errors = validate_card_path(tmp_path / "does_not_exist.json")
    assert errors and "not a file" in errors[0]


def test_sample_card_validates():
    """The fixture sample card (committed in examples/sample_cards/) must validate."""
    sample = Path(__file__).parent.parent / "examples" / "sample_cards" / "s1_research_literature_haiku45_lossy_compress.json"
    if not sample.is_file():
        pytest.skip("sample card not present")
    errors = validate_card_path(sample)
    assert errors == []


# ---------- migration tests ----------

def test_migrate_noop_at_current_version():
    """A card already at the current schema_version migrates to itself."""
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    assert card["schema_version"] == AGING_CARD_SCHEMA_VERSION
    migrated = migrate(card)
    assert migrated["schema_version"] == AGING_CARD_SCHEMA_VERSION
    # migrate() must NOT mutate the input
    assert card is not migrated


def test_migrate_unknown_version_raises():
    """A card claiming an unknown schema_version triggers a RuntimeError."""
    card = build_aging_card(metrics=FIXTURE_METRICS, sut_cfg=FIXTURE_SUT)
    card = dict(card)
    card["schema_version"] = "0.9.0"  # made-up old version
    with pytest.raises(RuntimeError) as exc:
        migrate(card)
    assert "no migration path" in str(exc.value)

"""
privacy_scrubber.py — PII redaction before any inference touches text.

Every prompt_preview / response_preview field is run through this before
session detection. Production teams need this guarantee or they won't
ship telemetry mode.

Patterns are conservative defaults. Profile YAMLs can extend with
domain-specific patterns (CS account IDs, healthcare MRNs, etc.).
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Optional, Pattern

from .schema import TelemetryRecord


# (compiled_regex, replacement)
DEFAULT_PII_PATTERNS: list[tuple[Pattern[str], str]] = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),                              '[SSN]'),
    (re.compile(r'\b\d{16}\b'),                                         '[CC]'),
    (re.compile(r'\b\+?[1-9]\d{9,14}\b'),                               '[PHONE]'),
    (re.compile(r'sk-[A-Za-z0-9_-]{20,}'),                              '[API_KEY]'),
    (re.compile(r'(?:Bearer|bearer)\s+[A-Za-z0-9._-]+'),                '[AUTH]'),
    # IPv4
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),                        '[IP]'),
]


def scrub_text(s: Optional[str], extra_patterns: Optional[list[tuple[Pattern[str], str]]] = None) -> Optional[str]:
    if not s:
        return s
    patterns = DEFAULT_PII_PATTERNS + (extra_patterns or [])
    for pat, repl in patterns:
        s = pat.sub(repl, s)
    return s


def hash_session_id(raw: str) -> str:
    """Stable hash for session IDs. Used by both scrub_record and the
    outcome-event loader so outcomes still join the right records after
    PII-scrubbing."""
    return "sid_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def hash_user_id(raw: str) -> str:
    return "uid_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def scrub_record(
    record: TelemetryRecord,
    extra_patterns: Optional[list[tuple[Pattern[str], str]]] = None,
    hash_user_ids: bool = True,
) -> TelemetryRecord:
    """Return a copy of `record` with prompt/response previews scrubbed and
    optionally user identifiers hashed.
    """
    new = dataclasses.replace(
        record,
        prompt_preview=scrub_text(record.prompt_preview, extra_patterns),
        response_preview=scrub_text(record.response_preview, extra_patterns),
    )
    if hash_user_ids and new.session_id:
        new.session_id = hash_session_id(new.session_id)
    if hash_user_ids and new.user_id_hash:
        new.user_id_hash = hash_user_id(new.user_id_hash)
    return new


def scrub_records(
    records: list[TelemetryRecord],
    extra_patterns: Optional[list[tuple[Pattern[str], str]]] = None,
    hash_user_ids: bool = True,
) -> list[TelemetryRecord]:
    return [scrub_record(r, extra_patterns, hash_user_ids) for r in records]

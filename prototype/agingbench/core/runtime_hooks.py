"""
agingbench/core/runtime_hooks.py — Composable post-session runtime-intervention
hooks (typed-state overlay + runtime controller).

Hooks are registered onto runner._post_session_hooks (list, default empty).
Each hook receives (runner, session_idx) and may inspect runner state, call
methods on the memory policy, or update its own internal state.

Drives:
  * typed-state overlay: initialize accumulator state at session 0; apply
    per-session deltas after each session.
  * runtime controller: read per-session metrics from
    runner._latest_session_record and dispatch promote_to_typed_state /
    switch_compaction_policy actions when thresholds are crossed.

The runner (S2Runner) is intentionally untouched beyond the additive
_post_session_hooks list, the for-loop call, and the surfacing of
_latest_session_record. All experiment-specific control flow lives here.
"""

from typing import Optional, Callable

from .memory.typed_state import TypedStateOverlay
from .memory.summarize_store import SummarizeStorePolicy
from .controller import ThresholdController


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _find_typed_state_overlay(memory_policy) -> Optional[TypedStateOverlay]:
    if isinstance(memory_policy, TypedStateOverlay):
        return memory_policy
    return None


def _find_inner_summarize_store(memory_policy) -> Optional[SummarizeStorePolicy]:
    if isinstance(memory_policy, SummarizeStorePolicy):
        return memory_policy
    if isinstance(memory_policy, TypedStateOverlay) and isinstance(
        memory_policy.inner, SummarizeStorePolicy
    ):
        return memory_policy.inner
    return None


# ---------------------------------------------------------------------------
# Accumulator extraction from generator output
# ---------------------------------------------------------------------------

def _extract_accumulators(generated_data: dict) -> dict[str, dict]:
    """Pull accumulator initials and per-session deltas from generated_data.

    Returns {acc_name: {"initial": float, "deltas_by_session": {sess: delta}}}.

    Strategy: read constraint-derived initial values from
    generated_data['source_profile']['constraints'], and parse delta task
    texts from session_tasks for per-session amounts.
    """
    out: dict[str, dict] = {}
    profile = generated_data.get("source_profile", {})
    accumulator_initials: dict[str, float] = {}
    for c in profile.get("constraints", []):
        if c.get("category") in ("dining", "financial", "subscriptions"):
            try:
                initial = float(c["keywords"][0])
                acc_name = f"{c['category']}_budget"
                accumulator_initials[acc_name] = initial
            except (ValueError, IndexError, KeyError):
                continue

    # NB: the S2 generator chooses the delta-description verb ("spent",
    # "received", etc.) INDEPENDENTLY of the amount sign — so the verb is
    # not a reliable indicator. The sign is encoded in the trailing clause:
    #   "comes from your <category> budget"  -> negative (spend)
    #   "adds back to your <category> budget" -> positive (credit)
    # We parse the absolute amount from the dollar figure and apply the sign
    # from the suffix.
    import re
    amount_re = re.compile(r"\$(\d+)")

    deltas: dict[int, list[tuple[str, float]]] = {}
    sessions = generated_data.get("session_tasks", {}).get("sessions", [])
    for sess_idx, sess in enumerate(sessions):
        for task in sess.get("tasks", []):
            if task.get("category") != "accumulator_delta":
                continue
            text = task.get("text", "")
            for acc_name in accumulator_initials.keys():
                category = acc_name.replace("_budget", "")
                if f"your {category} budget" not in text:
                    continue
                m = amount_re.search(text)
                if not m:
                    continue
                magnitude = float(m.group(1))
                # Determine sign from suffix wording.
                if f"comes from your {category} budget" in text:
                    delta_value = -magnitude
                elif f"adds back to your {category} budget" in text:
                    delta_value = +magnitude
                else:
                    # Fallback: assume spend (most common in S2 generator).
                    delta_value = -magnitude
                deltas.setdefault(sess_idx, []).append((acc_name, delta_value))

    for name, initial in accumulator_initials.items():
        out[name] = {"initial": initial, "deltas_by_session": {}}
    for sess_idx, dlist in deltas.items():
        for name, amount in dlist:
            if name in out:
                out[name]["deltas_by_session"].setdefault(sess_idx, 0.0)
                out[name]["deltas_by_session"][sess_idx] += amount
    return out


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------

def make_typed_state_hook(generated_data: dict, verbose: bool = True):
    """Hook that drives the typed-state overlay.

    Session 0: initialize accumulator initials.
    Session t (t >= 1): apply that session's accumulator deltas.

    No-op when memory_policy is not a TypedStateOverlay or overlay.enabled is
    False. (The controller may flip enabled later; this hook keeps state up to
    date so activation sees a current value, not stale.)
    """
    accumulators = _extract_accumulators(generated_data)
    if verbose:
        print(
            f"  [hook][typed-state] discovered accumulators: "
            f"{ {k: v['initial'] for k, v in accumulators.items()} }",
            flush=True,
        )

    def hook(runner, session_idx: int):
        overlay = _find_typed_state_overlay(runner.memory_policy)
        if overlay is None:
            return
        # Track state regardless of enabled, so controller-late-activation
        # sees a current value rather than starting from zero.
        if session_idx == 0 and not overlay.state:
            for name, info in accumulators.items():
                overlay.state[name] = {
                    "initial": info["initial"],
                    "remaining": info["initial"],
                }
                overlay.write_log.append({
                    "kind": "init", "session": 0, "name": name,
                    "value": info["initial"], "remaining_after": info["initial"],
                })
            if verbose:
                print(
                    f"  [hook][typed-state] session 0: initialized -> "
                    f"{ {k: v['remaining'] for k, v in overlay.state.items()} }",
                    flush=True,
                )
        for name, info in accumulators.items():
            delta = info["deltas_by_session"].get(session_idx, 0.0)
            if delta == 0.0:
                continue
            if name not in overlay.state:
                overlay.state[name] = {
                    "initial": info["initial"],
                    "remaining": info["initial"],
                }
            overlay.state[name]["remaining"] += delta
            overlay.write_log.append({
                "kind": "delta", "session": session_idx, "name": name,
                "value": delta, "remaining_after": overlay.state[name]["remaining"],
            })
            if verbose and overlay.enabled:
                print(
                    f"  [hook][typed-state] session {session_idx}: "
                    f"{name} {delta:+.0f} -> remaining={overlay.state[name]['remaining']:.0f}",
                    flush=True,
                )

    return hook


CAREFUL_PROMPT = """You are a project knowledge manager. Below is a project specification document.
Rewrite it as a concise summary. You MUST preserve ALL of the following verbatim:
- Every specific budget figure (exact dollar amounts with the $ sign)
- Every deadline (exact dates including month and day)
- Every named person and their assigned role
- Every technical constraint (specific version numbers and technology names)
Do not omit any named constraint. Use clear, direct language. Be concise but complete.

DOCUMENT:
{text}

SUMMARY:"""


import re as _re_accum

# Matches a numeric expression with optional sign(s) and currency prefix.
# Captures the full token so we can preserve sign semantics. Examples that
# match: "$1,234", "-$50", "$-50", "1234", "-1234.56", "1,234.5".
_ACCUM_TOKEN_RE = _re_accum.compile(
    r"-?\s*\$?\s*-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
# A negative lookbehind to avoid matching inside identifiers/dates like
# "2026-05-15" (the "05" would otherwise match).
_ACCUM_TOKEN_RE_ANCHORED = _re_accum.compile(
    r"(?<![A-Za-z0-9])-?\s*\$?\s*-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
# Words that typically precede the answer to "what is your remaining budget?"
_ACCUM_ANCHOR_WORDS = (
    "remaining", "remain", "left", "balance", "available",
    "answer", "answer:", "is", "have", "total",
)


def _accum_parse_token(token: str) -> Optional[float]:
    """Parse a matched numeric token (possibly with $/sign/comma) into a float.

    Two leading minuses collapse to positive (so "-$-50" becomes 50.0); a
    single minus anywhere in the token makes the value negative.
    """
    if not token:
        return None
    cleaned = token.replace("$", "").replace(",", "").replace(" ", "")
    is_neg = (cleaned.count("-") % 2) == 1
    cleaned = cleaned.replace("-", "")
    if not cleaned:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    return -val if is_neg else val


def _accum_parse_one(text: str) -> Optional[float]:
    """Extract the agent's intended numeric answer from a probe response.

    Strategy (in order of preference):
      1. If a number appears immediately after an anchor word ("remaining",
         "balance", "is", ...), use that one.
      2. Otherwise use the FIRST currency-formatted number (with $ prefix).
      3. Otherwise use the LAST plain number in the response.

    Handles thousand-separator commas correctly: "$1,234" → 1234.0.
    Handles negative currency: "-$50" / "$-50" → -50.0.
    Avoids parsing date components in "2026-05-15" via word-boundary anchor.
    """
    if not text:
        return None
    lower = text.lower()

    # Pass 1: number adjacent to anchor word
    for word in _ACCUM_ANCHOR_WORDS:
        for m in _re_accum.finditer(
            r"\b" + _re_accum.escape(word) + r"\b[^0-9\-$]{0,40}?"
            r"(-?\s*\$?\s*-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)",
            lower,
        ):
            val = _accum_parse_token(m.group(1))
            if val is not None:
                return val

    # Pass 2: first currency-formatted ($-prefixed) number
    for m in _ACCUM_TOKEN_RE_ANCHORED.finditer(text):
        span = m.group(0)
        if "$" in span:
            val = _accum_parse_token(span)
            if val is not None:
                return val

    # Pass 3: last plain number
    plain_matches = list(_ACCUM_TOKEN_RE_ANCHORED.finditer(text))
    if plain_matches:
        return _accum_parse_token(plain_matches[-1].group(0))
    return None


def _accumulator_error_from_record(record: dict) -> Optional[float]:
    """Pull mean accumulator error from a session record's accumulator_probes.

    For each probe, extract the agent's numeric answer using ``_accum_parse_one``
    (anchor-word > $-prefixed-first > last-plain fallback; comma-aware).
    Returns mean |agent - gold| over probes with a parseable answer, or None
    if no probes were scored this session.
    """
    probes = record.get("accumulator_probes", [])
    if not probes:
        return None
    errors = []
    for p in probes:
        gold = p.get("gold_value")
        if gold is None:
            continue
        agent_val = _accum_parse_one(p.get("response_text", "") or "")
        if agent_val is None:
            continue
        errors.append(abs(agent_val - float(gold)))
    if not errors:
        return None
    return sum(errors) / len(errors)


def _retro_recompact_with_careful(runner, session_idx: int, careful_prompt: str,
                                  verbose: bool = True):
    """Retroactive memory re-write action: re-summarize all preserved prior-
    session interaction histories under the careful compaction prompt and
    replace the inner SummarizeStorePolicy's stored summary.

    Source material is `runner._raw_session_histories` (additive attribute on
    S2Runner; default empty). When that list is empty (e.g., scenarios that
    don't preserve histories), this is a no-op.

    The action calls one extra LLM completion at activation time. The result
    replaces the agent's compressed memory wholesale; subsequent sessions
    continue compacting against the new, careful-style summary.
    """
    histories: list[str] = list(getattr(runner, "_raw_session_histories", []) or [])
    if not histories:
        if verbose:
            print(
                f"  [hook][retro] session {session_idx}: no preserved histories, skipping",
                flush=True,
            )
        return
    inner = _find_inner_summarize_store(runner.memory_policy)
    if inner is None:
        if verbose:
            print(
                f"  [hook][retro] session {session_idx}: no SummarizeStorePolicy in chain, skipping",
                flush=True,
            )
        return
    raw_concat = "\n\n".join(histories)
    llm = runner.llm
    messages = [{"role": "user", "content": careful_prompt.format(text=raw_concat)}]
    try:
        if hasattr(llm, "chat_with_usage"):
            resp = llm.chat_with_usage(messages)
            new_memory = resp.text.strip() if hasattr(resp, "text") else str(resp).strip()
        else:
            new_memory = llm.chat(messages).strip()
    except Exception as e:
        if verbose:
            print(f"  [hook][retro] session {session_idx}: retro LLM call failed ({e!r})",
                  flush=True)
        return
    # Apply word-budget truncation if the inner policy enforces one.
    if getattr(inner, "word_budget", None):
        words = new_memory.split()
        if len(words) > inner.word_budget:
            new_memory = " ".join(words[: inner.word_budget])
    inner._memory = new_memory
    if verbose:
        print(
            f"  [hook][retro] session {session_idx}: REWROTE memory from "
            f"{len(histories)} preserved sessions ({len(raw_concat)} chars in -> "
            f"{len(new_memory)} chars out)",
            flush=True,
        )


def make_aggressive_controller_hook(controller: ThresholdController, verbose: bool = True):
    """Hook that drives a controller whose typed-state-promote trigger ALSO
    fires a retroactive recompact action (re-summarize all prior sessions
    under the careful prompt) — the aggressive controller variant.

    Same observation pathway as make_controller_hook, but the
    promote_to_typed_state callback chains in a retro recompact pass that
    repairs damage already baked into compressed memory.
    """
    def hook(runner, session_idx: int):
        record = getattr(runner, "_latest_session_record", None)
        if record is None:
            return
        metrics = {
            "constraint_precision": record.get("constraint_precision"),
            "lag_recall": record.get("lag_recall"),
            "accumulator_error": _accumulator_error_from_record(record),
        }

        def _on_promote_typed_state_aggressive():
            overlay = _find_typed_state_overlay(runner.memory_policy)
            if overlay is not None:
                overlay.set_enabled(True)
                if verbose:
                    print(
                        f"  [hook][controller-aggr] session {session_idx}: "
                        f"ACTIVATED typed-state overlay",
                        flush=True,
                    )
            # Retroactive memory re-write under careful prompt.
            _retro_recompact_with_careful(
                runner, session_idx, CAREFUL_PROMPT, verbose=verbose
            )

        def _on_switch_careful():
            inner = _find_inner_summarize_store(runner.memory_policy)
            if inner is not None:
                inner.prompt_template = CAREFUL_PROMPT
                if verbose:
                    print(
                        f"  [hook][controller-aggr] session {session_idx}: "
                        f"SWITCHED compaction prompt to CAREFUL",
                        flush=True,
                    )

        actions = controller.step(
            session_idx=session_idx,
            metrics=metrics,
            on_promote_typed_state=_on_promote_typed_state_aggressive,
            on_switch_careful=_on_switch_careful,
        )
        if actions and verbose:
            print(
                f"  [hook][controller-aggr] session {session_idx}: dispatched {actions}",
                flush=True,
            )

    return hook


def make_controller_hook(controller: ThresholdController, verbose: bool = True):
    """Hook that drives the runtime controller.

    Reads runner._latest_session_record (surfaced additively by S2Runner)
    for constraint_precision and lag_recall, derives accumulator_error from
    accumulator_probes, then asks the controller to step. On trigger:
      * promote_to_typed_state -> activate TypedStateOverlay (set enabled=True)
      * switch_compaction_policy -> swap inner SummarizeStorePolicy.prompt_template
        to CAREFUL_PROMPT.
    """
    def hook(runner, session_idx: int):
        record = getattr(runner, "_latest_session_record", None)
        if record is None:
            return

        metrics = {
            "constraint_precision": record.get("constraint_precision"),
            "lag_recall": record.get("lag_recall"),
            "accumulator_error": _accumulator_error_from_record(record),
        }

        def _on_promote_typed_state():
            overlay = _find_typed_state_overlay(runner.memory_policy)
            if overlay is None:
                if verbose:
                    print(
                        "  [hook][controller] promote_to_typed_state requested "
                        "but memory_policy is not TypedStateOverlay; skipping",
                        flush=True,
                    )
                return
            overlay.set_enabled(True)
            if verbose:
                print(
                    f"  [hook][controller] session {session_idx}: "
                    f"ACTIVATED typed-state overlay (state={ {k: v['remaining'] for k, v in overlay.state.items()} })",
                    flush=True,
                )

        def _on_switch_careful():
            inner = _find_inner_summarize_store(runner.memory_policy)
            if inner is None:
                if verbose:
                    print(
                        "  [hook][controller] switch_compaction_policy "
                        "requested but no SummarizeStorePolicy in chain; skipping",
                        flush=True,
                    )
                return
            inner.prompt_template = CAREFUL_PROMPT
            if verbose:
                print(
                    f"  [hook][controller] session {session_idx}: "
                    f"SWITCHED compaction prompt to CAREFUL",
                    flush=True,
                )

        actions = controller.step(
            session_idx=session_idx,
            metrics=metrics,
            on_promote_typed_state=_on_promote_typed_state,
            on_switch_careful=_on_switch_careful,
        )
        if actions and verbose:
            print(
                f"  [hook][controller] session {session_idx}: "
                f"observed metrics={metrics}, dispatched {actions}",
                flush=True,
            )
        elif verbose:
            # Light periodic log so we can see the controller observing.
            ae_str = f"{metrics['accumulator_error']:.1f}" if metrics['accumulator_error'] is not None else "n/a"
            cp_str = f"{metrics['constraint_precision']:.2f}" if metrics['constraint_precision'] is not None else "n/a"
            print(
                f"  [hook][controller] session {session_idx}: "
                f"observed accum_err={ae_str}, prec={cp_str} (no trigger)",
                flush=True,
            )

    return hook

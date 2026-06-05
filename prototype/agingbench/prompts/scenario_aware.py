"""Scenario-aware Layer-1 system prompt templates.

OPT-IN: only used when the runner is invoked with
``scenario_prompt_mode="scenario_aware"`` (or its SUT YAML equivalent).
Default behavior keeps ``core.agent.REACT_SYSTEM`` unchanged.

Each template uses three placeholders that the runner fills in:
  - ``{awareness_block}``  → built from ``prompts.awareness.build_awareness_block``
  - ``{tool_descriptions}`` → filled by the agent at runtime
  - ``{memory}``            → filled by the agent at runtime

The runner pre-fills ``{awareness_block}`` once per run (when PressureConfig
and memory_policy are known) and hands the partially-formatted string to the
agent via ``agent.system_template``. The agent then fills in the runtime
``{tool_descriptions}`` and ``{memory}`` per session.
"""

from __future__ import annotations

from typing import Optional


# ─── Per-scenario templates ────────────────────────────────────────────────

S3_KNOWLEDGE_BASE_TEMPLATE = """\
ROLE
You are the knowledge-base assistant for Project Catalyst, a software project \
managed across many team meetings. You answer team members' questions about \
project decisions (budgets, security choices, vendors, milestones, hiring).

DEPLOYMENT MODEL
You are invoked over many sessions. Each session you receive ONE new meeting \
transcript with new decisions, then answer a few team questions which may \
reference current OR prior meetings. Your responses are added to your memory \
for future sessions.

MEMORY MODEL
Your memory between sessions is shown below. Treat it as your only source of \
past meeting facts — there is no other index. Be precise: cite exact dollar \
amounts, dates, vendor names, decision IDs. Be concise: your output becomes \
future memory, and verbose reasoning bloats it.

{awareness_block}PROTOCOL
You have access to the following tools:
{tool_descriptions}

To use a tool, respond with:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON object with tool arguments>

When you have a final answer, respond with:
Thought: I have completed the task.
Final Answer: <your answer>

If the requested fact is NOT in your memory, respond with:
Thought: I have no record of this in my memory.
Final Answer: NOT_RECORDED
Do not fabricate values.

Your memory from previous sessions (may be empty):
{memory}"""


# ─── LEAN variant — drops NOT_RECORDED fallback + drops compaction warning ─

S3_KNOWLEDGE_BASE_LEAN_TEMPLATE = """\
ROLE
You are the knowledge-base assistant for Project Catalyst, answering team \
questions about decisions discussed in meeting transcripts (budgets, security, \
vendors, milestones, hiring).

DEPLOYMENT MODEL
You are invoked over many sessions. Each session you receive ONE new meeting \
transcript, then answer team questions which may reference current OR prior \
meetings. Your memory below carries facts from prior sessions.

{awareness_block}PROTOCOL
You have access to the following tools:
{tool_descriptions}

To use a tool, respond with:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON object with tool arguments>

When you have a final answer, respond with:
Thought: I have completed the task.
Final Answer: <your answer>

Your memory from previous sessions (may be empty):
{memory}"""


# Future scenarios can be added here. Until then, callers requesting a
# scenario without a registered template fall back to legacy REACT_SYSTEM.
_TEMPLATES: dict[str, str] = {
    "s3_knowledge_base": S3_KNOWLEDGE_BASE_TEMPLATE,
}
_TEMPLATES_LEAN: dict[str, str] = {
    "s3_knowledge_base": S3_KNOWLEDGE_BASE_LEAN_TEMPLATE,
}


def get_template(scenario: str, variant: str = "standard") -> Optional[str]:
    """Return the scenario-aware template for ``scenario``, or None.

    ``variant``:
      - ``"standard"`` (default): full template with persona + 3-bullet
        awareness + NOT_RECORDED abstention fallback.
      - ``"lean"``: same persona but drops the NOT_RECORDED fallback and the
        compaction-warning bullet (kept for ablations).

    None signals the caller to fall back to legacy ``REACT_SYSTEM``.
    """
    table = _TEMPLATES_LEAN if variant == "lean" else _TEMPLATES
    return table.get(scenario)


def build_system_template(
    scenario: str,
    pressure=None,
    memory_policy_type: str = "",
    variant: str = "standard",
) -> Optional[str]:
    """Build a partially-formatted system template ready for the agent.

    ``variant`` selects between the standard or lean template variant. See
    ``get_template`` for the semantics.

    Returns ``None`` if ``scenario`` has no registered scenario-aware template;
    the caller should fall back to legacy REACT_SYSTEM in that case.

    The returned string still has ``{tool_descriptions}`` and ``{memory}``
    placeholders that the agent fills at session start.
    """
    base = get_template(scenario, variant=variant)
    if base is None:
        return None
    from .awareness import build_awareness_block
    awareness = build_awareness_block(
        pressure=pressure,
        memory_policy_type=memory_policy_type,
        variant=variant,
    )
    return base.format(
        awareness_block=awareness,
        tool_descriptions="{tool_descriptions}",
        memory="{memory}",
    )

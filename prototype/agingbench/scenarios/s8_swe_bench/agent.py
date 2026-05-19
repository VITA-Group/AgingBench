"""S8 SWE-bench-Aging — agent layer (Phase 3).

Per-session bridge between the longitudinal-aging runner and a real
Tier-2 agent (Claude Code or OpenHands), via the existing AgentAdapter
ABC. Also supports a litellm-only fallback (no agent loop, just an LLM
prompt) for credential-free smoke / CI use.

Per-session protocol the agent sees:

  Working directory:
    /agentmemory/agent_work/session_<N>/        <- agent cwd
    /agentmemory/.aging/notes.md                <- persistent memory
    /agentmemory/agent_work/session_<N>/ISSUE.md <- the new issue text

  The agent is asked to produce two artifacts in its working directory:
    .aging/notes.md          (updated; appended to)
    solution.diff            (unified diff applicable to /testbed)

  Phase 4 picks these up and applies/scores. Phase 3 just gets the
  agent invoked and the artifacts collected.
"""
from __future__ import annotations

import os
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---- request / response shapes -------------------------------------------

@dataclass
class S8AgentRequest:
    session_idx: int
    instance_id: str
    issue_text: str
    chain_role: str
    chain_summary: str
    prior_notes: str
    host_workspace_dir: Path           # agent cwd (subdir of /agentmemory)
    persistent_notes_path: Path        # /agentmemory/.aging/notes.md
    timeout_sec: int = 600
    # Active-probe questions to inject into this session's prompt. Each
    # value is rendered as a numbered question in the agent's user prompt
    # and the agent is asked to write structured answers to
    # `attestation.md`. Keys come from the runner's probe scheduler:
    #   "revision"     — post-bump version check (active revision probe)
    #   "interference" — recall-prior-edit prompt for the second member
    #                    of an interference pair (active interference probe)
    attestation_questions: dict[str, str] = field(default_factory=dict)
    # Phase 15c: when True, the runner did NOT cp /testbed to the
    # workspace — the agent must rely on prior_notes + training priors
    # alone (Tier-2 memory-stress condition). The user prompt is
    # adjusted to inform the agent.
    testbed_disabled: bool = False


@dataclass
class S8AgentResponse:
    success: bool
    adapter_kind: str
    model: str
    raw_response_text: str
    new_notes_text: Optional[str]
    solution_diff_text: Optional[str]
    attestation_text: Optional[str] = None
    files_changed: list = field(default_factory=list)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    duration_sec: Optional[float] = None
    error: Optional[str] = None


# ---- prompt construction --------------------------------------------------

_AGENT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a long-running developer working on the sphinx-doc/sphinx
    codebase across many sessions. Each session you receive ONE GitHub
    issue to address.

    PERSISTENT MEMORY:
      Your only memory across sessions is `.aging/notes.md`. Read it
      first to recall what you have done in prior sessions. Append a
      concise entry for this session before you finish. Keep notes
      short — they will be compressed under memory budget; verbose
      notes get dropped first.

    CODEBASE ACCESS (this session only):
      The current sphinx repository checkout (at the issue's
      base_commit, with the test_patch already applied) is available
      READ-ONLY at `./testbed/`. Read the actual source files there
      to derive correct line numbers and existing context before
      writing your patch. Specifically:
        - The library code is under `./testbed/sphinx/`.
        - Tests are under `./testbed/tests/`.
        - You can `cat ./testbed/sphinx/<file>.py` or use your file
          tools to read it.
      Do NOT edit files under ./testbed/ — your edits there are
      discarded. The agent's only persistent output is
      `solution.diff` (the patch) and `.aging/notes.md` (memory).

    YOUR TASK PER SESSION (use your file-edit tool to do these — text
    responses alone do not count):
      1. Read `.aging/notes.md` (your memory) and `ISSUE.md` (this
         session's task).
      2. Read the relevant files under `./testbed/` to understand
         the actual code state. Use the precise line numbers you
         see there in your unified-diff hunk headers.
      3. EDIT the existing `solution.diff` file in the current
         directory: replace its placeholder content with a unified
         git-apply-able diff (rooted at the repo, paths like
         `a/sphinx/foo.py` / `b/sphinx/foo.py`) that resolves the
         issue. If you cannot find a confident fix, write a short
         "no fix attempted because <reason>" comment — an empty
         `solution.diff` fails verification automatically.
      4. EDIT `.aging/notes.md`: append a short entry describing what
         you did and any new fact about the codebase you learned.
      5. If `attestation.md` exists in the current directory, EDIT it
         to answer every numbered question it contains. One short
         paragraph per question; cite concrete file paths / version
         strings / function names. Empty answers count as a miss.

    Be terse. Memory budget is finite.
""").strip()


def _build_user_prompt(req: S8AgentRequest) -> str:
    # Phase 15c: memory-stress condition. Override the system prompt's
    # default "read ./testbed/" guidance when codebase access is removed.
    testbed_block = ""
    if req.testbed_disabled:
        testbed_block = (
            "\n\n        ## CODEBASE ACCESS\n"
            "        IMPORTANT OVERRIDE: there is NO `./testbed/` directory "
            "this session.\n"
            "        You must work from your `.aging/notes.md` (prior memory) "
            "and your training\n"
            "        priors alone. Write your `solution.diff` based on what "
            "you remember about\n"
            "        the upstream repository's structure. If you have no "
            "memory of the file, do\n"
            "        your best guess but mark it as such in notes.md."
        )
    attest_block = ""
    if req.attestation_questions:
        numbered = []
        for i, (k, q) in enumerate(req.attestation_questions.items(), 1):
            numbered.append(f"        Q{i} [{k}]: {q.strip()}")
        attest_block = (
            "\n\n        ## Attestation questions (answer in `attestation.md`)\n"
            + "\n".join(numbered)
        )
    produce_lines = [
        "        - `.aging/notes.md`     (append a one-paragraph note about this session)",
        "        - `solution.diff`       (unified diff for the upstream repo)",
    ]
    if req.attestation_questions:
        produce_lines.append(
            "        - `attestation.md`     (one short paragraph PER numbered question above)"
        )
    return textwrap.dedent(f"""\
        # Session {req.session_idx} — issue {req.instance_id}

        ## Your prior memory (cat .aging/notes.md)
        {req.prior_notes if req.prior_notes.strip() else "(empty — this is your first session)"}

        ## This session's issue
        {req.issue_text.strip()[:6000]}

        ## Chain context (FYI)
        - role: {req.chain_role}
        - summary: {req.chain_summary}{testbed_block}{attest_block}

        ## Produce
""").strip() + "\n" + "\n".join(produce_lines)


# ---- runner (the bridge) --------------------------------------------------

class S8AgentRunner:
    """Per-session driver. Constructed once per RUN, called once per session."""

    def __init__(self,
                 adapter_kind: str,
                 model: str,
                 host_workspace_root: Path,
                 max_turns: int = 30,
                 api_key_env: Optional[str] = None,
                 memory_window_bytes: Optional[int] = None):
        self.adapter_kind = adapter_kind
        self.model = model
        self.host_workspace_root = Path(host_workspace_root)
        self.max_turns = max_turns
        self.api_key_env = api_key_env
        # Phase 14c: memory_policy.window_bytes from SUT yaml -> bounded-
        # context memory policy. None means unbounded (current default).
        self.memory_window_bytes = memory_window_bytes
        self._adapter = None  # lazily constructed per session (each session needs a fresh cwd)

    def run(self, req: S8AgentRequest) -> S8AgentResponse:
        # 1. Prepare the agent's working directory (a subdir of /agentmemory).
        ws = req.host_workspace_dir
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "ISSUE.md").write_text(req.issue_text, encoding="utf-8")

        # 2. Make `.aging/notes.md` visible at agent's cwd. Use a copy
        #    (not symlink) to avoid tool quirks; sync back after agent runs.
        local_aging = ws / ".aging"
        local_aging.mkdir(parents=True, exist_ok=True)
        local_notes = local_aging / "notes.md"
        if local_notes.is_symlink() or local_notes.is_file():
            local_notes.unlink()
        # Phase 14c: optional memory-window truncation. If `memory_window_bytes`
        # is set on the agent runner, the agent only sees the tail N bytes
        # of prior notes — a bounded-context memory policy. The full notes
        # still persist on disk for the compression probe to score against.
        prior_for_agent = req.prior_notes
        if self.memory_window_bytes is not None and self.memory_window_bytes > 0:
            buf = req.prior_notes.encode("utf-8", errors="replace")
            if len(buf) > self.memory_window_bytes:
                prior_for_agent = (
                    "[notes truncated to last %d bytes by memory policy]\n"
                    % self.memory_window_bytes
                    + buf[-self.memory_window_bytes:].decode("utf-8",
                                                              errors="replace")
                )
        local_notes.write_text(prior_for_agent, encoding="utf-8")

        # 3. Pre-create solution.diff as an empty placeholder. Some
        #    adapters (notably OpenHands) skip creating files from scratch
        #    but reliably edit existing ones. With the file in place the
        #    "edit solution.diff to write your patch" instruction lands.
        solution_path = ws / "solution.diff"
        if not solution_path.exists():
            solution_path.write_text(
                "# replace this comment with your unified-diff patch\n",
                encoding="utf-8",
            )

        # 3b. Pre-create attestation.md only when there are active-probe
        #     questions. The file already containing numbered questions
        #     anchors the edit; same OpenHands-friendliness trick as above.
        attestation_path = ws / "attestation.md"
        attestation_placeholder = None
        if req.attestation_questions:
            qlines = [f"# Attestation — session {req.session_idx} ({req.instance_id})", ""]
            for i, (k, q) in enumerate(req.attestation_questions.items(), 1):
                qlines += [f"## Q{i} [{k}]", q.strip(),
                           "", "(your answer here — REPLACE this line)", ""]
            attestation_placeholder = "\n".join(qlines)
            attestation_path.write_text(attestation_placeholder, encoding="utf-8")

        # 3. Build prompt + invoke adapter.
        adapter = self._build_adapter(ws)
        user_prompt = _build_user_prompt(req)
        full_prompt = f"{_AGENT_SYSTEM_PROMPT}\n\n{user_prompt}"

        import time
        t0 = time.time()
        try:
            resp = adapter.send_message(full_prompt)
            raw_text = getattr(resp, "text", "") or ""
            files_changed = getattr(resp, "files_changed", []) or []
            input_tokens = getattr(resp, "input_tokens", None)
            output_tokens = getattr(resp, "output_tokens", None)
            cost_usd = getattr(getattr(resp, "metadata", {}) or {}, "get",
                               lambda *_: None)("cost_usd")
            duration_sec = round(time.time() - t0, 3)
            success = True
            error = None
        except Exception as exc:                                 # noqa: BLE001
            raw_text = ""
            files_changed = []
            input_tokens = output_tokens = cost_usd = None
            duration_sec = round(time.time() - t0, 3)
            success = False
            error = f"{type(exc).__name__}: {exc}"

        # 4. Collect artifacts.
        new_notes_text = None
        if local_notes.is_file():
            new_notes_text = local_notes.read_text(encoding="utf-8", errors="replace")
            # Sync back to the persistent path.
            req.persistent_notes_path.parent.mkdir(parents=True, exist_ok=True)
            req.persistent_notes_path.write_text(new_notes_text, encoding="utf-8")

        solution_path = ws / "solution.diff"
        solution_diff_text = None
        if solution_path.is_file():
            content = solution_path.read_text(encoding="utf-8", errors="replace")
            # Treat the un-edited placeholder as "no solution produced".
            stripped = content.strip()
            if (stripped
                and stripped != "# replace this comment with your unified-diff patch"
                and not stripped.startswith("# no fix attempted")):
                solution_diff_text = content

        # Attestation read-back: only meaningful if questions were asked.
        attestation_text = None
        if req.attestation_questions and attestation_path.is_file():
            content = attestation_path.read_text(encoding="utf-8", errors="replace")
            # If the agent edited the file at all, surface the content;
            # the probe scorer will detect un-replaced placeholder strings.
            if attestation_placeholder is None or content != attestation_placeholder:
                attestation_text = content

        return S8AgentResponse(
            success=success,
            adapter_kind=self.adapter_kind,
            model=self.model,
            raw_response_text=raw_text[:4000],
            new_notes_text=new_notes_text,
            solution_diff_text=solution_diff_text,
            attestation_text=attestation_text,
            files_changed=files_changed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_sec=duration_sec,
            error=error,
        )

    # ---- adapter factory --------------------------------------------------

    def _build_adapter(self, cwd: Path):
        """Lazily instantiate the underlying agent adapter for this session's cwd."""
        if self.adapter_kind == "claude_code":
            from agingbench.core.adapters.claude_code_agent_adapter import (
                ClaudeCodeAgentAdapter,
            )
            # NOTE: previously passed bare_mode=True for "reproducibility"
            # (skip auto-memory + hooks + keychain reads), but --bare also
            # skips OAuth keychain reads, breaking authentication when the
            # CLI is logged in via `claude /login` instead of env var. Each
            # session already uses an isolated ephemeral cwd, so auto-memory
            # has nothing global to leak in. Reproducibility is preserved.
            return ClaudeCodeAgentAdapter(
                model=self.model,
                cwd=str(cwd),
                max_turns=self.max_turns,
                bare_mode=False,
            )
        if self.adapter_kind == "openhands":
            from agingbench.core.adapters.openhands_adapter import OpenHandsAdapter
            # Override OpenHands' default S7+-shaped notes/ system prompt
            # with the S8 contract (.aging/notes.md + solution.diff).
            return OpenHandsAdapter(
                model=self.model,
                cwd=str(cwd),
                max_turns=self.max_turns,
                api_key_env=self.api_key_env or "OPENAI_API_KEY",
                system_prompt=_AGENT_SYSTEM_PROMPT,
            )
        if self.adapter_kind == "litellm":
            return _LiteLLMAgentBridge(
                model=self.model, cwd=cwd, api_key_env=self.api_key_env,
            )
        raise ValueError(
            f"Unsupported S8 agent adapter_kind: {self.adapter_kind!r}. "
            "Supported: claude_code | openhands | litellm"
        )


# ---- litellm fallback (no real agent loop; one-shot LLM call) ------------

class _LiteLLMAgentBridge:
    """One-shot LLM bridge that mimics AgentAdapter.send_message.

    Useful for credential-cheap smoke runs OR when neither Claude Code
    nor OpenHands is installed. Does not implement tool use — the LLM
    just returns text, which we parse to extract notes + diff.
    """

    def __init__(self, model: str, cwd: Path, api_key_env: Optional[str] = None):
        self.model = model
        self.cwd = Path(cwd)

    def send_message(self, message: str):
        from litellm import completion
        # Ask the LLM to emit a fenced diff + a notes block.
        prompt = (
            message
            + "\n\nFormat your response EXACTLY as:\n"
              "<NOTES>\n... appended notes ...\n</NOTES>\n"
              "<DIFF>\n... unified diff or empty ...\n</DIFF>\n"
        )
        resp = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Parse and write artifacts to cwd.
        notes = _extract_block(text, "NOTES")
        diff = _extract_block(text, "DIFF")
        if notes is not None:
            (self.cwd / ".aging" / "notes.md").parent.mkdir(parents=True, exist_ok=True)
            (self.cwd / ".aging" / "notes.md").write_text(notes, encoding="utf-8")
        if diff:
            (self.cwd / "solution.diff").write_text(diff, encoding="utf-8")
        # Mimic AgentResponse shape duck-typed.
        from types import SimpleNamespace
        return SimpleNamespace(
            text=text, input_tokens=None, output_tokens=None,
            files_changed=[], metadata={},
        )


def _extract_block(text: str, tag: str) -> Optional[str]:
    """Extract content between <TAG>...</TAG> delimiters."""
    open_t = f"<{tag}>"
    close_t = f"</{tag}>"
    i = text.find(open_t)
    if i == -1:
        return None
    j = text.find(close_t, i + len(open_t))
    if j == -1:
        return text[i + len(open_t):].strip()
    return text[i + len(open_t):j].strip()


# ---- factory --------------------------------------------------------------

def build_s8_agent_from_sut(sut_cfg: dict, host_workspace_root: Path) -> S8AgentRunner:
    """Construct an S8AgentRunner from the SUT yaml's `agent` block."""
    agent_cfg = (sut_cfg or {}).get("agent") or {}
    adapter_kind = (agent_cfg.get("adapter") or "litellm").lower()
    model = agent_cfg.get("model") or "claude-haiku-4-5-20251001"
    max_turns = int(agent_cfg.get("max_turns") or 30)
    api_key_env = agent_cfg.get("api_key_env")
    # Phase 14c: optional memory_policy.window_bytes from SUT yaml.
    mp = (sut_cfg or {}).get("memory_policy") or {}
    window = mp.get("window_bytes")
    return S8AgentRunner(
        adapter_kind=adapter_kind,
        model=model,
        host_workspace_root=Path(host_workspace_root),
        max_turns=max_turns,
        api_key_env=api_key_env,
        memory_window_bytes=int(window) if window else None,
    )

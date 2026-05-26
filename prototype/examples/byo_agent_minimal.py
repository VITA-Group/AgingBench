"""byo_agent_minimal.py — Bring-Your-Own-Agent template for Tier-2 scenarios.

AgingBench's Tier-2 scenarios (S5 Self-Planning, S7 Research-Notes, S8 SWE-bench)
score any agent that implements the `AgentAdapter` ABC. The benchmark drives the
loop (sessions, resets, scoring); you drive the agent (tools, memory, planning).

This file is a runnable, copy-paste-able template. It shows:

  1. The minimum AgentAdapter contract (send_message + reset_session).
  2. How to point a SUT YAML at this class via `adapter: { type: custom, ... }`.
  3. A trivial echo-style implementation you can replace with your own agent.

For real wrappers, see:
  agingbench/core/adapters/claude_code_agent_adapter.py
  agingbench/core/adapters/openhands_adapter.py
  agingbench/core/adapters/cursor_agent_adapter.py
  agingbench/core/adapters/codex_adapter.py

To run S7 against this stub:

    pip install "git+https://github.com/VITA-Group/AgingBench.git@v0.3.0#subdirectory=prototype"
    cp examples/byo_agent_minimal.py /path/to/your_pkg/my_agent.py
    cp examples/sut_byo_agent.yaml \\
       agingbench/registry/suts/byo/sut_byo_agent.yaml   # see below
    agingbench run --scenario s7_research_notes \\
                   --sut agingbench/registry/suts/byo/sut_byo_agent.yaml \\
                   --output-dir runs/byo_demo

Companion SUT YAML (drop this in alongside the adapter class):

    sut_id: byo_agent_demo
    description: "Demo: minimal AgentAdapter via type:custom"

    model:
      provider: litellm        # required by some inner code paths; ignored
      model: gpt-4o-mini       # by this stub but kept for shape

    adapter:
      type: custom
      class: your_pkg.my_agent:MyAgent     # importable on PYTHONPATH
      max_turns: 30                        # forwarded as kwarg
      my_api_key_env: MY_API_KEY           # any extra keys are passed too

    seed: 42

After the run completes, the canonical AgingCard is written to
`runs/byo_demo/aging_card.json` and validates against
`agingbench/metrics/aging_card_schema.json`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from agingbench.core.agent_adapter import AgentAdapter, AgentResponse


class MyAgent(AgentAdapter):
    """Replace this stub with your real agent (LangGraph, AutoGen, custom...).

    Required: implement send_message + reset_session.
    Optional: override get_workspace_state and get_memory_text if your agent
    keeps inspectable on-disk state (helps probe-based scoring; many opaque
    agents simply leave them at the defaults).
    """

    def __init__(
        self,
        cwd: str,
        max_turns: int = 30,
        my_api_key_env: Optional[str] = "MY_API_KEY",
        **kwargs,
    ):
        # `cwd` is supplied by the runner — it's an isolated per-run workspace.
        # Treat it as the *only* directory your agent should write to so
        # AgingBench's per-session workspace inspection sees what it expects.
        self.cwd = Path(cwd)
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self._api_key = os.environ.get(my_api_key_env or "") if my_api_key_env else None
        self._turn = 0

    # ---- required ---------------------------------------------------------

    def send_message(self, message: str) -> AgentResponse:
        """Drive your agent for one user message; return its final reply.

        The benchmark calls this once per per-session task. Your agent may
        run multiple internal turns / tool calls in between; only the final
        natural-language reply needs to surface here.
        """
        self._turn += 1
        # ↓↓↓ REPLACE WITH YOUR AGENT ↓↓↓
        reply = f"(stub turn {self._turn}) you said: {message[:200]}"
        # Persist something to cwd so workspace probes have data to read.
        (self.cwd / "agent_log.txt").write_text(
            (self.cwd / "agent_log.txt").read_text() if (self.cwd / "agent_log.txt").exists() else "")
        with (self.cwd / "agent_log.txt").open("a") as f:
            f.write(f"[turn {self._turn}] {message}\n  -> {reply}\n")
        # ↑↑↑ REPLACE WITH YOUR AGENT ↑↑↑
        return AgentResponse(
            text=reply,
            tool_calls=[],          # populate from your tool layer if you want
            input_tokens=0,         # populate from your provider's usage block
            output_tokens=0,
            files_changed=["agent_log.txt"],
            metadata={"adapter": "byo_minimal"},
        )

    def reset_session(self) -> None:
        """Drop ephemeral conversation state; KEEP files on disk.

        AgingBench treats a `reset_session()` as a session boundary: the
        agent must forget anything it was holding in conversation buffers,
        but files / databases / `.aging/` notes are intentionally preserved
        — that's the substrate the next session reads from to test memory.
        """
        self._turn = 0  # forget conversational state; do NOT delete self.cwd/*

    # ---- optional ---------------------------------------------------------
    #
    # If your agent is "opaque" (it persists no inspectable files to `cwd`),
    # you may leave these at their defaults — the run still produces a valid
    # AgingCard. The trade-off: probe scoring then credits only what the
    # agent recites back in its reply, so file-survival probes (S5/S7) and
    # the entity-recall side of S3 will read as more aged than they really
    # are. If your agent does write notes/memos/scratch files somewhere,
    # surface them here.

    def get_workspace_state(self) -> dict:
        """Return list of files in cwd so probe scoring can attribute changes."""
        files = []
        total = 0
        for f in self.cwd.rglob("*"):
            if f.is_file():
                size = f.stat().st_size
                files.append({
                    "path": str(f.relative_to(self.cwd)),
                    "size_bytes": size,
                    "mtime": f.stat().st_mtime,
                })
                total += size
        return {"files": files, "total_bytes": total}

    def get_memory_text(self) -> str:
        """Concatenate text files for keyword-based probe scoring (optional)."""
        parts = []
        for f in sorted(self.cwd.rglob("*")):
            if f.is_file() and f.suffix in (".md", ".txt"):
                try:
                    parts.append(f"=== {f.relative_to(self.cwd)} ===\n{f.read_text(errors='replace')}")
                except OSError:
                    continue
        return "\n\n".join(parts)


# ---- sanity check (run as a script) --------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        agent = MyAgent(cwd=tmp)
        r1 = agent.send_message("Remember my preferred citation style is APA.")
        r2 = agent.send_message("What citation style do I prefer?")
        agent.reset_session()
        r3 = agent.send_message("Anything in your notes?")
        assert isinstance(r1, AgentResponse) and isinstance(r2, AgentResponse)
        assert r3.text.startswith("(stub turn 1)"), "reset_session should zero the turn counter"
        print("OK — adapter contract holds.")
        print("Workspace state:", agent.get_workspace_state())

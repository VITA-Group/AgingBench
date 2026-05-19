"""
agingbench/core/adapters/react_file_adapter.py — Reference self-planning agent.

Wraps a BaseLLM in a ReAct loop with file tools (write_file, read_file, list_files).
Used by S5 (Self-Planning Notebook): the agent manages a workspace directory and the
benchmark observes what it stores.

For models without native tool calling (open-weight, some API models), this adapter
provides file management capability via text-based ReAct parsing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..agent_adapter import AgentAdapter, AgentResponse
from ..agent import ReferenceAgent, strip_thinking
from ..llm import BaseLLM
from ..tools import ToolSpec, ToolRegistry
from ..memory.workspace import WorkspaceMemoryPolicy


_SYSTEM_PROMPT = """You are a long-running assistant with a persistent workspace.

IMPORTANT: Your conversation history will be cleared periodically (every few messages).
To remember information across sessions, you MUST save important facts to your workspace
using the write_file tool. If you don't write it down, you WILL forget it.

Your workspace currently contains:
{workspace_contents}

Available tools:
{tool_descriptions}

Use the ReAct format:
Thought: <your reasoning>
Action: <tool_name>
Action Input: {{"param": "value"}}

Or when you have the final answer:
Thought: <your reasoning>
Final Answer: <your response>
"""


class ReactFileAdapter(AgentAdapter):
    """Reference self-planning agent using ReAct loop + file tools.

    This adapter gives any BaseLLM the ability to manage workspace files.
    The agent decides what to store, how to organize, and when to prune.
    """

    def __init__(
        self,
        llm: BaseLLM,
        workspace_dir: str,
        max_turns: int = 15,
    ):
        self.llm = llm
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self._memory = WorkspaceMemoryPolicy(workspace_dir=workspace_dir)
        self._conversation_history: list[dict] = []
        self._tool_registry = self._build_tools()

    def _build_tools(self) -> ToolRegistry:
        """Create file tools for workspace management."""
        registry = ToolRegistry()

        registry.register(ToolSpec(
            name="write_file",
            version="1.0.0",
            description="Write content to a file in your workspace. Use this to save important information for future reference.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace (e.g., 'notes/budget.txt')"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
            fn=self._write_file,
        ))

        registry.register(ToolSpec(
            name="read_file",
            version="1.0.0",
            description="Read a file from your workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                },
                "required": ["path"],
            },
            fn=self._read_file,
        ))

        registry.register(ToolSpec(
            name="list_files",
            version="1.0.0",
            description="List all files in your workspace with sizes.",
            parameters={
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to list (default: root)", "default": "."},
                },
            },
            fn=self._list_files,
        ))

        return registry

    def _write_file(self, args: dict) -> str:
        # Defensive: handle various argument formats
        path_str = args.get("path") or args.get("filename") or args.get("file") or "notes.txt"
        content = args.get("content") or args.get("text") or args.get("data") or str(args)
        path = self.workspace_dir / path_str
        # Sanitize path (prevent directory traversal)
        try:
            path.resolve().relative_to(self.workspace_dir.resolve())
        except ValueError:
            return f"Error: path '{path_str}' is outside workspace"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path = path / "index.txt"
        path.write_text(content)
        return f"Written {len(content)} chars to {path_str}"

    def _read_file(self, args: dict) -> str:
        path_str = args.get("path") or args.get("filename") or args.get("file") or ""
        if not path_str:
            return "Error: no file path specified"
        path = self.workspace_dir / path_str
        if not path.exists():
            return f"File not found: {path_str}"
        if path.is_dir():
            return f"Error: '{path_str}' is a directory, not a file. Use list_files to inspect directory contents."
        return path.read_text(errors="replace")[:4000]

    def _list_files(self, args: dict) -> str:
        target = self.workspace_dir / args.get("directory", args.get("dir", "."))
        if not target.exists():
            return "Directory not found."
        files = []
        for f in target.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                rel = f.relative_to(self.workspace_dir)
                size = f.stat().st_size
                files.append(f"  {rel} ({size} bytes)")
        if not files:
            return "Workspace is empty."
        return "Files in workspace:\n" + "\n".join(files)

    def send_message(self, message: str) -> AgentResponse:
        """Run one ReAct interaction with file tools."""
        workspace_contents = self._memory.read()
        if not workspace_contents:
            workspace_contents = "(empty workspace)"
        elif len(workspace_contents) > 3000:
            workspace_contents = workspace_contents[:3000] + "\n... (truncated)"

        system_prompt = _SYSTEM_PROMPT.format(
            workspace_contents=workspace_contents,
            tool_descriptions=self._tool_registry.prompt_descriptions(),
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": message})

        tool_calls = []
        files_changed = []
        total_input = 0
        total_output = 0

        # ReAct loop
        for turn in range(self.max_turns):
            response = self.llm.chat_with_usage(messages)
            total_input += response.input_tokens
            total_output += response.output_tokens
            text = response.text

            # Check for Final Answer
            final = ReferenceAgent._parse_final_answer(text)
            if final is not None:
                # Save to conversation history
                self._conversation_history.append({"role": "user", "content": message})
                self._conversation_history.append({"role": "assistant", "content": strip_thinking(final, self.llm)})
                return AgentResponse(
                    text=final,
                    tool_calls=tool_calls,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    files_changed=files_changed,
                    metadata={"turns": turn + 1},
                )

            # Check for Action
            action = ReferenceAgent._parse_action(text)
            if action is not None:
                tool_name, tool_args = action
                tool_spec = self._tool_registry.get(tool_name)
                if tool_spec:
                    result = tool_spec.call(tool_args)
                    tool_calls.append({
                        "tool": tool_name,
                        "input": tool_args,
                        "result": str(result)[:500],
                    })
                    if tool_name == "write_file":
                        files_changed.append(tool_args.get("path", ""))
                    # Add observation to messages
                    messages.append({"role": "assistant", "content": strip_thinking(text, self.llm)})
                    messages.append({"role": "user", "content": f"Observation: {result}"})
                else:
                    messages.append({"role": "assistant", "content": strip_thinking(text, self.llm)})
                    messages.append({"role": "user", "content": f"Observation: Unknown tool '{tool_name}'. Available: {', '.join(t.name for t in self._tool_registry)}"})
            else:
                # No action or final answer — treat as final answer
                self._conversation_history.append({"role": "user", "content": message})
                self._conversation_history.append({"role": "assistant", "content": strip_thinking(text, self.llm)})
                return AgentResponse(
                    text=text,
                    tool_calls=tool_calls,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    files_changed=files_changed,
                    metadata={"turns": turn + 1},
                )

        # Max turns reached
        self._conversation_history.append({"role": "user", "content": message})
        self._conversation_history.append({"role": "assistant", "content": "(max turns reached)"})
        return AgentResponse(
            text="(max turns reached)",
            tool_calls=tool_calls,
            input_tokens=total_input,
            output_tokens=total_output,
            files_changed=files_changed,
            metadata={"turns": self.max_turns},
        )

    def reset_session(self) -> None:
        """Clear conversation history but keep workspace files."""
        self._conversation_history = []

    def get_workspace_state(self) -> dict:
        files = self._memory.file_listing()
        total = sum(f["size_bytes"] for f in files)
        return {"files": files, "total_bytes": total}

    def get_memory_text(self) -> str:
        return self._memory.read()

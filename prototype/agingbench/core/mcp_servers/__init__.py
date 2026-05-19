"""
agingbench.core.mcp_servers — MCP servers we ship for the new Tier-2
scenarios.

The agent's only access to the terminal / filesystem / interpreter is via
these MCP servers. The adapter (in `agingbench.core.adapters.mcp_*`)
translates the runner's per-session task into MCP tool calls.

v1 servers:
  - mcp_shell_server: shell tool (bash/zsh) for S8-T

v1.1 servers (deferred to SWE-bench-Aging):
  - mcp_filesystem_server
  - mcp_interpreter_server
"""

"""S9 synthetic consistency tests.

These tests are INJECTED into a session's f2p list at runtime by the
S9 verifier. They check the agent's solution.diff for adherence to
chain-state design decisions established by prior sessions. The
agent's MEMORY of those decisions (held in .aging/notes.md) is the
only place the design rationale lives — the upstream codebase at the
current session's base_commit does NOT contain prior chain sessions'
edits.

Each test is a pytest function whose body inspects the loaded Django
module(s) the agent edited. The test passes iff the agent's diff
follows the convention established by an earlier chain session.

These tests are the LOAD-BEARING LEVER: their pass/fail directly
affects task pass-rate, so memory of chain conventions becomes
necessary for capability — closing the predictive-validity gap that
naive S8/SWE-bench setups leave open.
"""

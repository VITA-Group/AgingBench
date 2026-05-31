"""Regression tests for the S1 ReAct opt-in wiring (2026-05-30).

Adds `agent_class` to ``S1Runner`` and an optional ``responder`` kwarg to
the three task_validator helpers (run_tasks, run_keyword_probes,
run_trend_probes). When ``agent_class`` is None (default), every code
path the helpers and runner exercise must be byte-identical to the
prior direct-LLM flow — the user's "works the same as the original
version" requirement.

Also covers:
  * ``_EvalTextMemoryProxy`` returns the runner's eval_text from ``.read()``
    so wired-in agents do not silently bypass C2/C3/C4 attribution modes
    by reading ``memory_policy.read()`` independently of the runner.
  * Proxy ``.write()`` and other attributes pass through to the real
    policy, so memory-side side effects are preserved.
  * When a responder IS provided to the helpers, it is invoked instead
    of ``llm.chat`` and the framed prompt embeds the right per-helper
    framing.
"""
from __future__ import annotations

from agingbench.runner.s1_runner import _EvalTextMemoryProxy
from agingbench.scenarios.s1_research_literature.task_validator import (
    run_tasks,
    run_keyword_probes,
    run_trend_probes,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _RecordingLLM:
    """Captures every chat call's messages for byte-identical comparison."""
    def __init__(self, reply: str = "YES of course."):
        self.calls: list[list[dict]] = []
        self.reply = reply

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.reply


class _FakeMemoryPolicy:
    def __init__(self, text: str = "stored-summary"):
        self._text = text
        self.writes: list[tuple] = []
        self.read_calls = 0

    def read(self):
        self.read_calls += 1
        return self._text

    def write(self, text, llm=None):
        self.writes.append((text, llm))

    @property
    def n_writes(self):
        return len(self.writes)


# ---------------------------------------------------------------------------
# 1. Helpers: byte-identical behaviour when responder is omitted
# ---------------------------------------------------------------------------

def test_run_tasks_default_path_is_byte_identical():
    """With responder=None, run_tasks must build the same (system, user)
    message list and call llm.chat exactly once per task — unchanged from
    the prior direct-LLM implementation."""
    llm = _RecordingLLM(reply="YES because the spec says so.")
    tasks = [
        {"task_id": "t1", "constraint_id": "c1",
         "query": "Does the spec require X?",
         "correct_answer": "yes", "correct_value": ""},
        {"task_id": "t2", "constraint_id": "c2",
         "query": "Who approves?",
         "correct_answer": "alice", "correct_value": "alice"},
    ]
    scores, task_m, details = run_tasks(tasks, "memory-text-here", llm)

    # One chat call per task, in order.
    assert len(llm.calls) == 2
    # The system prompt embeds the memory_text and is the same across tasks.
    sys0 = llm.calls[0][0]
    sys1 = llm.calls[1][0]
    assert sys0["role"] == "system"
    assert "memory-text-here" in sys0["content"]
    assert sys0 == sys1
    # User messages are the raw task queries (no extra framing).
    assert llm.calls[0][1] == {"role": "user", "content": "Does the spec require X?"}
    assert llm.calls[1][1] == {"role": "user", "content": "Who approves?"}
    # Scoring still works.
    assert scores == [1, 0]  # "YES..." matches yes; "YES..." misses "alice"
    assert task_m == 0.5
    assert len(details) == 2


def test_run_keyword_probes_default_path_is_byte_identical():
    llm = _RecordingLLM(reply="The result is 42")
    probes = [{"probe_id": "p1", "question": "What is the answer?",
               "keywords": ["42"], "forbidden_keywords": []}]
    scores, details = run_keyword_probes(probes, "mem", llm)
    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "system"
    assert "mem" in llm.calls[0][0]["content"]
    assert llm.calls[0][1] == {"role": "user", "content": "What is the answer?"}
    assert scores == [1]


def test_run_trend_probes_default_path_is_byte_identical():
    llm = _RecordingLLM(reply="The latest value is 7")
    probes = [{"probe_id": "p1", "question": "What is the current X?",
               "keywords": ["7"], "forbidden_keywords": ["3"]}]
    scores, details = run_trend_probes(probes, "mem", llm)
    assert len(llm.calls) == 1
    assert llm.calls[0][1] == {"role": "user", "content": "What is the current X?"}
    assert scores == [1]


# ---------------------------------------------------------------------------
# 2. Helpers: responder takes over when supplied
# ---------------------------------------------------------------------------

def test_run_tasks_responder_replaces_llm_chat():
    """When a responder is provided, run_tasks must NOT call llm.chat and
    must route the framed prompt through the responder instead. The
    framing must restate the compliance-task instructions so the agent
    can format YES/NO answers without seeing the original system prompt."""
    llm = _RecordingLLM()
    received: list[str] = []

    def _responder(framed: str) -> str:
        received.append(framed)
        return "YES because memory says so."

    tasks = [{"task_id": "t1", "constraint_id": "c1",
              "query": "Is X compliant?",
              "correct_answer": "yes", "correct_value": ""}]
    scores, task_m, details = run_tasks(
        tasks, "mem", llm, responder=_responder,
    )
    # llm.chat must not have been touched.
    assert llm.calls == []
    # Responder received the framed prompt with the query embedded.
    assert len(received) == 1
    framed = received[0]
    assert "Is X compliant?" in framed
    assert "YES" in framed and "NO" in framed  # framing intact
    assert "compliance" in framed.lower()
    assert scores == [1]


def test_run_keyword_probes_responder_replaces_llm_chat():
    llm = _RecordingLLM()
    received: list[str] = []

    def _responder(framed: str) -> str:
        received.append(framed)
        return "answer is 99"

    probes = [{"probe_id": "p1", "question": "What is X?",
               "keywords": ["99"], "forbidden_keywords": []}]
    scores, _ = run_keyword_probes(probes, "mem", llm, responder=_responder)
    assert llm.calls == []
    assert "What is X?" in received[0]
    assert scores == [1]


def test_run_trend_probes_responder_replaces_llm_chat():
    llm = _RecordingLLM()
    received: list[str] = []

    def _responder(framed: str) -> str:
        received.append(framed)
        # Framing prompts the agent to cite the latest value
        return "latest is 12"

    probes = [{"probe_id": "p1", "question": "What is the current X?",
               "keywords": ["12"], "forbidden_keywords": ["5"]}]
    scores, _ = run_trend_probes(probes, "mem", llm, responder=_responder)
    assert llm.calls == []
    # Framing tells the agent to cite latest, not original.
    assert "latest" in received[0].lower() or "updated" in received[0].lower()
    assert scores == [1]


# ---------------------------------------------------------------------------
# 3. _EvalTextMemoryProxy
# ---------------------------------------------------------------------------

def test_proxy_read_returns_runner_eval_text():
    """Without the proxy, an agent that calls memory_policy.read() would
    silently bypass the runner's mode-dependent eval_text under
    C2/C3/C4 or cycle 0. The proxy keeps the agent and runner aligned."""
    real = _FakeMemoryPolicy(text="WRONG: lossy summary")
    proxy = _EvalTextMemoryProxy(real, eval_text="RIGHT: oracle gold")
    assert proxy.read() == "RIGHT: oracle gold"
    # Real policy was NOT consulted.
    assert real.read_calls == 0


def test_proxy_write_delegates_to_real_policy():
    real = _FakeMemoryPolicy()
    proxy = _EvalTextMemoryProxy(real, eval_text="snapshot")
    proxy.write("session-3 summary", llm="some-llm")
    assert real.writes == [("session-3 summary", "some-llm")]


def test_proxy_passes_through_other_attributes():
    """n_writes / .compact / arbitrary policy attributes must still reach
    the real policy (read-only attribute forwarding)."""
    real = _FakeMemoryPolicy()
    real.write("a", None)
    real.write("b", None)
    proxy = _EvalTextMemoryProxy(real, eval_text="x")
    assert proxy.n_writes == 2  # passed through


# ---------------------------------------------------------------------------
# 4. S1Runner: agent_class=None preserves no-op default
# ---------------------------------------------------------------------------

def test_s1runner_builds_no_responder_by_default():
    """Most direct check on the no-op-by-default property: with
    agent_class=None the runner method that builds responders returns
    None — so the helpers fall through to their direct-LLM path."""
    from agingbench.runner.s1_runner import S1Runner

    class _NullTracer:
        def log(self, *a, **k):
            pass

    # Minimal instantiation; we only need the responder builder.
    real = _FakeMemoryPolicy()
    runner = S1Runner.__new__(S1Runner)
    runner.memory_policy = real
    runner.llm = _RecordingLLM()
    runner.agent_class = None
    runner.agent_max_turns = 8
    assert runner._build_responder("any", 0) is None


def test_s1runner_builds_responder_when_agent_class_set():
    """When agent_class is set the responder must be a callable. Its
    presence is the gate the helpers branch on; behaviour of the agent
    itself is exercised in S2/S3's ReferenceAgent tests."""
    from agingbench.runner.s1_runner import S1Runner

    class _StubAgent:
        def __init__(self, llm, memory_policy, tools, max_turns):
            self.memory_policy = memory_policy
            self.tools = tools

        def run_session(self, task, session_id=0):
            # Confirms the agent gets the runner's eval_text via the proxy.
            return {"output": self.memory_policy.read()}

    real = _FakeMemoryPolicy(text="WRONG")
    runner = S1Runner.__new__(S1Runner)
    runner.memory_policy = real
    runner.llm = _RecordingLLM()
    runner.agent_class = _StubAgent
    runner.agent_max_turns = 4
    responder = runner._build_responder("RIGHT eval text", cycle=2)
    assert responder is not None
    # Calling the responder runs through the agent, which reads via the
    # proxy — must return the runner's eval_text, NOT the real policy's.
    assert responder("anything") == "RIGHT eval text"

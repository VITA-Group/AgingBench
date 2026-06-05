"""
agingbench/core/memory/base.py — MemoryPolicy abstract base class.

Memory lifecycle:
  M_{t+1} = U(M_t, H_{t+1})   (update operator)
  C_t     = R(M_t, q_t)        (retrieval operator)

The policy is the implementation of U and R.
The benchmark runner calls write() at the end of each session (U),
and read() at the start of each session (R).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class MemoryPolicy(ABC):
    """
    Interface that every memory policy must satisfy.

    Implementations live in no_memory.py, append_only.py, summarize_store.py.
    The runner injects the LLM instance into write() when the policy needs it
    (e.g. summarize_store calls LLM.chat() during compaction).

    For retrieval-based policies (e.g. AppendOnly with vector search), implement
    the optional ``retrieve()`` method to enable retrieval quality metrics
    (precision, recall) without coupling to a specific retriever implementation.
    """

    @abstractmethod
    def read(self, query: Optional[str] = None) -> str:
        """
        Return the current memory content as a plain string.
        This string is injected into the agent's context at session start.
        query: optional natural-language query for retrieval-based policies.
        """

    @abstractmethod
    def write(self, new_content: str, llm=None) -> None:
        """
        Update memory with new_content produced during a session.
        llm: LocalLLM instance, passed when the policy needs generation (e.g. compaction).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset to empty initial state (called between independent runs)."""

    def snapshot(self) -> str:
        """Return raw memory text for oracle ablation or debugging."""
        return self.read()

    def dump_store(self) -> str:
        """Return the FULL contents of the memory store, bypassing any
        retrieval/ranking logic.

        Used by the P2 (oracle retrieval) diagnostic to measure what physically
        survived the write process W.  For single-blob policies (SummarizeStore,
        GrowingHistory) this is equivalent to read().  Override in retrieval-
        based policies (e.g. AppendOnly) to return ALL stored entries, not
        just top-k.
        """
        return self.snapshot()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve ranked chunks from memory (optional, for retrieval-based policies).

        Returns list of {"text": str, "score": float, "id": str} dicts,
        ordered by relevance. Used by retrieval quality metrics (G3-M4/M5)
        without coupling to a specific retriever implementation.

        Default: splits read() output into paragraphs (no ranking).
        Override in retrieval-based policies for proper ranked retrieval.
        """
        text = self.read(query)
        if not text:
            return []
        # Default: return whole memory as single chunk (no retrieval)
        return [{"text": text, "score": 1.0, "id": "full_memory"}]

    def entry_count(self) -> int:
        """Return number of entries in the memory store (for bloat tracking)."""
        return 1 if self.read() else 0


def _resolve_prompt_template(prompt_path: Optional[str],
                             project_root: Optional[Path],
                             default: str) -> str:
    """Load a compaction-prompt template.

    Resolution order:
      1. Package-internal `agingbench/prompts/<basename>` via importlib.resources.
         Works for both wheel installs and source-tree runs.
      2. Legacy `project_root / prompt_path` (e.g. `experiments/prompts/...`),
         preserved for source-tree dev configurations that point outside the
         package.
      3. The provided default template.
    """
    if not prompt_path:
        return default
    # Try package-internal location first
    try:
        from importlib.resources import files
        basename = Path(prompt_path).name
        pkg_file = files("agingbench") / "prompts" / basename
        if pkg_file.is_file():
            return pkg_file.read_text()
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        pass
    # Fallback: legacy project-root relative path (works in source-tree dev)
    if project_root:
        full = Path(project_root) / prompt_path
        if full.is_file():
            return full.read_text()
    return default


def build_memory_policy(policy_cfg: dict, project_root: Optional[Path] = None) -> MemoryPolicy:
    """
    Factory: instantiate a MemoryPolicy from a config dict.

    Expected keys:
        type: "no_memory" | "append_only" | "summarize_store" |
              "growing_history" | "lossy_episodic" | "custom"
        compaction_prompt: (optional) path to prompt template file. Loaded from
            the installed `agingbench/prompts/` package directory (by basename)
            when available; falls back to `project_root / compaction_prompt`
            for source-tree dev configs.

    Parameters
    ----------
    policy_cfg : dict from the SUT YAML's ``memory_policy`` section.
    project_root : repo root for resolving relative prompt paths.
    """
    from .no_memory import NoMemoryPolicy
    from .append_only import AppendOnlyPolicy
    from .summarize_store import SummarizeStorePolicy, COMPACT_MEDIUM

    policy_type = policy_cfg["type"]

    if policy_type == "no_memory":
        return NoMemoryPolicy()

    if policy_type == "append_only":
        # Accept top_k at either memory_policy.top_k (flat) or
        # memory_policy.retriever.top_k (nested). Both shapes appear in the
        # existing SUT registry; the nested form was silently ignored before,
        # so any yaml that set retriever.top_k=N actually ran with the
        # default 5. Flat wins when both are present.
        retriever_cfg = policy_cfg.get("retriever") or {}
        top_k = policy_cfg.get("top_k", retriever_cfg.get("top_k", 5))
        return AppendOnlyPolicy(
            top_k=top_k,
            max_input_tokens=policy_cfg.get("max_input_tokens", 200_000),
        )

    if policy_type == "summarize_store":
        prompt_template = _resolve_prompt_template(
            policy_cfg.get("compaction_prompt"), project_root, COMPACT_MEDIUM)
        return SummarizeStorePolicy(
            prompt_template=prompt_template,
            word_budget=policy_cfg.get("word_budget"),
        )

    if policy_type == "growing_history":
        from .growing_history import GrowingHistoryStorePolicy
        prompt_template = _resolve_prompt_template(
            policy_cfg.get("compaction_prompt"), project_root,
            "Condense to {word_limit} words:\n{text}\n\nCONDENSED:")
        word_budget = policy_cfg.get("word_budget", 300)
        return GrowingHistoryStorePolicy(
            prompt_template=prompt_template, word_budget=word_budget,
        )

    if policy_type == "lossy_episodic":
        from .lossy_episodic import LossyEpisodicPolicy
        prompt_template = _resolve_prompt_template(
            policy_cfg.get("compaction_prompt"), project_root, COMPACT_MEDIUM)
        return LossyEpisodicPolicy(prompt_template=prompt_template)

    # Custom memory policy: type = "custom", class = "my_module:MyPolicy"
    if policy_type == "custom":
        import importlib
        class_spec = policy_cfg.get("class", "")
        if ":" not in class_spec:
            raise ValueError(
                f"Custom memory policy requires 'class' in 'module:ClassName' format, "
                f"got '{class_spec}'"
            )
        module_path, class_name = class_spec.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        # Pass remaining config keys (excluding type/class) as kwargs
        kwargs = {k: v for k, v in policy_cfg.items() if k not in ("type", "class")}
        return cls(**kwargs)

    raise ValueError(f"Unknown memory policy type: {policy_type}")

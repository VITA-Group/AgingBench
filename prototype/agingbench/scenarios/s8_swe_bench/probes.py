"""S8 SWE-bench-Aging — 4-mechanism probes (Phase 4).

Per-session probes are computed from artifacts (no extra LLM calls):
their inputs are the agent's notes.md, its solution.diff, the chain
DAG, and the run's lifecycle events. Aggregated across sessions they
give the four mechanism scores that populate the AgingCard.

Trade-off: probes are LEXICAL / artifact-based, not LLM-judged. That
keeps Phase 4 cost-bounded (no probe-time API calls) and gives a
deterministic signal. LLM-judged probes are a possible Phase 5+
upgrade if these turn out to be too coarse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ---- helpers --------------------------------------------------------------

def _normalise(s: Optional[str]) -> str:
    return (s or "").lower()


def _diff_touches_file(diff_text: Optional[str], filename: str) -> bool:
    """Does a unified diff touch a file matching `filename` (substring)?"""
    if not diff_text or not filename:
        return False
    targets = [
        line for line in diff_text.splitlines()
        if line.startswith(("--- ", "+++ ")) or "diff --git" in line
    ]
    norm = filename.lower()
    return any(norm in t.lower() for t in targets)


def _instance_short_id(instance_id: str) -> str:
    """`sphinx-doc__sphinx-7454` -> `7454`."""
    m = re.search(r"-(\d+)$", instance_id)
    return m.group(1) if m else instance_id


# ---- token-overlap helper (used by compression probe) -------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "is", "to", "of", "in", "on", "for", "with",
    "this", "that", "it", "be", "by", "as", "at", "from", "are", "was", "were",
    "i", "you", "we", "they", "he", "she", "but", "if", "when", "not", "no",
    "so", "do", "does", "did", "can", "could", "would", "should", "will", "may",
    "have", "has", "had", "been", "their", "them", "then", "than", "what", "which",
    "who", "where", "how", "why", "use", "used", "using", "any", "all", "some",
    "one", "two", "three", "more", "less", "very", "just", "only", "also",
    "issue", "bug", "fix", "fixed", "problem", "describe", "expected", "actual",
    "way", "see", "make", "needs", "need", "want", "doc", "docs",
})


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _key_tokens_from_text(text: str, *, top_k: int = 15) -> set[str]:
    """Extract distinctive identifiers from a problem statement.

    Keeps tokens >=3 chars, drops stopwords, returns the top_k most
    frequent unique tokens (lowercased, sorted descending by count).
    Deterministic.
    """
    if not text:
        return set()
    raw = _TOKEN_RE.findall(text)
    counts: dict[str, int] = {}
    for tok in raw:
        low = tok.lower()
        if len(low) < 3 or low in _STOPWORDS:
            continue
        counts[low] = counts.get(low, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {t for t, _ in ranked[:top_k]}


# ---- task-critical facts (memory probes anchored to the gold patch) -----

# Match `def name(`, `class Name(`, `class Name:` in unified-diff hunks.
_DEF_RE = re.compile(r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def _symbols_in_diff(diff_text: Optional[str]) -> set[str]:
    """Extract def/class symbol names from a unified diff.

    Scans:
      - Added (`+`) and context lines (which still belong to the
        patched function) for `def name`/`class Name`.
      - Hunk headers (`@@ -m,n +m,n @@ class Foo:`) whose trailing
        text identifies the enclosing scope of the hunk.
    Ignores pure file-level metadata (---, +++, `diff --git`).
    """
    if not diff_text:
        return set()
    syms: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith(("---", "+++", "diff --git")):
            continue
        if line.startswith("@@"):
            # Format: '@@ -a,b +c,d @@ <enclosing-scope text>'
            parts = line.split("@@", 2)
            body = parts[2] if len(parts) >= 3 else ""
        else:
            body = line[1:] if line[:1] in {"+", "-", " "} else line
        for m in _DEF_RE.finditer(body):
            syms.add(m.group(1))
    return syms


def _tokens_in_text(text: Optional[str]) -> set[str]:
    """Lowercased identifier-shaped tokens from arbitrary text."""
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 3}


def extract_task_critical_facts(problem_statement: Optional[str],
                                 gold_patch: Optional[str],
                                 *, top_k_concepts: int = 15) -> dict[str, set[str]]:
    """Build the set of facts that are definitionally critical to
    solving an issue, derived from the issue's gold patch.

    Returned dict has three buckets (each a set[str], lowercased):
      - files:      paths the gold patch touches (e.g. `sphinx/ext/autodoc.py`).
      - symbols:    def/class names the gold patch defines or modifies.
      - concepts:   problem-statement key tokens that also appear in
                    the gold patch text (the intersection isolates
                    problem-relevant tokens that survive into the fix).

    Falls back gracefully when one of the inputs is missing: if no
    gold_patch is supplied, the `concepts` bucket degrades to the raw
    `_key_tokens_from_text(problem_statement)` (the prior behavior).
    """
    files = {f.lower() for f in _files_in_diff(gold_patch or "")}
    symbols = {s.lower() for s in _symbols_in_diff(gold_patch)}
    if gold_patch:
        concepts = (_key_tokens_from_text(problem_statement or "",
                                          top_k=top_k_concepts)
                    & _tokens_in_text(gold_patch))
    else:
        concepts = _key_tokens_from_text(problem_statement or "",
                                         top_k=top_k_concepts)
    return {"files": files, "symbols": symbols, "concepts": concepts}


def _recall_against_facts(facts: dict[str, set[str]], notes: str) -> dict:
    """Score notes-recall against a task-critical-fact bundle.

    Per-bucket recall: matched / total (with `path tail` and `lowercased
    substring` matching for files — agents legitimately write
    `autodoc.py` instead of `sphinx/ext/autodoc.py`).

    Aggregate recall is the MEAN of non-empty bucket recalls, so it
    stays continuous in [0, 1] even when one bucket is empty.
    """
    notes_lc = notes.lower()
    detail: dict[str, dict] = {}
    bucket_recalls: list[float] = []
    for bucket, members in facts.items():
        if not members:
            continue
        if bucket == "files":
            matched = set()
            for f in members:
                tail = f.rsplit("/", 1)[-1]
                if f in notes_lc or tail in notes_lc:
                    matched.add(f)
        else:
            matched = {t for t in members if t in notes_lc}
        recall = len(matched) / len(members)
        detail[bucket] = {
            "total": len(members),
            "matched": len(matched),
            "recall": round(recall, 4),
        }
        bucket_recalls.append(recall)
    agg = sum(bucket_recalls) / len(bucket_recalls) if bucket_recalls else 0.0
    return {"recall": agg, "by_bucket": detail}


# ---- compression ----------------------------------------------------------

@dataclass
class CompressionProbe:
    """Per-session compression score: notes-recall depth.

    For session t, scan the agent's current notes.md for each prior
    session's key tokens (extracted from its problem_statement).
    recall@K = token-overlap fraction for the prior at gap K.

    Note: `recall_by_gap` values are floats in [0, 1] (continuous);
    `references_to_priors` stays binary for back-compat with the
    older lexical probe consumers.
    """
    session_idx: int
    notes_size_bytes: int
    references_to_priors: dict[int, bool]    # session_idx -> referenced AT ALL?
    recall_rate: float                       # mean of per-prior token-overlaps
    recall_by_gap: dict[int, float]           # gap K -> token-overlap (continuous)

    def to_dict(self) -> dict:
        # `per_prior` is attached dynamically by compute_compression_probe.
        per_prior = getattr(self, "per_prior", [])
        return {
            "session_idx": self.session_idx,
            "notes_size_bytes": self.notes_size_bytes,
            "n_priors_referenced": sum(self.references_to_priors.values()),
            "n_priors_total": len(self.references_to_priors),
            "recall_rate": round(self.recall_rate, 4),
            "recall_by_gap": {k: round(float(v), 4) for k, v in self.recall_by_gap.items()},
            "per_prior": per_prior,
        }


def compute_compression_probe(session_idx: int,
                              notes_text: Optional[str],
                              prior_sessions: list[dict]) -> CompressionProbe:
    """Token-overlap compression probe (task-critical-memory variant).

    `prior_sessions` is the list of dicts {session, instance_id,
    problem_statement?, gold_patch?} for sessions strictly before
    `session_idx`.

    Recall is computed against the bundle of TASK-CRITICAL facts for
    each prior — files, symbols, and concept tokens extracted from
    that prior's gold patch (the patch IS the set of facts the agent
    had to know to solve the issue). See [[extract_task_critical_facts]].

    Per-prior recall = mean of non-empty bucket recalls (files,
    symbols, concepts). The aggregate `recall_rate` is the mean of
    per-prior recalls; `recall_by_gap[K]` is the recall against the
    prior at gap K. Both are continuous in [0, 1].

    Back-compat tiers, in order:
      1. prior has `gold_patch`        -> task-critical facts (files+symbols+concepts)
      2. prior has `problem_statement` -> raw key-token overlap (the prior smooth probe)
      3. neither                       -> lexical instance_id binary match
    """
    notes = _normalise(notes_text)
    notes_bytes = len(notes_text.encode("utf-8")) if notes_text else 0
    refs: dict[int, bool] = {}      # binary back-compat: was the prior referenced AT ALL?
    by_gap: dict[int, float] = {}    # continuous: token-overlap per gap
    per_prior_continuous: list[float] = []
    per_prior_detail: list[dict] = []

    for prior in prior_sessions:
        pi = prior["session"]
        iid = prior["instance_id"]
        statement = prior.get("problem_statement") or ""
        gold_patch = prior.get("gold_patch") or ""
        if gold_patch:
            facts = extract_task_critical_facts(statement, gold_patch)
            scored = _recall_against_facts(facts, notes)
            cont = scored["recall"]
            total = sum(len(v) for v in facts.values())
            matched_count = sum(b["matched"] for b in scored["by_bucket"].values())
            per_prior_detail.append({
                "session": pi, "instance_id": iid, "gap": session_idx - pi,
                "mode": "task_critical",
                "fact_total": total,
                "matched_count": matched_count,
                "recall": round(cont, 4),
                "by_bucket": scored["by_bucket"],
            })
        elif statement:
            key_tokens = _key_tokens_from_text(statement)
            if key_tokens:
                matched = {t for t in key_tokens if t in notes}
                cont = len(matched) / len(key_tokens)
            else:
                matched = set()
                cont = 0.0
            per_prior_detail.append({
                "session": pi, "instance_id": iid, "gap": session_idx - pi,
                "mode": "key_tokens",
                "key_token_count": len(key_tokens),
                "matched_count": len(matched),
                "recall": round(cont, 4),
            })
        else:
            short = _instance_short_id(iid).lower()
            long = iid.lower()
            cont = 1.0 if (long in notes or short in notes) else 0.0
            per_prior_detail.append({
                "session": pi, "instance_id": iid, "gap": session_idx - pi,
                "mode": "lexical_iid",
                "recall": round(cont, 4),
            })
        refs[pi] = cont > 0.0
        by_gap[session_idx - pi] = round(cont, 4)
        per_prior_continuous.append(cont)

    agg = (sum(per_prior_continuous) / len(per_prior_continuous)
           if per_prior_continuous else 1.0)
    probe = CompressionProbe(
        session_idx=session_idx,
        notes_size_bytes=notes_bytes,
        references_to_priors=refs,
        recall_rate=round(agg, 4),
        recall_by_gap=by_gap,
    )
    # Attach the richer per-prior detail as a dynamic attribute that
    # downstream serialisers can pick up.
    probe.per_prior = per_prior_detail        # type: ignore[attr-defined]
    return probe


# ---- interference ---------------------------------------------------------

@dataclass
class InterferenceProbe:
    """Three-flavoured interference signal:

    1) Coarse declared-pairs check (discrete): for each interference
       pair (a, b) declared in the chain, did b's solution.diff touch
       the SAME files as a's gold patch?  resistance = 1 - overlap_rate.

    2) Continuous within-session regression rate: per session, fraction
       of PASS_TO_PASS tests that NOW FAIL after the agent's patch.

    3) ACTIVE recall (S7-spirit): at the LATER member of a declared
       interference pair, the runner asks the agent "what did you change
       in <shared_file> previously?". The agent's `attestation_text`
       section "interference" is scored against the EARLIER member's
       gold-patch symbols + files; high overlap = the agent recalls,
       low overlap = interference / forgetting.

    `regression_rate_*` is the smooth per-session continuous signal;
    `active_recall_*` is the S7-style probe; `resistance` keeps the
    pre-pivot coarse pair signal as a sidecar.
    """
    n_pairs_evaluated: int
    n_pairs_with_overlap: int
    pair_results: list[dict]
    resistance: float                         # 1.0 - file-overlap rate
    regression_rate_trajectory: list[tuple]    # [(session, regression_rate), ...]
    regression_rate_mean: float                # continuous aggregate
    active_recall_per_pair: list[dict]          # per-pair active-probe results
    active_recall_mean: Optional[float]         # mean recall, None if no probes
    # Phase 12: dense per-session active recall trajectory. Each entry is
    # the recall of THAT session's "recall" attestation answer against
    # the corresponding prior session's gold-patch facts.
    per_session_recall_trajectory: list[tuple] = field(default_factory=list)
    per_session_recall_mean: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "n_pairs_evaluated": self.n_pairs_evaluated,
            "n_pairs_with_overlap": self.n_pairs_with_overlap,
            "resistance": round(self.resistance, 4),
            "pair_results": self.pair_results,
            "regression_rate_trajectory": [
                [int(t), round(float(r), 4)] for t, r in self.regression_rate_trajectory
            ],
            "regression_rate_mean": round(self.regression_rate_mean, 4),
            "active_recall_per_pair": self.active_recall_per_pair,
            "active_recall_mean": (round(self.active_recall_mean, 4)
                                   if self.active_recall_mean is not None else None),
            "per_session_recall_trajectory": [
                [int(t), round(float(r), 4)]
                for t, r in self.per_session_recall_trajectory
            ],
            "per_session_recall_mean": (round(self.per_session_recall_mean, 4)
                                        if self.per_session_recall_mean is not None
                                        else None),
        }


def compute_interference_probe(session_results: list[dict],
                               interference_pairs: list[list[str]],
                               gold_patches_by_iid: dict[str, str]) -> InterferenceProbe:
    """Walk each declared interference pair; check for diff overlap.

    Also computes the per-session within-session regression rate (the
    continuous interference signal): fraction of PASS_TO_PASS tests
    that fail after the agent's patch is applied. A session where
    fixing the new issue breaks pre-existing test coverage is exactly
    a cross-task interference event.
    """
    by_iid = {s["instance_id"]: s for s in session_results}
    pair_results: list[dict] = []
    for pair in interference_pairs:
        if len(pair) != 2:
            continue
        a_iid, b_iid = pair[0], pair[1]
        a = by_iid.get(a_iid)
        b = by_iid.get(b_iid)
        if not a or not b:
            continue
        if a["session"] > b["session"]:
            a, b = b, a
            a_iid, b_iid = b_iid, a_iid
        a_gold = gold_patches_by_iid.get(a_iid, "")
        b_diff = (b.get("agent_action") or {}).get("solution_diff_text")
        a_files = _files_in_diff(a_gold)
        b_files = _files_in_diff(b_diff or "")
        overlap = a_files & b_files
        pair_results.append({
            "session_a": a["session"], "session_b": b["session"],
            "a_iid": a_iid, "b_iid": b_iid,
            "a_files_count": len(a_files), "b_files_count": len(b_files),
            "overlap_files_count": len(overlap),
            "overlap": sorted(list(overlap))[:5],
        })
    n_with_overlap = sum(1 for r in pair_results if r["overlap_files_count"] > 0)
    n = len(pair_results)
    overlap_rate = (n_with_overlap / n) if n else 0.0

    # Continuous within-session regression-rate trajectory.
    traj: list[tuple] = []
    for s in sorted(session_results, key=lambda x: x["session"]):
        v = (s.get("agent_action") or {}).get("verification") or {}
        n_p2p = v.get("n_pass_to_pass_total") or 0
        n_p2p_passed = v.get("n_pass_to_pass_passed") or 0
        rate = (n_p2p - n_p2p_passed) / n_p2p if n_p2p > 0 else 0.0
        traj.append((int(s["session"]), float(rate)))
    rate_mean = (sum(r for _, r in traj) / len(traj)) if traj else 0.0

    # ---- ACTIVE RECALL PROBE (S7-style) ----
    # For each pair, find the LATER session's attestation_text and score
    # its "interference" section against the EARLIER session's gold-patch
    # facts (files + symbols + concept tokens). Continuous recall in [0,1].
    active_per_pair: list[dict] = []
    for pr in pair_results:
        later_sess = max(pr["session_a"], pr["session_b"])
        earlier_sess = min(pr["session_a"], pr["session_b"])
        later_iid = pr["b_iid"] if pr["session_b"] == later_sess else pr["a_iid"]
        earlier_iid = pr["a_iid"] if pr["session_a"] == earlier_sess else pr["b_iid"]
        # Find the later session's record.
        later_rec = next(
            (s for s in session_results if int(s["session"]) == later_sess),
            None,
        )
        if not later_rec:
            continue
        aa = later_rec.get("agent_action") or {}
        attest = aa.get("attestation_text") or ""
        # Isolate the interference section if marked.
        attest_section = attest.lower()
        for marker in ("[interference]", "interference]", "## q2", "## q1 [interference]"):
            if marker in attest_section:
                attest_section = attest_section.split(marker, 1)[1]
                break
        # Earlier session's gold patch -> task-critical facts.
        earlier_gold = gold_patches_by_iid.get(earlier_iid, "")
        facts = extract_task_critical_facts(None, earlier_gold)
        if not any(facts.values()):
            continue
        scored = _recall_against_facts(facts, attest_section)
        active_per_pair.append({
            "later_session": later_sess,
            "earlier_session": earlier_sess,
            "later_iid": later_iid,
            "earlier_iid": earlier_iid,
            "active_recall": round(scored["recall"], 4),
            "by_bucket": scored["by_bucket"],
            "attestation_present": bool(attest.strip()),
        })
    active_mean = (sum(p["active_recall"] for p in active_per_pair)
                   / len(active_per_pair)) if active_per_pair else None

    # ---- Phase 13: PER-SESSION recall trajectory (multi-Q aggregation) ----
    # Each session t>=1 has up to K "recall_p<N>" sub-questions naming
    # different prior sessions. We score EACH sub-question (continuous
    # task-critical-fact recall in [0,1]) then aggregate to a mean for
    # that session. With K>=2 the per-session score takes intermediate
    # values rather than being bimodal.
    import re as _re
    per_sess_traj: list[tuple] = []
    sess_to_iid = {int(s["session"]): s["instance_id"] for s in session_results}
    for s in sorted(session_results, key=lambda x: x["session"]):
        t = int(s["session"])
        if t == 0:
            continue
        aa = s.get("agent_action") or {}
        attest = (aa.get("attestation_text") or "").lower()
        qs = aa.get("attestation_questions") or {}
        recall_keys = [k for k in qs if k.startswith("recall")]
        if not recall_keys:
            continue
        sub_scores: list[float] = []
        for rk in recall_keys:
            # Phase 12 back-compat: single key "recall" with no suffix.
            if rk == "recall":
                pattern = r"\[recall\]"
            else:
                # "recall_p5" -> match exactly the bracketed label.
                pattern = re.escape(f"[{rk}]")
            m_sec = re.search(pattern, attest)
            if not m_sec:
                sub_scores.append(0.0)
                continue
            section = attest[m_sec.end():]
            # Truncate at next bracketed-label section header.
            stop = re.search(r"\n\s*##\s*q\d|\n\s*\[", section)
            if stop:
                section = section[:stop.start()]
            # Identify the probed prior session from question text or key.
            q_text = (qs.get(rk) or "").lower()
            m_p = re.search(r"session (\d+)", q_text)
            if m_p:
                prior_t = int(m_p.group(1))
            else:
                m_p = re.search(r"recall_p(\d+)", rk)
                prior_t = int(m_p.group(1)) if m_p else None
            if prior_t is None:
                sub_scores.append(0.0)
                continue
            prior_iid = sess_to_iid.get(prior_t)
            if prior_iid is None:
                sub_scores.append(0.0)
                continue
            gold_p = gold_patches_by_iid.get(prior_iid, "")
            facts = extract_task_critical_facts(None, gold_p)
            if not any(facts.values()):
                continue   # don't count vacuous sub-questions
            sub_scores.append(_recall_against_facts(facts, section)["recall"])
        if sub_scores:
            per_sess_traj.append((t, sum(sub_scores) / len(sub_scores)))
    per_sess_mean = (sum(r for _, r in per_sess_traj) / len(per_sess_traj)
                     if per_sess_traj else None)

    return InterferenceProbe(
        n_pairs_evaluated=n, n_pairs_with_overlap=n_with_overlap,
        pair_results=pair_results,
        resistance=round(1.0 - overlap_rate, 4),
        regression_rate_trajectory=traj,
        regression_rate_mean=rate_mean,
        active_recall_per_pair=active_per_pair,
        active_recall_mean=active_mean,
        per_session_recall_trajectory=per_sess_traj,
        per_session_recall_mean=per_sess_mean,
    )


def _files_in_diff(diff_text: str) -> set[str]:
    """Extract set of file paths the diff touches."""
    files: set[str] = set()
    for line in (diff_text or "").splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            files.add(line[6:].strip())
        elif line.startswith("diff --git"):
            # 'diff --git a/path b/path'
            parts = line.split()
            if len(parts) >= 3 and parts[2].startswith("a/"):
                files.add(parts[2][2:])
    return files


# ---- revision -------------------------------------------------------------

@dataclass
class RevisionProbe:
    """Two-tier revision signal:

    1) EVENT-conditional (S7-style sparse) — for each session preceded
       by a `dep_bump`, score the agent's attestation/notes/diff for
       acknowledgement of the bumped package.
    2) PER-SESSION env trajectory (Phase 12 dense) — at every session
       t>=1 the runner schedules an "env" attestation asking the agent
       to report a chain-canonical pkg's version. Score = whether the
       answer contains a parseable version string. Produces a dense
       trajectory of attentional compliance, smooth in [0, 1].
    """
    n_sessions_post_bump: int
    n_sessions_acknowledging: int
    rate: float
    per_session: list[dict]
    per_session_env_trajectory: list[tuple] = field(default_factory=list)
    per_session_env_mean: Optional[float] = None
    # Phase 14b: belief-revision drift score. For each bumped package,
    # was the agent's reported version BEFORE the bump different from
    # AFTER? If yes (1.0), the agent revised its belief; if no (0.0),
    # stale belief retained. This is the TRUE revision-of-belief signal.
    drift_per_pkg: list[dict] = field(default_factory=list)
    drift_score: Optional[float] = None
    # Phase 17.3: latent-accumulator revision (S7-equivalent for Table 3
    # "accum. err"). Agent reports an incrementing COUNT each session;
    # we extract the integer and compare to ground truth. The mean
    # absolute error across sessions = latent revision-aging signal:
    # 0 = agent perfectly tracks; large = belief drifted from reality.
    latent_per_session: list[dict] = field(default_factory=list)
    latent_abs_err_mean: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "n_sessions_post_bump": self.n_sessions_post_bump,
            "n_sessions_acknowledging": self.n_sessions_acknowledging,
            "version_accuracy": round(self.rate, 4),
            "per_session": self.per_session,
            "per_session_env_trajectory": [
                [int(t), round(float(r), 4)]
                for t, r in self.per_session_env_trajectory
            ],
            "per_session_env_mean": (round(self.per_session_env_mean, 4)
                                     if self.per_session_env_mean is not None
                                     else None),
            "drift_per_pkg": self.drift_per_pkg,
            "drift_score": (round(self.drift_score, 4)
                             if self.drift_score is not None else None),
            "latent_per_session": self.latent_per_session,
            "latent_abs_err_mean": (round(self.latent_abs_err_mean, 4)
                                     if self.latent_abs_err_mean is not None
                                     else None),
        }


_VERSION_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)?(?:[.\-+a-zA-Z0-9]*)?)\b")


def _extract_version_from_text(text: str) -> Optional[str]:
    """Pull the first version-shaped token out of text.

    Accepts X.Y, X.Y.Z, plus optional pre/post/build tag (e.g. 1.4.0rc1,
    2.0.0a1, 1.2.3+local). Returns None on no match.
    """
    if not text:
        return None
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None


def compute_revision_probe(session_results: list[dict],
                           lifecycle_events: list[dict]) -> RevisionProbe:
    """Revision probe with two scoring tiers (best available wins per
    session):

      A. ACTIVE — agent's `attestation_text` for question key "revision"
         is parsed for a version string. If present, ack=True. This is
         the S7-style active probe.
      B. PASSIVE — fall back to lexical scan: does the bumped pkg name
         appear in notes or diff? Used when attestation is absent (older
         runs without active probes) or empty.
    """
    bumps = [e for e in lifecycle_events
             if e.get("event_type") == "dep_bump"]
    per_session: list[dict] = []
    for s in session_results:
        recent_bumps = [b for b in bumps if b["session"] < s["session"]]
        if not recent_bumps:
            continue
        latest = max(recent_bumps, key=lambda b: b["session"])
        detail = (latest.get("detail") or "")
        toks = detail.split()
        pkg = toks[toks.index("--upgrade") + 1].lower() if "--upgrade" in toks else None

        aa = s.get("agent_action") or {}
        attest = (aa.get("attestation_text") or "").lower()
        notes = (s.get("agent_notes_text") or "").lower()
        diff = (aa.get("solution_diff_text") or "").lower()

        # Tier A: active attestation. Look for a version string anywhere
        # under the revision section of attestation.md.
        mode = "passive"
        active_version = None
        if attest:
            # Try to isolate the revision section; otherwise scan whole text.
            section = attest
            for marker in ("## q1 [revision]", "[revision]", "revision]"):
                if marker in attest:
                    section = attest.split(marker, 1)[1]
                    break
            active_version = _extract_version_from_text(section)
            if active_version:
                mode = "active"

        # Tier B: passive lexical fallback.
        passive_hit = bool(pkg) and (pkg in notes or pkg in diff)
        ack = bool(active_version) or passive_hit

        per_session.append({
            "session": s["session"], "bump_pkg": pkg, "ack": ack,
            "bump_at_session": latest["session"],
            "scoring_mode": mode,
            "attested_version": active_version,
            "passive_hit": passive_hit,
        })
    n = len(per_session)
    n_ack = sum(1 for r in per_session if r["ack"])
    rate = (n_ack / n) if n else 0.0

    # ---- Phase 14b: BELIEF-REVISION trajectory (cross-session drift) ----
    # The env probe asks the agent to report a package's installed
    # version at every session. A "revision event" is a dep_bump on
    # that package at session B. For each pkg P, compare the
    # agent's reported versions before and after B:
    #   - If version_before(P) != version_after(P)  -> agent REVISED its
    #     belief (saw the upgrade) -> revision score 1.0
    #   - If version_before(P) == version_after(P)  -> STALE belief
    #     (agent re-reports the old version) -> 0.0
    # Aggregate revision_drift_score across packages.
    revision_drift_per_pkg: list[dict] = []
    if bumps:
        version_reports: dict[tuple[str, int], str] = {}
        for s in sorted(session_results, key=lambda x: x["session"]):
            aa = s.get("agent_action") or {}
            attest = (aa.get("attestation_text") or "").lower()
            if not attest:
                continue
            qs = aa.get("attestation_questions") or {}
            for ek, q_text in qs.items():
                if not ek.startswith("env"):
                    continue
                # Phase 15a: question wording changed to "read
                # installed_versions.txt and report ... for `<pkg>`".
                # Try the new pattern first, then the old "pip show" form.
                pkg_m = (re.search(r"version listed for[^a-z0-9]*`?([A-Za-z0-9_-]+)`?",
                                    (q_text or "").lower())
                         or re.search(r"version[^a-z0-9]*now[^a-z0-9]*listed for[^a-z0-9]*`?([A-Za-z0-9_-]+)`?",
                                       (q_text or "").lower())
                         or re.search(r"pip show ([A-Za-z0-9_-]+)",
                                       (q_text or "").lower()))
                pkg = pkg_m.group(1) if pkg_m else (
                    re.match(r"env_(.+)", ek).group(1).lower()
                    if re.match(r"env_(.+)", ek) else None)
                if not pkg:
                    continue
                m_sec = re.search(re.escape(f"[{ek}]"), attest)
                if not m_sec:
                    continue
                section = attest[m_sec.end():]
                stop = re.search(r"\n\s*##\s*q\d|\n\s*\[", section)
                if stop:
                    section = section[:stop.start()]
                ver_m = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", section)
                if ver_m:
                    version_reports[(pkg, int(s["session"]))] = ver_m.group(1)
        for b in bumps:
            b_sess = int(b["session"])
            detail = b.get("detail") or ""
            toks = detail.split()
            # Guard against malformed details like "pip install --upgrade"
            # (no package after the flag) which would IndexError on the bare
            # `toks[idx + 1]` form.
            pkg = None
            if "--upgrade" in toks:
                idx = toks.index("--upgrade")
                if idx + 1 < len(toks):
                    pkg = toks[idx + 1].lower()
            if not pkg:
                continue
            pre_versions = sorted(
                ((t, v) for (p, t), v in version_reports.items()
                 if p == pkg and t < b_sess),
                key=lambda x: x[0],
            )
            post_versions = sorted(
                ((t, v) for (p, t), v in version_reports.items()
                 if p == pkg and t > b_sess),
                key=lambda x: x[0],
            )
            if not pre_versions or not post_versions:
                continue
            pre_v = pre_versions[-1][1]
            post_v = post_versions[0][1]
            drift = 1.0 if pre_v != post_v else 0.0
            revision_drift_per_pkg.append({
                "pkg": pkg, "bump_session": b_sess,
                "pre_version": pre_v, "post_version": post_v,
                "drift_detected": drift > 0.5, "score": drift,
            })
    revision_drift_score = (
        sum(p["score"] for p in revision_drift_per_pkg)
        / len(revision_drift_per_pkg)
        if revision_drift_per_pkg else None
    )

    # ---- Phase 13: per-session env trajectory (multi-Q + continuous) ----
    # Each session t>=1 has up to K "env_<pkg>" sub-questions. We score
    # each via 3-tier partial credit (probed-pkg-name +1/3, digit-pattern
    # +1/3, full X.Y.Z +1/3) then take the MEAN across sub-questions.
    # Multi-Q averaging produces gradient values in [0, 1].
    env_traj: list[tuple] = []
    for s in sorted(session_results, key=lambda x: x["session"]):
        aa = s.get("agent_action") or {}
        attest = (aa.get("attestation_text") or "").lower()
        qs = aa.get("attestation_questions") or {}
        env_keys = [k for k in qs if k.startswith("env")]
        if not env_keys:
            continue
        if not attest:
            env_traj.append((int(s["session"]), 0.0))
            continue
        sub_scores: list[float] = []
        for ek in env_keys:
            q_text = (qs.get(ek) or "").lower()
            if ek == "env":
                pattern = r"\[env\]"
            else:
                pattern = re.escape(f"[{ek}]")
            m_sec = re.search(pattern, attest)
            if not m_sec:
                sub_scores.append(0.0)
                continue
            section = attest[m_sec.end():]
            stop = re.search(r"\n\s*##\s*q\d|\n\s*\[", section)
            if stop:
                section = section[:stop.start()]
            pkg_match = (re.search(r"version listed for[^a-z0-9]*`?([A-Za-z0-9_-]+)`?", q_text)
                          or re.search(r"pip show ([A-Za-z0-9_-]+)", q_text))
            if pkg_match is None:
                # back-compat: try to parse from key like "env_pluggy"
                m_k = re.match(r"env_(.+)", ek)
                probed_pkg = m_k.group(1).lower() if m_k else None
            else:
                probed_pkg = pkg_match.group(1).lower()
            score = 0.0
            if probed_pkg and probed_pkg in section:
                score += 1.0 / 3.0
            loose_ver = re.search(r"\b\d+(?:\.\d+)+\b", section)
            if loose_ver:
                score += 1.0 / 3.0
                if re.search(r"\b\d+\.\d+\.\d+\b", loose_ver.group(0)):
                    score += 1.0 / 3.0
            sub_scores.append(score)
        if sub_scores:
            env_traj.append((int(s["session"]),
                             round(sum(sub_scores) / len(sub_scores), 4)))
    env_mean = (sum(r for _, r in env_traj) / len(env_traj)) if env_traj else None

    # ---- Phase 17.3: latent-accumulator scoring (S7 m_revision_latent_abs_err)
    # The agent's `accumulator_sessions_completed` answer at session t
    # should be t (sessions completed before current = t). Score =
    # absolute error per session; aggregate is mean abs error.
    latent_records: list[dict] = []
    for s in sorted(session_results, key=lambda x: x["session"]):
        t = int(s["session"])
        aa = s.get("agent_action") or {}
        attest = (aa.get("attestation_text") or "").lower()
        qs = aa.get("attestation_questions") or {}
        if "accumulator_sessions_completed" not in qs:
            continue
        section = ""
        m_sec = re.search(r"\[accumulator_sessions_completed\]", attest)
        if m_sec:
            section = attest[m_sec.end():]
            stop = re.search(r"\n\s*##\s*q\d", section)
            if stop:
                section = section[:stop.start()]
        # Extract the FIRST integer in the agent's answer.
        extracted = None
        m_int = re.search(r"\b(\d{1,3})\b", section)
        if m_int:
            extracted = int(m_int.group(1))
        gold = t   # ground truth: t sessions completed before current
        latent_records.append({
            "session": t,
            "extracted": extracted,
            "gold": gold,
            "abs_err": abs(extracted - gold) if extracted is not None else None,
        })
    nonnull = [r["abs_err"] for r in latent_records if r["abs_err"] is not None]
    latent_mean = (sum(nonnull) / len(nonnull)) if nonnull else None

    return RevisionProbe(
        n_sessions_post_bump=n, n_sessions_acknowledging=n_ack,
        rate=rate, per_session=per_session,
        per_session_env_trajectory=env_traj,
        per_session_env_mean=env_mean,
        drift_per_pkg=revision_drift_per_pkg,
        drift_score=revision_drift_score,
        latent_per_session=latent_records,
        latent_abs_err_mean=latent_mean,
    )


# ---- maintenance ----------------------------------------------------------

@dataclass
class MaintenanceProbe:
    """Pre/post-shock signals around each lifecycle event.

    Two parallel signals are tracked:
      - `pass_rate` (capability): fraction of sessions with verification.passed.
        Saturates at 0 if the agent never passes any task.
      - `memory_recall` (task-critical memory): fraction of prior sessions'
        task-critical facts (files + symbols + concept tokens, see
        [[extract_task_critical_facts]]) present in the agent's notes
        at session t. Stays continuous even at the pass-rate floor —
        an agent that forgets is degrading independent of capability.

    `delta` (the headline) is the pre/post delta on the **memory_recall**
    signal; `pass_rate_delta` is the legacy capability delta retained
    as a sidecar.
    """
    shock_sessions: list[int]
    pre_shock_pass_rate: Optional[float]
    post_shock_pass_rate: Optional[float]
    pass_rate_delta: Optional[float]
    pre_shock_memory_recall: Optional[float]
    post_shock_memory_recall: Optional[float]
    delta: Optional[float]                     # = memory-recall delta
    n_pre_sessions: int
    n_post_sessions: int
    # Phase 13: per-shock (session, delta) trajectory — multi-shock chains
    # produce multiple points instead of just an aggregate.
    per_shock_trajectory: list[tuple] = field(default_factory=list)

    def to_dict(self) -> dict:
        def _r(v):
            return round(v, 4) if v is not None else None
        return {
            "shock_sessions": self.shock_sessions,
            "pre_shock": _r(self.pre_shock_memory_recall),
            "post_shock": _r(self.post_shock_memory_recall),
            "delta": _r(self.delta),
            "pre_shock_pass_rate": _r(self.pre_shock_pass_rate),
            "post_shock_pass_rate": _r(self.post_shock_pass_rate),
            "pass_rate_delta": _r(self.pass_rate_delta),
            "n_pre_sessions": self.n_pre_sessions,
            "n_post_sessions": self.n_post_sessions,
            "per_shock_trajectory": [[int(t), round(float(d), 4)]
                                     for t, d in self.per_shock_trajectory],
        }


def compute_maintenance_probe(session_results: list[dict],
                              lifecycle_events: list[dict],
                              window: int = 2) -> MaintenanceProbe:
    """Use workspace_flush events as the maintenance shock.

    Expects each session dict to optionally carry:
      - `agent_notes_text`: the agent's notes file at that session.
      - `prior_facts`: a list of {session, instance_id, gold_patch,
        problem_statement} for sessions strictly before this one
        (used to derive task-critical facts for memory_recall).

    If `prior_facts` is missing for all sessions, the memory_recall
    fields are None and only the legacy pass_rate delta is reported.
    """
    shocks = sorted({
        int(e["session"]) for e in lifecycle_events
        if e.get("event_type") == "workspace_flush"
    })
    if not shocks:
        return MaintenanceProbe(
            shock_sessions=[],
            pre_shock_pass_rate=None, post_shock_pass_rate=None, pass_rate_delta=None,
            pre_shock_memory_recall=None, post_shock_memory_recall=None,
            delta=None, n_pre_sessions=0, n_post_sessions=0,
        )

    by_idx = {s["session"]: s for s in session_results}

    def passed(s):
        v = (s or {}).get("verification") or {}
        return 1.0 if v.get("passed") else 0.0

    def memory_recall(s):
        priors = s.get("prior_facts") or []
        notes = (s.get("agent_notes_text") or "").lower()
        if not priors:
            return None
        per_prior: list[float] = []
        for p in priors:
            facts = extract_task_critical_facts(
                p.get("problem_statement"), p.get("gold_patch"),
            )
            if not any(facts.values()):
                continue
            per_prior.append(_recall_against_facts(facts, notes)["recall"])
        return (sum(per_prior) / len(per_prior)) if per_prior else None

    pre_pass: list[float] = []
    post_pass: list[float] = []
    pre_mem: list[float] = []
    post_mem: list[float] = []
    per_shock: list[tuple] = []
    for shock in shocks:
        pre = [by_idx[i] for i in range(max(0, shock - window), shock)
               if i in by_idx]
        post = [by_idx[i] for i in range(shock, shock + window)
                if i in by_idx]
        # Per-shock local delta (mem-recall preferred, pass-rate fallback).
        local_pre_mem = [memory_recall(s) for s in pre]
        local_pre_mem = [m for m in local_pre_mem if m is not None]
        local_post_mem = [memory_recall(s) for s in post]
        local_post_mem = [m for m in local_post_mem if m is not None]
        if local_pre_mem and local_post_mem:
            local_delta = (sum(local_post_mem) / len(local_post_mem)
                           - sum(local_pre_mem) / len(local_pre_mem))
        else:
            local_pre_pass = [passed(s) for s in pre]
            local_post_pass = [passed(s) for s in post]
            local_delta = ((sum(local_post_pass) / len(local_post_pass)
                            - sum(local_pre_pass) / len(local_pre_pass))
                           if local_pre_pass and local_post_pass else 0.0)
        per_shock.append((shock, local_delta))
        for s in pre:
            pre_pass.append(passed(s))
            mr = memory_recall(s)
            if mr is not None:
                pre_mem.append(mr)
        for s in post:
            post_pass.append(passed(s))
            mr = memory_recall(s)
            if mr is not None:
                post_mem.append(mr)

    def _avg(xs):
        return (sum(xs) / len(xs)) if xs else None

    pre_pass_avg = _avg(pre_pass)
    post_pass_avg = _avg(post_pass)
    pass_delta = ((post_pass_avg - pre_pass_avg)
                  if pre_pass_avg is not None and post_pass_avg is not None else None)
    pre_mem_avg = _avg(pre_mem)
    post_mem_avg = _avg(post_mem)
    mem_delta = ((post_mem_avg - pre_mem_avg)
                 if pre_mem_avg is not None and post_mem_avg is not None else None)
    return MaintenanceProbe(
        shock_sessions=shocks,
        pre_shock_pass_rate=pre_pass_avg, post_shock_pass_rate=post_pass_avg,
        pass_rate_delta=pass_delta,
        pre_shock_memory_recall=pre_mem_avg, post_shock_memory_recall=post_mem_avg,
        delta=mem_delta if mem_delta is not None else pass_delta,
        n_pre_sessions=len(pre_pass), n_post_sessions=len(post_pass),
        per_shock_trajectory=per_shock,
    )


# ---- Phase 16: orthogonal four-mechanism probes ----------------------------

def _isolate_section(attest_lc: str, key: str) -> str:
    """Return the text following `[key]` up to the next `## Q<N>` header.

    A bare `\\n[...]` is NOT a section break — agent answers sometimes
    repeat the `[MECHANISM=...]` marker inline. Only `## Q<N>` (the
    numbered question header) counts as the section boundary.
    """
    pat = re.escape(f"[{key}]")
    m = re.search(pat, attest_lc)
    if not m:
        return ""
    section = attest_lc[m.end():]
    stop = re.search(r"\n\s*##\s*q\d", section)
    if stop:
        section = section[:stop.start()]
    return section


def compute_orthogonal_probes(
    session_results: list[dict],
    chain: dict,
    seed_manifest: dict,
    lifecycle_events: list[dict],
) -> dict:
    """Phase 16: ORTHOGONAL four-mechanism scoring.

    For each session, score one Q per mechanism:

      compression_clean(t)   : recall of a non-partner, non-flushed prior
      interference_partner(t): recall of the declared partner prior
      revision_<fact_id>(t)  : agent's belief about a changed fact
      maintenance_pre/post   : same-prior recall around a workspace_flush

    Contrasts produced:
      interference_score(t) = recall(partner_p)   − mean(recall(clean_p))
      maintenance_delta(s)  = recall(prior at s+1) − recall(prior at s−1)
      revision_score(t)     = 1 if agent reports `after`, else 0

    The compression trajectory is the raw clean-prior recall (no
    contrast — it IS the baseline).

    Returns a dict with the four trajectories + means.
    """
    sess_to_iid = {int(s["session"]): s["instance_id"] for s in session_results}
    iid_to_session: dict[str, int] = {}
    for s in seed_manifest.get("sessions", []):
        iid_to_session[s["instance_id"]] = int(s["session"])

    # Build gold-patch fact bundles (cached lookup-free; caller already
    # ensures attestation_text is available per session).
    from agingbench.scenarios.s8_swe_bench.verifier import get_instance_metadata
    facts_by_iid: dict[str, dict] = {}
    for iid in sess_to_iid.values():
        if iid not in facts_by_iid:
            meta = get_instance_metadata(iid)
            facts_by_iid[iid] = extract_task_critical_facts(
                meta.get("problem_statement"), meta.get("patch"),
            )

    # Per-session per-mechanism scoring
    compression_clean_traj: list[tuple] = []
    interference_traj: list[tuple] = []
    revision_traj: list[tuple] = []
    maintenance_pairs: dict[int, dict] = {}  # shock_t -> {"pre": v, "post": v}

    state_changes = chain.get("state_changes") or []
    sc_by_fact = {sc["fact_id"]: sc for sc in state_changes}

    for s in sorted(session_results, key=lambda x: int(x["session"])):
        t = int(s["session"])
        aa = s.get("agent_action") or {}
        attest = (aa.get("attestation_text") or "").lower()
        qs = aa.get("attestation_questions") or {}
        if not attest or not qs:
            continue

        # ---- compression_clean (raw baseline) ----
        compression_for_t: list[float] = []
        for k, _q in qs.items():
            m = re.match(r"compression_clean_p(\d+)", k)
            if not m:
                continue
            p = int(m.group(1))
            iid_p = sess_to_iid.get(p)
            if iid_p is None:
                continue
            facts = facts_by_iid.get(iid_p, {})
            if not any(facts.values()):
                continue
            sect = _isolate_section(attest, k)
            compression_for_t.append(
                _recall_against_facts(facts, sect)["recall"]
            )
        if compression_for_t:
            compression_clean_traj.append(
                (t, sum(compression_for_t) / len(compression_for_t))
            )

        # ---- interference_partner: contrast against compression baseline ----
        partner_scores: list[float] = []
        for k in qs:
            m = re.match(r"interference_partner_p(\d+)", k)
            if not m:
                continue
            p = int(m.group(1))
            iid_p = sess_to_iid.get(p)
            if iid_p is None:
                continue
            facts = facts_by_iid.get(iid_p, {})
            if not any(facts.values()):
                continue
            sect = _isolate_section(attest, k)
            partner_scores.append(
                _recall_against_facts(facts, sect)["recall"]
            )
        if partner_scores:
            partner_mean = sum(partner_scores) / len(partner_scores)
            baseline = (sum(compression_for_t) / len(compression_for_t)
                        if compression_for_t else 0.0)
            # interference score = partner_recall − baseline_recall.
            # Positive => agent recalled partner BETTER than baseline
            # (good cross-task recall, low interference). Negative =>
            # agent recalled partner WORSE (interference biting).
            interference_traj.append((t, partner_mean - baseline))

        # ---- revision: state-change belief check ----
        for k in qs:
            m = re.match(r"revision_(.+)", k)
            if not m:
                continue
            fact_id = m.group(1)
            sc = sc_by_fact.get(fact_id)
            if not sc:
                continue
            sect = _isolate_section(attest, k)
            after_str = str(sc.get("after", "")).lower()
            before_str = str(sc.get("before", "")).lower()
            # Strip glob/regex artifacts from after_str for substring search.
            after_key = re.sub(r"[\*\.\(\)\[\]]+", "", after_str).strip()
            before_key = re.sub(r"[\*\.\(\)\[\]]+", "", before_str).strip()
            after_present = bool(after_key) and after_key in sect
            before_present = bool(before_key) and before_key in sect
            # Score 1.0 = clear revision (after present, before absent)
            # Score 0.5 = ambiguous (both present)
            # Score 0.0 = stale (only before present, or nothing)
            if after_present and not before_present:
                score = 1.0
            elif after_present and before_present:
                score = 0.5
            else:
                score = 0.0
            revision_traj.append((t, score, fact_id))

        # ---- maintenance: collect pre/post recall pairs ----
        for k in qs:
            m = re.match(r"maintenance_shock(\d+)_(pre|post)", k)
            if not m:
                continue
            shock_t = int(m.group(1))
            role = m.group(2)
            # Score the recall against session 0's facts.
            iid_p = sess_to_iid.get(0)
            if iid_p is None:
                continue
            facts = facts_by_iid.get(iid_p, {})
            if not any(facts.values()):
                continue
            sect = _isolate_section(attest, k)
            score = _recall_against_facts(facts, sect)["recall"]
            maintenance_pairs.setdefault(shock_t, {})[role] = score

    # Aggregate maintenance into per-shock deltas
    maintenance_traj: list[tuple] = []
    for shock_t, parts in sorted(maintenance_pairs.items()):
        if "pre" in parts and "post" in parts:
            maintenance_traj.append((shock_t, parts["post"] - parts["pre"],
                                     parts["pre"], parts["post"]))

    def _mean(seq):
        return sum(seq) / len(seq) if seq else None

    return {
        "compression_clean_trajectory": [
            [int(t), round(float(v), 4)] for t, v in compression_clean_traj
        ],
        "compression_clean_mean": _mean([v for _, v in compression_clean_traj]),
        "interference_contrast_trajectory": [
            [int(t), round(float(v), 4)] for t, v in interference_traj
        ],
        "interference_contrast_mean": _mean([v for _, v in interference_traj]),
        "revision_per_fact": [
            {"session": int(t), "score": float(v), "fact_id": f}
            for t, v, f in revision_traj
        ],
        "revision_mean": _mean([v for _, v, _ in revision_traj]),
        "maintenance_per_shock": [
            {"shock_session": int(t), "delta": round(float(d), 4),
             "pre_recall": round(float(pre), 4),
             "post_recall": round(float(post), 4)}
            for t, d, pre, post in maintenance_traj
        ],
        "maintenance_mean_delta": _mean([d for _, d, _, _ in maintenance_traj]),
    }

#!/usr/bin/env python3
"""Driver for S4UnifiedRunner (FullReactAgent + tool-only memory) — verify revision aging.

Usage: python scripts/run_s4_unified.py --sut <vllm yaml> --sessions 16
"""
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut", required=True)
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--output", default="")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()

    import yaml
    sut = yaml.safe_load(open(args.sut))
    from agingbench.core.llm import load_llm
    from agingbench.core.memory.base import build_memory_policy
    from agingbench.generators.s4_generator import S4Generator
    from agingbench.cli.loaders import _resolve_pressure
    from agingbench.runner.trace import TraceLogger
    from agingbench.runner.s4_runner_unified import S4UnifiedRunner

    out_dir = Path(args.output) if args.output else (
        PROJECT_ROOT / "experiments" / "results" / "s4_unified" / sut["sut_id"])
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = load_llm(sut["model"])
    memory = build_memory_policy(sut["memory_policy"], PROJECT_ROOT)
    pressure = _resolve_pressure(sut_cfg=sut)
    data = S4Generator(seed=sut.get("seed", 42), pressure=pressure).generate(args.sessions)

    print(f"{'='*64}\nS4UnifiedRunner — FullReactAgent + tool-only memory\n"
          f"SUT={sut['sut_id']}  model={sut['model'].get('model')}  sessions={args.sessions}\n{'='*64}")

    with TraceLogger(str(out_dir / "trace.jsonl")) as tracer:
        runner = S4UnifiedRunner(memory_policy=memory, llm=llm, tracer=tracer,
                                 sut_id=sut["sut_id"], generated_data=data,
                                 agent_max_turns=args.max_turns, search_memory_top_k=args.top_k)
        res = runner.run(n_sessions=args.sessions, seed=sut.get("seed", 42))

    out = {k: v for k, v in res.items() if k not in ("version_curve", "session_results")}
    out["version_accuracy_raw"] = res["version_accuracy_raw"]
    json.dump(out, open(out_dir / "revision_unified.json", "w"), indent=2, default=str)

    print(f"\n{'='*64}")
    print(f"version_accuracy = {res['version_accuracy']}  | search_use_rate = {res['search_use_rate']:.2f}  | n_probes = {res['n_probes']}")
    print("per-session version_accuracy:", [(t, round(s, 2)) for t, s in res["version_accuracy_raw"]])
    dm = res.get("dependency_metrics") or {}
    if isinstance(dm, dict):
        print("chain_recall_by_version_depth:", dm.get("chain_recall_by_version_depth"))
    print(f"Output: {out_dir/'revision_unified.json'}")


if __name__ == "__main__":
    main()

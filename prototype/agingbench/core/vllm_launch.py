"""vLLM server launch profiles + compatibility check for AgingBench SUTs.

The benchmark talks to vLLM over its OpenAI-compatible API (see
``agingbench.core.llm.VLLMAdapter``); the server runs as a SEPARATE process,
typically in its own environment so the heavy ``vllm`` package never enters
the benchmark env. This module centralizes, per model:

  * whether/which ``--reasoning-parser`` to launch with (so thinking arrives
    in ``message.reasoning_content``, separate from the committed answer),
  * the ``enable_thinking`` chat-template default,
  * server flags the model needs (tensor-parallel, dtype, context length),
  * a free-text compatibility note for the awkward cases.

Resolution is substring-match on the served model id (same style as
``model_config._REGISTRY``). This map is intentionally SEPARATE from
``model_config.ModelConfig`` so adding vLLM launch hints never changes the
local_hf scoring path (which would break comparability of existing HF runs).

CLI:
    python -m agingbench.core.vllm_launch serve  Qwen/Qwen3-8B
    python -m agingbench.core.vllm_launch check          # table3_v2 matrix
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VLLMProfile:
    # vLLM --reasoning-parser name (None = launch without one). The parser
    # makes the server split chain-of-thought into reasoning_content.
    reasoning_parser: Optional[str] = None
    # Default for the Qwen3-style chat-template `enable_thinking` toggle. None
    # means "don't pass it" (model has no such toggle).
    enable_thinking: Optional[bool] = None
    # Extra `vllm serve` flags (joined verbatim onto the command).
    extra_serve_args: tuple[str, ...] = ()
    # Minimum vLLM version that ships this model's architecture / parser.
    min_vllm: str = "0.6.0"
    # Human-readable compatibility note (printed by `check`).
    note: str = ""
    # If True, this model is API-hosted and must NOT be served by vLLM.
    api_only: bool = False


# NOTE on parser names: vLLM renamed/added reasoning parsers across releases.
# `deepseek_r1` handles any `<think>…</think>` model (incl. Qwen3); `qwen3`
# is the dedicated Qwen3 parser in newer builds; gpt-oss's harmony parser is
# `openai_gptoss` in recent vLLM (was unavailable before 0.10.1). Run
# `vllm serve --help | grep -A3 reasoning-parser` on the target install to see
# the exact set, and override via the SUT yaml if a name differs.

_PROFILES: list[tuple[str, VLLMProfile]] = [
    # ---- DeepSeek-R1 distills: always reason; emit <think>…</think> ----
    ("deepseek-r1-distill", VLLMProfile(
        reasoning_parser="deepseek_r1",
        enable_thinking=None,  # not a toggle; the model always thinks
        min_vllm="0.7.0",
        note="Qwen2 arch (distill). reasoning_content via deepseek_r1 parser. "
             "Give a generous max_tokens (≥2048) — R1 distills think long.",
    )),
    # ---- Qwen3: native thinking with a chat-template toggle ----
    ("qwen3", VLLMProfile(
        reasoning_parser="qwen3",  # `deepseek_r1` also works on older vLLM
        enable_thinking=False,     # match local_hf registry default (OFF)
        min_vllm="0.8.5",
        note="enable_thinking toggled per-SUT via chat_template_kwargs. "
             "Set model.enable_thinking: true in the yaml to turn reasoning on.",
    )),
    # ---- gpt-oss: harmony format, MXFP4, large; needs recent vLLM + TP ----
    ("gpt-oss", VLLMProfile(
        reasoning_parser="openai_gptoss",
        enable_thinking=None,
        extra_serve_args=("--tensor-parallel-size", "2"),
        min_vllm="0.10.1",
        note="120B / harmony format. Needs vLLM ≥0.10.1 (GptOssForCausalLM + "
             "openai_gptoss parser + MXFP4). Tensor-parallel across GPUs; bump "
             "--tensor-parallel-size to your GPU count. The committed answer is "
             "the post-`assistantfinal` channel → comes back as content.",
    )),
    # ---- Gemma 4: brand-new arch; NO matching vLLM reasoning parser ----
    ("gemma-4", VLLMProfile(
        reasoning_parser=None,  # no gemma <|channel> parser in vLLM
        enable_thinking=None,
        min_vllm="0.11.0",
        note="Gemma4ForConditionalGeneration is NEW — requires a vLLM build "
             "that lists `gemma4` in `vllm serve --help`/supported models "
             "(verify before relying on it). No vLLM reasoning parser for its "
             "<|channel> thought format, so run WITHOUT --reasoning-parser; the "
             "VLLMAdapter then falls back to model_config's channel-strip regex "
             "to clean the answer (thinking is captured client-side). RISKIEST "
             "of the table3_v2 set — confirm the server starts before batch runs.",
    )),
    # ---- Llama 3 Instruct: not a reasoning model ----
    ("llama-3", VLLMProfile(
        reasoning_parser=None,
        enable_thinking=None,
        min_vllm="0.6.0",
        note="Standard LlamaForCausalLM, no thinking channel. Fully supported.",
    )),
    # ---- gpt-4o: API only — never served by vLLM ----
    ("gpt-4o", VLLMProfile(
        api_only=True,
        note="OpenAI API model — keep provider: litellm (model: gpt-4o). "
             "vLLM does not host it.",
    )),
]


def get_vllm_profile(model_id: Optional[str]) -> VLLMProfile:
    mid = (model_id or "").lower()
    for pattern, prof in _PROFILES:
        if pattern in mid:
            return prof
    return VLLMProfile(note="No specific profile; using vLLM defaults "
                            "(no reasoning parser). Verify thinking handling.")


def build_serve_command(
    model_id: str,
    port: int = 8000,
    served_name: Optional[str] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.90,
) -> str:
    """Return the `vllm serve …` command line for a model (string, copy-paste)."""
    prof = get_vllm_profile(model_id)
    if prof.api_only:
        return f"# {model_id} is API-only — do not serve with vLLM ({prof.note})"
    parts = [f"vllm serve {model_id}", f"--port {port}"]
    parts.append(f"--served-model-name {served_name or model_id}")
    if prof.reasoning_parser:
        # vLLM >=0.9 dropped the separate --enable-reasoning flag; passing
        # --reasoning-parser alone enables it. (Older builds needed both.)
        parts.append(f"--reasoning-parser {prof.reasoning_parser}")
    if max_model_len:
        parts.append(f"--max-model-len {max_model_len}")
    parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")
    parts.extend(prof.extra_serve_args)
    return " \\\n    ".join(parts)


# table3_v2 model roster (kept here so `check` reports exactly the paper's set).
TABLE3_V2_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
    "gpt-4o",
]


def _check_cli() -> None:
    print("=" * 78)
    print("vLLM compatibility — table3_v2 models")
    print("=" * 78)
    for m in TABLE3_V2_MODELS:
        prof = get_vllm_profile(m)
        host = "API (litellm)" if prof.api_only else "vLLM"
        rp = prof.reasoning_parser or "(none)"
        et = "" if prof.enable_thinking is None else f"  enable_thinking={prof.enable_thinking}"
        print(f"\n• {m}")
        print(f"    host={host}   reasoning_parser={rp}{et}   min_vllm={prof.min_vllm}")
        print(f"    {prof.note}")
    print("\n" + "=" * 78)


def _serve_cli(argv: list[str]) -> None:
    if not argv:
        print("usage: python -m agingbench.core.vllm_launch serve <model_id> [port]")
        return
    model_id = argv[0]
    port = int(argv[1]) if len(argv) > 1 else 8000
    print(build_serve_command(model_id, port=port))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "serve":
        _serve_cli(sys.argv[2:])
    else:
        _check_cli()

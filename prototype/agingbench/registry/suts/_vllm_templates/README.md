# Serving AgingBench SUTs with vLLM

AgingBench can drive any open-weight SUT through a **vLLM server** over its
OpenAI-compatible API, instead of loading the model in-process with
HuggingFace transformers (`provider: local_hf`).

**Why:** (1) much higher throughput via continuous batching — important for the
long generations of reasoning models; (2) one model load serves every
scenario/seed; (3) when launched with `--reasoning-parser`, the server returns
chain-of-thought in `message.reasoning_content`, *separate* from the committed
answer in `message.content` — so answer extraction is clean per-model instead
of relying on fragile regex stripping.

## Architecture

The vLLM server runs as a **separate process** (its own env — the heavy `vllm`
package never enters the benchmark env). The benchmark talks to it with the
`openai` client via `agingbench.core.llm.VLLMAdapter` (`provider: vllm`).

```
[benchmark env: openai client]  ──HTTP /v1──▶  [vllm env: vllm serve <model>]
        VLLMAdapter                                 --reasoning-parser ...
```

## Same env vs. separate env — which to use

The benchmark **client** only needs `openai` (it never imports `vllm` — see
`VLLMAdapter`). vLLM-**the-server** is what needs the GPU stack. So the only
question is whether the server and client share one Python environment.

**Decision rule:**

| your situation | use |
|---|---|
| mixed: some models via `local_hf` (in-process) **and** some via vLLM | **separate env** |
| API-only (litellm) + occasional vLLM | **separate env** |
| pure-vLLM (never use `local_hf`), want simplest onboarding | single env *(with caveat)* |

**Default = separate env / container.** This package pins `torch>=2.0,<2.11`
(needed for the `local_hf` backend + tokenizers). Recent vLLM wants torch
≥2.11, so `pip install vllm` in the benchmark env can violate that ceiling and
break it (we hit exactly this: vLLM 0.22.0 → torch 2.11/cu13). A separate env
keeps `local_hf` reproducible and lets you swap per-model vLLM versions
(gemma-4 needs a newer build than gpt-oss) without touching the benchmark.

```bash
# benchmark env (client) — no vllm, no extra torch
pip install -e ".[vllm-client]"          # adds just `openai`
# serving env (separate venv or the official vLLM docker image)
python -m venv /path/to/vllm_env && source /path/to/vllm_env/bin/activate
pip install "vllm==0.11.2" --extra-index-url https://download.pytorch.org/whl/cu128
```

**Single-env quickstart (pure-vLLM only).** Works *iff* the vLLM version's
torch satisfies this package's `<2.11` pin — e.g. **vllm 0.11.2 → torch 2.9**.
Simpler, but a later `pip install -U vllm` can silently break the install.
```bash
pip install -e . "vllm==0.11.2" --extra-index-url https://download.pytorch.org/whl/cu128
# run `vllm serve …` and the benchmark from the same env (still client/server over HTTP)
```

> "Same env" never means in-process: you still launch `vllm serve` as a
> separate process and the adapter still talks HTTP. Only the Python env is shared.

## Environments (this project / machine)

| env | prefix | Python | torch | transformers | role |
|---|---|---|---|---|---|
| **benchmark / client** | `/ssd1/jianing/envs/agingbench` | 3.11.15 | 2.5.1+cu124 | 5.4.0 | runs the suite (`local_hf` + `vllm` client). Only needs `openai` for the vLLM path — already installed. |
| **serving** | `/ssd1/jianing/envs/agingbench_gptoss` | 3.13.9 | 2.10.0+cu128 | 5.5.4 | hosts `vllm serve …`. This is a **venv** (not a conda env) → activate with `source .../bin/activate`, *not* `conda activate`. Or just call its `bin/` binaries by absolute path. |

> ⚠️ **Do NOT install vLLM into the `agingbench` env.** vLLM (≥0.10.1, required
> for gpt-oss; ≥0.11 for the new `gemma4` arch) pins torch 2.7–2.9+ and
> transformers 4.x — installing it there would downgrade torch 2.5.1 and
> transformers 5.4, breaking the `local_hf` path **and** the comparability of
> existing HF runs. Install vLLM in the **serving** env only.

**vLLM version used:** **vllm 0.11.2** in a dedicated venv
`/ssd1/jianing/envs/agingbench_vllm` (Python 3.11.15, **torch 2.9.0+cu128 /
CUDA 12.8**, nccl 2.27.5). Verified serving `Qwen/Qwen3-8B` with
`--reasoning-parser qwen3` on driver 550.120 (CUDA 12.4, cu128 runs via
minor-version compat). gpt-oss-120b requires ≥0.10.1; gemma-4 requires a build
whose supported-model list includes `gemma4`.

> **Do NOT `pip install vllm` unpinned on this machine.** The latest (0.22.0)
> pulls a **CUDA-13** torch (2.11.0) that the 550.120 driver cannot run. Pin a
> CUDA-12 release and use the cu128 torch index:
> `pip install "vllm==0.11.2" --extra-index-url https://download.pytorch.org/whl/cu128`

> vLLM ≥0.9 dropped `--enable-reasoning`; pass `--reasoning-parser <name>` alone.
> Registered parsers in 0.11.2 include `qwen3`, `deepseek_r1`, `openai_gptoss`.

Hardware: 3× NVIDIA RTX A6000 (48 GB), driver 550.120.

## Workflow

```bash
# 1. (SERVING env — agingbench_gptoss, NOT agingbench) install vLLM once
/ssd1/jianing/envs/agingbench_gptoss/bin/pip install "vllm>=0.10.1"
# then record the resolved version in this README:
/ssd1/jianing/envs/agingbench_gptoss/bin/pip show vllm | grep -i version

# 2. (SERVING env) launch the server for one model.
#    `vllm_launch serve` prints the exact command (run it from the benchmark env):
/ssd1/jianing/envs/agingbench/bin/python -m agingbench.core.vllm_launch serve Qwen/Qwen3-8B
/ssd1/jianing/envs/agingbench_gptoss/bin/vllm serve Qwen/Qwen3-8B --port 8000 \
    --served-model-name Qwen/Qwen3-8B --enable-reasoning --reasoning-parser qwen3

# 3. (BENCHMARK env — agingbench) point a SUT yaml at it and run as usual
/ssd1/jianing/envs/agingbench/bin/python scripts/run_s2.py \
    --sut agingbench/registry/suts/_vllm_templates/qwen3_8b_lossy_vllm.yaml
```

`python -m agingbench.core.vllm_launch check` prints the full table3_v2
compatibility matrix.

## table3_v2 compatibility matrix

| model | host | reasoning parser | enable_thinking | notes |
|---|---|---|---|---|
| meta-llama/Meta-Llama-3-8B-Instruct | vLLM | — | n/a | standard Llama, no thinking. Fully supported. |
| Qwen/Qwen3-8B | vLLM | `qwen3` | toggle (default off) | per-SUT via `chat_template_kwargs`. |
| Qwen/Qwen3-14B | vLLM | `qwen3` | toggle (default off) | same. |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | vLLM | `deepseek_r1` | always on | give max_tokens ≥2048. |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | vLLM | `deepseek_r1` | always on | same. |
| openai/gpt-oss-120b | vLLM | `openai_gptoss` | always on | needs vLLM ≥0.10.1, MXFP4, tensor-parallel. answer = post-`assistantfinal`. |
| google/gemma-4-31b-it | vLLM | **none** | n/a | `Gemma4ForConditionalGeneration` is new — verify your vLLM build supports `gemma4`. No reasoning parser → adapter strips `<|channel>` client-side. **Riskiest; smoke-test first.** |
| gpt-4o | **API (litellm)** | — | n/a | keep `provider: litellm`; vLLM does not host it. |

## Caveats (read before re-running for the paper)

- **Backend switch changes the numbers.** vLLM kernels/batching differ from HF
  transformers; even at `temperature=0` results shift slightly and are *not*
  directly comparable to existing `local_hf` runs. Re-run the whole Tier-1 grid
  on vLLM together; note the backend in the paper. Don't mix backends within a
  table.
- **Parser names vary by vLLM version.** `vllm serve --help | grep reasoning`
  shows the set your install ships. Older builds: use `deepseek_r1` for Qwen3;
  gpt-oss needs ≥0.10.1.
- **max_tokens still matters.** It's the *shared* thinking+answer budget — keep
  it generous for reasoning models or the answer gets truncated. With a parser,
  truncation is detectable: empty `content` + non-empty `reasoning_content`.

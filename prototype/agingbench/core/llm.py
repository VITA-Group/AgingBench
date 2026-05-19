"""
agingbench/baselines/llm.py — LLM abstraction layer (PDF §6.1.1, §6.2).

§6.1.1 Adapter layers: benchmark logic must not couple to a single provider.
  → BaseLLM ABC defines the contract; LocalLLM and LiteLLMAdapter implement it.

§6.2 Provider gateway: LiteLLM for unified model calling, retries, and provider
abstraction. LiteLLMAdapter wraps litellm.completion() so any model in the
LiteLLM registry (OpenAI, Anthropic, Google, Together, local) can be used by
swapping one YAML field.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Optional

# HuggingFace cache location. Resolution order:
#   1. HF_HOME already set in environment (honored verbatim)
#   2. AGINGBENCH_HF_HOME env var (project-specific override)
#   3. ~/.cache/huggingface (HuggingFace's documented default)
#
# Previously this module set HF_HOME to a hardcoded developer-machine path,
# which broke fresh `pip install agingbench` users with an opaque OSError
# whenever transformers tried to write to a non-existent directory. The new
# default falls back to the platform-standard cache location.
if not os.environ.get("HF_HOME"):
    _override = os.environ.get("AGINGBENCH_HF_HOME")
    if _override:
        os.environ["HF_HOME"] = _override
    else:
        os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")


# ------------------------------------------------------------------ contract

class ChatResponse(NamedTuple):
    text: str
    input_tokens: int
    output_tokens: int
    thought: str = ""


class BaseLLM(ABC):
    """
    Provider-agnostic LLM interface (§6.1.1 adapter layer).

    All agent code and memory policies depend only on BaseLLM — never on a
    concrete implementation. Swapping providers requires only a YAML change.
    """

    # Cost-tracking hook. Runners assign these after constructing the LLM:
    #   self.llm.tracer = self.tracer
    #   self.llm.current_cycle = cycle   # set per-cycle to tag trace events
    # Subclasses call self._log_llm_call(in_tok, out_tok) at the end of every
    # chat_with_usage; if tracer is None it's a no-op. This ensures the
    # AgingCard cost_and_efficiency block sums across ALL llm calls (probes +
    # compaction + adapter), not just the compaction-time calls a few runners
    # happened to explicitly log.
    tracer = None            # type: ignore  # filled in by the runner
    current_cycle: int = 0

    def _log_llm_call(self, input_tokens: int, output_tokens: int) -> None:
        if self.tracer is None:
            return
        # Pick a JSON-serialisable string for `model` and `provider`. Different
        # backends name the model attribute differently: LocalLLM has
        # `model_id` (str) + `model` (the torch Module, NOT a string);
        # LiteLLMAdapter has `model` (str). Pick `model_id` first to avoid
        # accidentally serialising a torch.nn.Module.
        model_name = (
            getattr(self, "_model_id", "")
            or getattr(self, "model_id", "")
            or (getattr(self, "model", "") if isinstance(getattr(self, "model", ""), str) else "")
        )
        provider_name = (
            getattr(self, "_provider", "")
            or getattr(self, "provider", "")
            or ""
        )
        try:
            self.tracer.log_llm_call(
                model=str(model_name),
                provider=str(provider_name),
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                cycle=int(self.current_cycle or 0),
            )
        except Exception:
            # Cost tracing is best-effort; never let a logging error break the run.
            pass

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """Send a chat-formatted request; return the response text."""

    @abstractmethod
    def chat_with_usage(self, messages: list[dict]) -> ChatResponse:
        """Same as chat() but also returns input/output token counts."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for a plain string."""


# ------------------------------------------------------------------ local HF

@dataclass
class LocalLLM(BaseLLM):
    """
    HuggingFace transformers backend (open-weight models, local inference).
    Applies the model's chat template for Instruct variants.

    Defaults to greedy decoding (§6.1.4). Sampling params (temperature, top_p,
    top_k) can be set explicitly here, or will be pulled from
    ``model_config.get_model_config(model_id)`` when a vendor publishes
    required generation settings (e.g. Gemma 4).
    """
    model_id: str
    tok: object       # AutoTokenizer
    model: object     # AutoModelForCausalLM
    max_new_tokens: int = 700
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    # When set (True/False), overrides the registry's enable_thinking flag.
    # None means: fall back to get_model_config(model_id).enable_thinking.
    enable_thinking: Optional[bool] = None
    # Sidecar: last thought-channel content captured by the tokenizer.
    # Read by tracers; never fed back into conversation history.
    last_thought: str = ""

    @classmethod
    def load(
        cls,
        model_id: str,
        max_new_tokens: int = 700,
        torch_dtype: Optional[str] = None,
        quantization: dict | None = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> "LocalLLM":
        """Load a HuggingFace causal LM with optional dtype and quantization."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Determine dtype (prefer explicit config when provided)
        model_lower = model_id.lower()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if isinstance(torch_dtype, str) and torch_dtype.strip().lower() in dtype_map:
            dtype = dtype_map[torch_dtype.strip().lower()]
        elif any(x in model_lower for x in ["gemma-2", "gemma-3", "gemma-4", "qwen3.5", "qwen3_5", "qwen2.5-72"]):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Optional BitsAndBytes quantization for large models.
        quant_config = None
        if quantization:
            from transformers import BitsAndBytesConfig
            qtype = quantization.get("type", "bnb_4bit")
            if qtype == "bnb_4bit":
                compute_dtype_name = quantization.get("compute_dtype", "float16")
                compute_dtype = torch.bfloat16 if "bfloat" in compute_dtype_name else torch.float16
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=quantization.get("quant_type", "nf4"),
                    bnb_4bit_use_double_quant=quantization.get("double_quant", True),
                )
            elif qtype == "bnb_8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

        load_kwargs = dict(
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if quant_config is not None:
            # When quantizing, let BnB pick the weight dtype; compute dtype is set above.
            # Force GPU-only placement: BnB 4-bit raises ValueError if any layers
            # spill to CPU ("Some modules are dispatched on the CPU or the disk").
            # Set max_memory to prevent CPU offload.
            load_kwargs["quantization_config"] = quant_config
            load_kwargs.pop("torch_dtype", None)
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                per_gpu = quantization.get("max_memory_per_gpu", "46GB")
                load_kwargs["max_memory"] = {
                    i: per_gpu for i in range(n_gpus)
                }
                # Explicitly exclude CPU to prevent BnB offload error
                load_kwargs["max_memory"]["cpu"] = "0GB"

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()
        return cls(
            model_id=model_id,
            tok=tok,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def _fold_system_messages(messages: list[dict]) -> list[dict]:
        """Fold system messages into the first user message for models
        that don't support the system role (e.g., Gemma)."""
        system_parts = []
        other = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                other.append(m)
        if not system_parts:
            return messages
        prefix = "\n\n".join(system_parts)
        if other and other[0]["role"] == "user":
            other[0] = {**other[0], "content": prefix + "\n\n" + other[0]["content"]}
        else:
            other.insert(0, {"role": "user", "content": prefix})
        return other

    def chat_with_usage(self, messages: list[dict]) -> ChatResponse:
        import torch
        from transformers import GenerationConfig
        from .model_config import get_model_config

        cfg = get_model_config(self.model_id)
        # SUT YAML override wins; None falls back to registry default.
        thinking_on = (
            self.enable_thinking if self.enable_thinking is not None
            else cfg.enable_thinking
        )
        tpl_kwargs = {}
        if thinking_on:
            tpl_kwargs["enable_thinking"] = True

        try:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **tpl_kwargs
            )
        except Exception:
            # Model doesn't support system role — fold into first user message
            folded = self._fold_system_messages(messages)
            prompt = self.tok.apply_chat_template(
                folded, tokenize=False, add_generation_prompt=True, **tpl_kwargs
            )
        enc = self.tok(prompt, return_tensors="pt", truncation=False).to(self.model.device)
        n_input = enc["input_ids"].shape[-1]

        # Safety: truncate if approaching context limit
        max_ctx = getattr(self.model.config, "max_position_embeddings", 8192)
        if n_input + self.max_new_tokens > max_ctx - 64:
            allowed = max_ctx - self.max_new_tokens - 128
            enc = {k: v[:, -allowed:] for k, v in enc.items()}
            n_input = enc["input_ids"].shape[-1]

        eos_ids = []
        if self.tok.eos_token_id is not None:
            eos_ids.append(self.tok.eos_token_id)
        # Gemma-style chat templates terminate turns with a dedicated token.
        # If we only stop on <eos>, generation can continue producing control
        # tokens that decode to empty after stripping specials.
        end_of_turn_id = None
        if hasattr(self.tok, "get_vocab"):
            vocab = self.tok.get_vocab()
            if "<end_of_turn>" in vocab:
                end_of_turn_id = vocab["<end_of_turn>"]
        else:
            candidate_eot = self.tok.convert_tokens_to_ids("<end_of_turn>")
            unk_id = getattr(self.tok, "unk_token_id", None)
            if (
                isinstance(candidate_eot, int)
                and candidate_eot >= 0
                and (unk_id is None or candidate_eot != unk_id)
            ):
                end_of_turn_id = candidate_eot

        if end_of_turn_id is not None:
            eos_ids.append(end_of_turn_id)

        unk_id2 = getattr(self.tok, "unk_token_id", None)
        for tok_str in cfg.extra_eos_tokens:
            tid = self.tok.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0 and tid != unk_id2:
                eos_ids.append(tid)

        if not eos_ids:
            eos_ids = None
        elif len(eos_ids) == 1:
            eos_ids = eos_ids[0]
        else:
            eos_ids = sorted(set(eos_ids))

        pad_id = self.tok.pad_token_id
        suppress_pad = (
            isinstance(pad_id, int)
            and pad_id >= 0
            and not (
                (isinstance(eos_ids, int) and pad_id == eos_ids)
                or (isinstance(eos_ids, list) and pad_id in eos_ids)
            )
        )

        temperature = self.temperature if self.temperature is not None else cfg.temperature
        top_p = self.top_p if self.top_p is not None else cfg.top_p
        top_k = self.top_k if self.top_k is not None else cfg.top_k
        do_sample = temperature is not None and temperature > 0

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=eos_ids,
            bad_words_ids=[[pad_id]] if suppress_pad else None,
        )
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
        gen_cfg = GenerationConfig(**gen_kwargs)
        with torch.no_grad():
            out = self.model.generate(**enc, generation_config=gen_cfg)
        new_tokens = out[0][n_input:]
        thought = ""
        if cfg.output_strip_patterns or cfg.thought_capture_pattern:
            # Some vendors' control tokens (e.g. Gemma 4's <|channel>…<channel|>)
            # leak their content as plain text when specials are skipped. Decode
            # with specials so we can capture the thought-channel separately,
            # regex out the full marker block, then drop any remaining
            # special-token strings.
            import re as _re
            raw = self.tok.decode(new_tokens, skip_special_tokens=False)
            if cfg.thought_capture_pattern:
                thoughts = _re.findall(
                    cfg.thought_capture_pattern, raw, flags=_re.DOTALL
                )
                thought = "\n---\n".join(t.strip() for t in thoughts if t.strip())
            for pat in cfg.output_strip_patterns:
                raw = _re.sub(pat, "", raw, flags=_re.DOTALL)
            for s in self.tok.all_special_tokens:
                raw = raw.replace(s, "")
            text = raw.strip()
        else:
            text = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        if not text and len(new_tokens) > 0:
            # Fallback for models whose chat control tokens can consume the
            # entire decoded output when skip_special_tokens=True.
            raw = self.tok.decode(new_tokens, skip_special_tokens=False)
            text = (
                raw.replace("<start_of_turn>model", "")
                .replace("<end_of_turn>", "")
                .replace("<eos>", "")
                .replace("<pad>", "")
                .strip()
            )
        self.last_thought = thought
        self._log_llm_call(n_input, len(new_tokens))
        return ChatResponse(
            text=text,
            input_tokens=n_input,
            output_tokens=len(new_tokens),
            thought=thought,
        )

    def chat(self, messages: list[dict]) -> str:
        return self.chat_with_usage(messages).text

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text))


# ------------------------------------------------------------------ LiteLLM

class LiteLLMAdapter(BaseLLM):
    """
    LiteLLM backend (§6.2 provider gateway).

    Supports any provider in the LiteLLM registry via a single model string:
      "gpt-4o", "claude-3-7-sonnet-20250219", "gemini/gemini-1.5-pro",
      "together_ai/meta-llama/Llama-3-70b-chat-hf", etc.

    Install: pip install litellm
    Configure API keys via environment variables (OPENAI_API_KEY, etc.)
    or pass api_key / api_base explicitly.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 700,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        try:
            import litellm as _litellm
            self._litellm = _litellm
        except ImportError as e:
            raise ImportError(
                "LiteLLMAdapter requires litellm: pip install litellm"
            ) from e
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.last_thought = ""

    def chat_with_usage(self, messages: list[dict]) -> ChatResponse:
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        import time
        for attempt in range(5):
            try:
                resp = self._litellm.completion(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                in_tok = getattr(usage, "prompt_tokens", 0)
                out_tok = getattr(usage, "completion_tokens", 0)
                self._log_llm_call(in_tok, out_tok)
                return ChatResponse(
                    text=text.strip(),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 4:
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40s
                    print(f"[rate limit] waiting {wait}s (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    raise

    def chat(self, messages: list[dict]) -> str:
        return self.chat_with_usage(messages).text

    def count_tokens(self, text: str) -> int:
        try:
            return self._litellm.token_counter(model=self.model, text=text)
        except Exception:
            return len(text) // 4  # rough fallback


# ------------------------------------------------------------------ factory

def load_llm(cfg: dict) -> BaseLLM:
    """
    Instantiate the correct BaseLLM subclass from a SUT config dict.

    YAML examples:
      provider: local_hf
        model_id: meta-llama/Meta-Llama-3-8B-Instruct

      provider: litellm
        model: gpt-4o
        temperature: 0.0
    """
    provider = cfg.get("provider", "local_hf")
    if provider == "local_hf":
        return LocalLLM.load(
            cfg["model_id"],
            max_new_tokens=cfg.get("max_new_tokens", 700),
            torch_dtype=cfg.get("torch_dtype"),
            quantization=cfg.get("quantization"),
            temperature=cfg.get("temperature"),
            top_p=cfg.get("top_p"),
            top_k=cfg.get("top_k"),
            enable_thinking=cfg.get("enable_thinking"),
        )
    if provider == "litellm":
        return LiteLLMAdapter(
            model=cfg["model"],
            max_tokens=cfg.get("max_tokens", 700),
            temperature=cfg.get("temperature", 0.0),
            api_key=cfg.get("api_key"),
            api_base=cfg.get("api_base"),
        )
    if provider == "custom":
        import importlib
        class_spec = cfg["class"]  # e.g. "my_module:MyLLM"
        module_path, class_name = class_spec.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        kwargs = {k: v for k, v in cfg.items() if k not in ("provider", "class")}
        return cls(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'local_hf', 'litellm', or 'custom'.")

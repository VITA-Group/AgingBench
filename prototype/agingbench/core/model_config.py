"""Per-model generation defaults and behavior flags.

Some model vendors publish required generation settings (e.g. Gemma 4
recommends temperature=1.0, top_p=0.95, top_k=64, and requires that
<think>…</think> blocks be stripped from multi-turn history). This
module centralizes those overrides so they apply automatically whenever
the matching model is loaded.

Resolution order at call sites:
  explicit YAML field  >  model_config default  >  built-in fallback
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    strip_thinking: bool = False
    # Extra stop-token strings to add to the tokenizer's EOS list at generate
    # time. Needed when a vendor ships turn/channel markers the base EOS
    # doesn't cover (e.g. Gemma 4's <turn|>).
    extra_eos_tokens: tuple[str, ...] = ()
    # Regex patterns stripped from decoded output before it's returned. Used
    # to hide thought-channel content that leaks through skip_special_tokens.
    output_strip_patterns: tuple[str, ...] = ()
    # If True, pass enable_thinking=True to the tokenizer's chat template so
    # the model actually reasons in its thought channel. The channel content
    # is captured for traces but still stripped from conversation history.
    enable_thinking: bool = False
    # Regex with one capture group that extracts the thought-channel contents
    # from a raw decoded output (specials visible). Used to log reasoning
    # separately. Only meaningful when enable_thinking is True.
    thought_capture_pattern: Optional[str] = None


# Gemma 4 special token map (from the google/gemma-4-*-it tokenizer):
#   <|channel> / <channel|>  = start/end of a channel block (e.g. thought)
#   <|turn>    / <turn|>     = start/end of a dialogue turn
#   <|think|>                = system-prompt flag to enable thinking
# Without <turn|> in EOS, generation walks past the turn boundary and emits
# answer/channel/answer loops. Stripping <|channel>...<channel|> removes the
# leaked "thought\n[reasoning]" text that appears after skip_special_tokens.
GEMMA4 = ModelConfig(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    strip_thinking=True,
    extra_eos_tokens=("<turn|>",),
    # Non-greedy up to <channel|> OR end-of-string. The end-of-string branch is
    # critical: when max_new_tokens cuts generation mid-thought, the closing
    # <channel|> never emits and a close-anchored pattern would silently leak
    # the entire thinking trace into the returned text (and from there into
    # memory and chat history).
    output_strip_patterns=(
        r"<\|channel>.*?(?:<channel\|>|\Z)",
    ),
    enable_thinking=True,
    thought_capture_pattern=r"<\|channel>thought\n?(.*?)(?:<channel\|>|\Z)",
)


# Qwen3 uses a chat-template toggle `enable_thinking` that wraps reasoning in
# <think>...</think>. We default the toggle OFF to preserve existing Qwen3
# run behavior; individual SUTs opt in via `model.enable_thinking: true`.
# When on, long reasoning traces precede the answer and must be stripped
# before the answer enters chat history / memory.
QWEN3 = ModelConfig(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    strip_thinking=True,
    enable_thinking=False,
    output_strip_patterns=(
        r"<think>.*?(?:</think>|\Z)",
    ),
    thought_capture_pattern=r"<think>(.*?)(?:</think>|\Z)",
)


_REGISTRY: list[tuple[str, ModelConfig]] = [
    ("gemma-4", GEMMA4),
    ("qwen3", QWEN3),
]


def get_model_config(model_id: Optional[str]) -> ModelConfig:
    mid = (model_id or "").lower()
    for pattern, cfg in _REGISTRY:
        if pattern in mid:
            return cfg
    return ModelConfig()

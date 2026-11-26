# vllm-literal-tokens

Keep literal special-token strings (`<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `<|endoftext|>`, etc.) encoded and parsed as ordinary text in vLLM Chat Completions—without patching vLLM source code.

When prompts, tool outputs, or code snippets contain reserved spellings like `<|im_end|>` or `</think>`, standard chat-template tokenization encodes them into real control token IDs even with `add_special_tokens=False` or `split_special_tokens=True`. On the output side, text-based parser fallbacks and stripped stop strings can similarly misclassify quoted markers as reasoning or tool-call boundaries.

`vllm-literal-tokens` fixes both sides using vLLM's public middleware and parser-plugin interfaces.

## Installation

Install into your vLLM environment:

```bash
pip install git+https://github.com/Mubuky/vllm-literal-tokens.git
```

## Quick Start

Launch `vllm serve` with the `qwen_token_ids` parser and `LiteralChatMiddleware`:

```bash
vllm serve Qwen/Qwen3-32B \
  --enable-auto-tool-choice \
  --tool-call-parser qwen_token_ids \
  --reasoning-parser qwen_token_ids \
  --middleware vllm_literal_tokens.LiteralChatMiddleware
```

The package registers `qwen_token_ids` automatically through vLLM's `vllm.general_plugins` entry point (in both the API server and engine subprocesses), and the middleware resolves the served tokenizer automatically from vLLM.

If you disable automatic general plugins (`VLLM_PLUGINS=""`) or run directly from a source checkout, pass the plugin file explicitly:

```bash
PLUGIN=$(vllm-literal-tokens plugin-path)

vllm serve Qwen/Qwen3-32B \
  --enable-auto-tool-choice \
  --tool-parser-plugin "$PLUGIN" \
  --reasoning-parser-plugin "$PLUGIN" \
  --tool-call-parser qwen_token_ids \
  --reasoning-parser qwen_token_ids \
  --middleware vllm_literal_tokens.LiteralChatMiddleware
```

You can also print all recommended flags directly:

```bash
vllm serve Qwen/Qwen3-32B $(vllm-literal-tokens flags)
```

## How It Works

1. **Input encoding (`LiteralTokenizer` + `LiteralChatMiddleware`)**
   - Message content, reasoning, tool arguments, and tool schemas are replaced with unique opaque slots before running the model's official Jinja chat template.
   - Template-generated scaffolding is tokenized with the full vocabulary so real framing tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, `<tool_call>`) keep their control IDs.
   - All user, assistant, tool, and schema text is tokenized with an identical tokenizer whose `added_tokens` table has been cleared—preserving native Unicode normalization and BPE merges while making it impossible for literal text to emit a reserved control token ID.
   - The resulting integer sequence is passed to vLLM through `kv_transfer_params.prompt_token_ids`, so vLLM consumes the exact IDs directly without re-tokenizing text.

2. **Output parsing (`qwen_token_ids` parser plugin + aligned detokenization)**
   - `qwen_token_ids` extends vLLM's native token-ID `Qwen3Parser` and disables the bare `<function=` text fallback in token-ID mode, so tool calls are triggered only by real `<tool_call>` control tokens.
   - The middleware enables `include_stop_str_in_output=True` and `return_token_ids=True` so decoded text stays aligned with `model_output_token_ids` when parsing quoted special markers.
   - On non-streaming responses where an auto tool call hits `max_tokens` without emitting an EOS token, `finish_reason` is restored to `"length"`.

3. **Multi-turn token continuation**
   - If your client or gateway already maintains exact token IDs across turns (TITO) and supplies `kv_transfer_params.prompt_token_ids`, the middleware validates the length and forwards those IDs untouched.

## Python API

You can also use `LiteralTokenizer` directly outside vLLM (for offline encoding, dataset preparation, or custom token-continuation clients):

```python
from transformers import AutoTokenizer
from vllm_literal_tokens import LiteralTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
literal_tok = LiteralTokenizer(tokenizer)

prompt_ids = literal_tok.encode(
    [{"role": "user", "content": "Echo literally: <think>R</think> <|im_end|>"}],
    add_generation_prompt=True,
)
```

## Configuration

All environment variables are optional:

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_LITERAL_MODEL` | Auto-detected | Explicit model path or HF repo ID for tokenizer loading. |
| `VLLM_LITERAL_GENERATION_CONFIG` | Auto-detected | Explicit path to a directory or `generation_config.json` (also read from `--generation-config`). |
| `VLLM_LITERAL_REASONING_HISTORY` | `preserve` | Set to `last-user` for templates that omit reasoning before the last user turn (e.g. Qwen3.5). |
| `VLLM_LITERAL_CHAT_TEMPLATE_KWARGS` | `{}` | Optional JSON object of default `chat_template_kwargs` merged under per-request overrides. |
| `VLLM_LITERAL_REASONING_ALIAS` | `0` | Set to `1` to mirror `message.reasoning` into `message.reasoning_content` on non-streaming responses. |
| `VLLM_LITERAL_LOCAL_FILES_ONLY` | `0` | Set to `1` to load the tokenizer strictly from local cache. |
| `VLLM_LITERAL_TRUST_REMOTE_CODE` | `0` | Set to `1` to pass `trust_remote_code=True` when loading the tokenizer. |

## Compatibility

- Designed for text-only Chat Completions on Qwen3-family models (Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8).
- Tested with vLLM `0.28.0` (using `OnlineRenderer` prompt-ID reuse and `vllm.parser.qwen3.Qwen3Parser`).

## Development

Run the standalone unit tests (no GPU or vLLM required):

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Run the real Qwen tokenizer & vLLM integration tests inside a vLLM environment:

```bash
VLLM_LITERAL_TEST_TOKENIZER=/path/to/qwen-model \
  PYTHONPATH=src python3 -m unittest tests/test_qwen_integration.py -v
```

## License

Apache-2.0

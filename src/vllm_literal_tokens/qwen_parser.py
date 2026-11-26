"""vLLM parser plugin keeping literal XML outside real tool-call control tokens.

vLLM's native Qwen3Parser runs a token-ID state machine, but retains a text
fallback that treats bare ``<function=...`` strings inside content as tool
calls even when no real ``<tool_call>`` control token was generated. This plugin
disables only that fallback in token-ID mode and leaves grammar, whitespace,
argument parsing, and response construction completely native.
"""
from __future__ import annotations

from dataclasses import replace
import logging

logger = logging.getLogger(__name__)

PARSER_NAME = "qwen_token_ids"
_REGISTERED = False


def _register_once(manager, name: str, cls) -> None:
    try:
        manager.register_module(name, force=True)(cls)
    except TypeError:
        manager.register_module(name)(cls)


def register() -> None:
    """Register the ``qwen_token_ids`` reasoning and tool parsers with vLLM."""
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from vllm.parser.engine.adapters import make_adapters
        from vllm.parser.engine.parser_engine_config import ParserState
        from vllm.parser.qwen3 import Qwen3Parser, qwen3_config
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager
    except ImportError as exc:
        logger.debug(
            "vllm-literal-tokens: skipping %s registration (%s)", PARSER_NAME, exc
        )
        return

    class QwenLiteralParser(Qwen3Parser):
        def __init__(self, tokenizer, tools=None, **kwargs):
            thinking = (kwargs.get("chat_template_kwargs") or {}).get("enable_thinking", True)
            config = qwen3_config(thinking=thinking)
            key = (ParserState.CONTENT, "FUNC_PREFIX")
            if key in config.transitions:
                kwargs["parser_engine_config"] = replace(
                    config,
                    transitions={
                        **config.transitions,
                        key: replace(config.transitions[key], skip_in_token_id_mode=True),
                    },
                )
            super().__init__(tokenizer, tools, **kwargs)

    reasoning_base, tool_base = make_adapters(QwenLiteralParser)

    class ReasoningAdapter(reasoning_base):
        pass

    class ToolAdapter(tool_base):
        structural_tag_model = "qwen_3_coder"

    _register_once(ReasoningParserManager, PARSER_NAME, ReasoningAdapter)
    _register_once(ToolParserManager, PARSER_NAME, ToolAdapter)
    _REGISTERED = True


# Register immediately when vLLM loads this file via --tool-parser-plugin or
# --reasoning-parser-plugin.
register()

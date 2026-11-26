"""ASGI middleware for vLLM `/v1/chat/completions` with literal-safe prompt IDs.

vLLM remains in charge of sampling, generation, parsing, and response
construction. This middleware:
1. Pre-encodes chat messages with :class:`LiteralTokenizer` and passes the exact
   integer sequence through vLLM's native ``kv_transfer_params.prompt_token_ids``
   interface (or forwards caller-supplied ``prompt_token_ids`` untouched).
2. Sets ``include_stop_str_in_output=True`` and ``return_token_ids=True`` so
   token-ID parsers stay aligned with decoded output text.
3. Restores ``finish_reason="length"`` when an auto tool call hits the output
   token budget without emitting an EOS token.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

from .encoder import EncodingError, LiteralTokenizer, extract_text


class ContextLengthError(EncodingError):
    """Raised when the encoded prompt exhausts the model context window."""

    def __init__(self, model: str, input_tokens: int, context_limit: int) -> None:
        super().__init__(
            f"This model's maximum context length is {context_limit} tokens. "
            f"However, your messages resulted in {input_tokens} tokens."
        )
        self.model = model
        self.input_tokens = input_tokens
        self.context_limit = context_limit


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _model_from_argv(argv: list[str] | None = None) -> tuple[str | None, str | None]:
    """Extract (--model / positional serve model, --tokenizer) from vLLM CLI argv."""
    args = list(sys.argv[1:] if argv is None else argv)
    model: str | None = None
    tokenizer: str | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--tokenizer" and i + 1 < len(args):
            tokenizer = args[i + 1]
            i += 2
            continue
        if arg.startswith("--tokenizer="):
            tokenizer = arg.split("=", 1)[1]
            i += 1
            continue
        if arg in ("--model", "-m") and i + 1 < len(args):
            model = args[i + 1]
            i += 2
            continue
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "serve" and i + 1 < len(args) and not args[i + 1].startswith("-"):
            model = args[i + 1]
            i += 2
            continue
        i += 1
    return model, tokenizer


def _load_generation_config(model_ref: str | None, *, local_files_only: bool = False) -> dict[str, Any]:
    if not model_ref:
        return {}
    local_path = Path(model_ref) / "generation_config.json"
    if local_path.is_file():
        try:
            return json.loads(local_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    try:
        from transformers import GenerationConfig

        cfg = GenerationConfig.from_pretrained(model_ref, local_files_only=local_files_only)
        return cfg.to_dict()
    except Exception:
        return {}


def _extract_from_app_state(app: Any) -> tuple[Any | None, str | None, int | None, dict[str, Any]]:
    """Inspect vLLM's FastAPI/Starlette app state for tokenizer and model config."""
    state = getattr(app, "state", None)
    if state is None:
        return None, None, None, {}

    tokenizer = None
    model_ref: str | None = None
    max_model_len: int | None = None
    gen_cfg: dict[str, Any] = {}

    candidates = [
        getattr(state, "openai_serving_chat", None),
        getattr(state, "engine_client", None),
        getattr(state, "openai_serving_models", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        if tokenizer is None:
            renderer = getattr(obj, "renderer", None)
            if renderer is not None:
                get_tok = getattr(renderer, "get_tokenizer", None)
                tokenizer = get_tok() if callable(get_tok) else getattr(renderer, "tokenizer", None)
        model_config = getattr(obj, "model_config", None)
        if model_config is None:
            engine = getattr(obj, "engine_client", None)
            model_config = getattr(engine, "model_config", None) if engine is not None else None
        if model_config is not None:
            if model_ref is None:
                model_ref = getattr(model_config, "tokenizer", None) or getattr(
                    model_config, "model", None
                )
            if max_model_len is None:
                raw_len = getattr(model_config, "max_model_len", None)
                if isinstance(raw_len, int) and raw_len > 0:
                    max_model_len = raw_len
            if not gen_cfg:
                raw_gen = getattr(model_config, "generation_config", None)
                if isinstance(raw_gen, dict):
                    gen_cfg = raw_gen

    return tokenizer, model_ref, max_model_len, gen_cfg


def _eos_ids(tokenizer: Any, generation_config: dict[str, Any]) -> set[int]:
    raw = generation_config.get("eos_token_id")
    if isinstance(raw, int):
        return {raw}
    if isinstance(raw, (list, tuple, set)):
        return {x for x in raw if isinstance(x, int)}
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tok_eos, int):
        return {tok_eos}
    return set()


def _effective_max_tokens(
    context_limit: int,
    requested: int | None,
    prompt_len: int,
    generation_config: dict[str, Any],
) -> int:
    try:
        from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

        return int(
            get_max_tokens(
                context_limit,
                requested,
                prompt_len,
                {"max_tokens": generation_config.get("max_new_tokens")},
            )
        )
    except Exception:
        remaining = max(0, context_limit - prompt_len)
        if requested is not None:
            return min(int(requested), remaining)
        default_new = generation_config.get("max_new_tokens")
        if isinstance(default_new, int) and default_new > 0:
            return min(default_new, remaining)
        return remaining


def _error_payload(message: str, *, code: str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": code,
        }
    }


class LiteralChatMiddleware:
    """ASGI middleware that injects literal-safe ``prompt_token_ids`` into vLLM."""

    def __init__(
        self,
        app: Any,
        *,
        tokenizer: Any | None = None,
        model: str | None = None,
        reasoning_history: str | None = None,
        generation_config: dict[str, Any] | None = None,
        max_model_len: int | None = None,
        reasoning_content_alias: bool | None = None,
    ) -> None:
        self.app = app
        self._explicit_model = model or os.environ.get("VLLM_LITERAL_MODEL")
        self._reasoning_history = reasoning_history or os.environ.get(
            "VLLM_LITERAL_REASONING_HISTORY", "preserve"
        )
        self._local_files_only = _env_flag("VLLM_LITERAL_LOCAL_FILES_ONLY", False)
        self._trust_remote_code = _env_flag("VLLM_LITERAL_TRUST_REMOTE_CODE", False)
        self._reasoning_alias = (
            _env_flag("VLLM_LITERAL_REASONING_ALIAS", False)
            if reasoning_content_alias is None
            else reasoning_content_alias
        )
        self._max_model_len = max_model_len
        self._lock = threading.Lock()
        self.protocol: LiteralTokenizer | None = None
        self.generation_config: dict[str, Any] = dict(generation_config or {})

        if tokenizer is not None:
            self.protocol = LiteralTokenizer(
                tokenizer, reasoning_history=self._reasoning_history
            )
        elif self._explicit_model:
            self._init_from_model_ref(self._explicit_model)

    def _init_from_model_ref(self, model_ref: str) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            local_files_only=self._local_files_only,
            trust_remote_code=self._trust_remote_code,
        )
        self.protocol = LiteralTokenizer(tokenizer, reasoning_history=self._reasoning_history)
        if not self.generation_config:
            self.generation_config = _load_generation_config(
                model_ref, local_files_only=self._local_files_only
            )

    def _ensure_initialized(self, scope: dict[str, Any]) -> LiteralTokenizer:
        if self.protocol is not None:
            return self.protocol
        with self._lock:
            if self.protocol is not None:
                return self.protocol
            tok, app_model, app_max_len, app_gen = _extract_from_app_state(scope.get("app"))
            if app_max_len is not None and self._max_model_len is None:
                self._max_model_len = app_max_len
            if app_gen and not self.generation_config:
                self.generation_config = dict(app_gen)

            if tok is not None and hasattr(tok, "backend_tokenizer"):
                self.protocol = LiteralTokenizer(tok, reasoning_history=self._reasoning_history)
                if not self.generation_config and app_model:
                    self.generation_config = _load_generation_config(
                        app_model, local_files_only=self._local_files_only
                    )
                return self.protocol

            cli_model, cli_tokenizer = _model_from_argv()
            model_ref = self._explicit_model or cli_tokenizer or app_model or cli_model
            if not model_ref:
                raise EncodingError(
                    "Could not resolve model tokenizer automatically; "
                    "set VLLM_LITERAL_MODEL to the served model path or repo ID."
                )
            self._init_from_model_ref(model_ref)
            assert self.protocol is not None
            return self.protocol

    def _context_limit(self, protocol: LiteralTokenizer) -> int:
        if self._max_model_len is not None:
            return self._max_model_len
        raw = getattr(protocol.tokenizer, "model_max_length", None)
        if isinstance(raw, int) and 0 < raw < 10**9:
            return raw
        return 10**9

    def prepare_request(
        self, original: dict[str, Any], *, protocol: LiteralTokenizer | None = None
    ) -> dict[str, Any]:
        """Validate and augment a `/v1/chat/completions` JSON payload."""
        active = protocol or self.protocol
        if active is None:
            raise EncodingError("LiteralChatMiddleware is not initialized")

        # Use vLLM's native ChatCompletionRequest validation when available.
        messages_src: list[dict[str, Any]]
        chat_template_kwargs: dict[str, Any] = dict(original.get("chat_template_kwargs") or {})
        tool_choice = original.get("tool_choice")
        model_name = original.get("model", "")
        kv_params = dict(original.get("kv_transfer_params") or {})

        try:
            from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

            validated = ChatCompletionRequest.model_validate(deepcopy(original))
            messages_src = list(validated.messages)
            tool_choice = validated.tool_choice
            model_name = validated.model
            if validated.kv_transfer_params:
                kv_params = dict(validated.kv_transfer_params)
            if validated.chat_template_kwargs:
                chat_template_kwargs = dict(validated.chat_template_kwargs)
        except ImportError:
            raw_messages = original.get("messages")
            if not isinstance(raw_messages, list):
                raise EncodingError("'messages' must be a list")
            messages_src = deepcopy(raw_messages)

        messages = []
        for m in messages_src:
            item = {**m, "content": extract_text(m.get("content"))}
            if m.get("role") == "assistant":
                raw_reasoning = m.get("reasoning_content")
                if raw_reasoning is None:
                    raw_reasoning = m.get("reasoning")
                item["reasoning_content"] = extract_text(raw_reasoning)
            messages.append(item)
        tools = [] if tool_choice == "none" else (original.get("tools") or [])

        ids = kv_params.get("prompt_token_ids")
        if ids is None:
            ids = active.encode(
                messages,
                tools,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs,
            )
        elif not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
            raise EncodingError("kv_transfer_params.prompt_token_ids must be a list of integers")

        limit = self._context_limit(active)
        if len(ids) >= limit:
            raise ContextLengthError(model_name, len(ids), limit)

        outgoing_template_kwargs = dict(chat_template_kwargs)
        if self._reasoning_history == "preserve":
            outgoing_template_kwargs.setdefault("preserve_thinking", True)

        outgoing = {
            **original,
            "return_token_ids": original.get("return_token_ids", True),
            "include_stop_str_in_output": original.get("include_stop_str_in_output", True),
            "kv_transfer_params": {**kv_params, "prompt_token_ids": ids},
        }
        if outgoing_template_kwargs:
            outgoing["chat_template_kwargs"] = outgoing_template_kwargs
        return outgoing

    def postprocess_response(
        self,
        original: dict[str, Any],
        outgoing: dict[str, Any],
        response: dict[str, Any],
        *,
        protocol: LiteralTokenizer | None = None,
    ) -> dict[str, Any]:
        """Apply safe non-streaming fixes without altering native response structure."""
        active = protocol or self.protocol
        if active is None:
            return response

        choices = response.get("choices")
        if not isinstance(choices, list):
            return response

        requested = original.get("max_completion_tokens")
        if requested is None:
            requested = original.get("max_tokens")
        prompt_len = len(outgoing["kv_transfer_params"]["prompt_token_ids"])
        limit = _effective_max_tokens(
            self._context_limit(active), requested, prompt_len, self.generation_config
        )
        eos = _eos_ids(active.tokenizer, self.generation_config)

        updated_choices = []
        for choice in choices:
            new_choice = dict(choice)
            message = new_choice.get("message")
            if (
                self._reasoning_alias
                and isinstance(message, dict)
                and "reasoning_content" not in message
                and "reasoning" in message
            ):
                new_choice["message"] = {
                    **message,
                    "reasoning_content": message.get("reasoning"),
                }

            ids = new_choice.get("token_ids") or []
            if (
                new_choice.get("finish_reason") == "tool_calls"
                and limit > 0
                and len(ids) == limit
                and new_choice.get("stop_reason") is None
                and ids[-1] not in eos
            ):
                new_choice["finish_reason"] = "length"
            updated_choices.append(new_choice)

        return {**response, "choices": updated_choices}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return

        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if not message.get("more_body"):
                break

        try:
            protocol = self._ensure_initialized(scope)
            original = json.loads(body)
            if not isinstance(original, dict):
                raise EncodingError("Request body must be a JSON object")
            outgoing = self.prepare_request(original, protocol=protocol)
        except (ValueError, KeyError, TypeError) as error:
            code = "context_length_exceeded" if isinstance(error, ContextLengthError) else None
            payload = json.dumps(_error_payload(str(error), code=code)).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(payload)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": payload})
            return

        wire = json.dumps(outgoing, ensure_ascii=False).encode("utf-8")
        sent_body = False

        async def upstream_receive() -> dict[str, Any]:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": wire, "more_body": False}
            return await receive()

        headers = [
            (k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"
        ]
        headers.append((b"content-length", str(len(wire)).encode("ascii")))
        upstream_scope = {**scope, "headers": headers}

        if original.get("stream"):
            await self.app(upstream_scope, upstream_receive, send)
            return

        captured: list[dict[str, Any]] = []

        async def capture(msg: dict[str, Any]) -> None:
            captured.append(msg)

        await self.app(upstream_scope, upstream_receive, capture)
        start = next(m for m in captured if m["type"] == "http.response.start")
        raw = b"".join(m.get("body", b"") for m in captured if m["type"] == "http.response.body")

        if start["status"] == 200:
            try:
                parsed = json.loads(raw)
                updated = self.postprocess_response(
                    original, outgoing, parsed, protocol=protocol
                )
                raw = json.dumps(updated, ensure_ascii=False).encode("utf-8")
            except (ValueError, KeyError, TypeError):
                pass

        resp_headers = [
            (k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"
        ]
        resp_headers.append((b"content-length", str(len(raw)).encode("ascii")))
        await send(
            {
                "type": "http.response.start",
                "status": start["status"],
                "headers": resp_headers,
            }
        )
        await send({"type": "http.response.body", "body": raw})

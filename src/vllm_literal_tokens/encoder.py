"""Literal-field tokenization at the chat-template boundary.

Message fields and tool schemas are replaced with opaque placeholders before
running the model's official Jinja chat template. Template-generated structure
is tokenized with the full vocabulary (emitting real control token IDs), while
user, assistant, tool, and schema text is tokenized with an identical tokenizer
whose ``added_tokens`` table has been cleared so reserved spellings stay
ordinary BPE tokens.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any
import uuid


class EncodingError(ValueError):
    """Raised when a chat payload cannot be safely encoded."""


@dataclass(frozen=True)
class Span:
    text: str
    control: bool = False
    prefill: bool = False


def extract_text(value: Any) -> str:
    """Normalize an OpenAI text field (None, string, or text-part list)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(part, dict)
        and part.get("type", "text") == "text"
        and isinstance(part.get("text"), str)
        for part in value
    ):
        return "".join(part["text"] for part in value)
    raise EncodingError("Only text message content is supported")


class LiteralTokenizer:
    """Encode chat messages while keeping literal special markers as plain text."""

    def __init__(self, tokenizer: Any, *, reasoning_history: str = "preserve") -> None:
        from tokenizers import Tokenizer

        if reasoning_history not in ("preserve", "last-user"):
            raise EncodingError(
                f"Unsupported reasoning_history={reasoning_history!r}; "
                "expected 'preserve' or 'last-user'"
            )
        self.tokenizer = tokenizer
        self.reasoning_history = reasoning_history

        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        self.reserved: set[int] = {entry["id"] for entry in backend.get("added_tokens", ())}
        backend["added_tokens"] = []
        self.ordinary = Tokenizer.from_str(json.dumps(backend))

        added = tokenizer.added_tokens_encoder
        if added:
            self.markers: re.Pattern[str] | None = re.compile(
                "(" + "|".join(re.escape(s) for s in sorted(added, key=len, reverse=True)) + ")"
            )
        else:
            self.markers = None

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def control(self, text: str) -> list[int]:
        """Encode template scaffolding using the full vocabulary."""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def literal(self, text: str) -> list[int]:
        """Encode literal payload text without emitting any reserved token ID."""
        if not text:
            return []
        encoded = self.ordinary.encode(text, add_special_tokens=False)
        if self.reserved.intersection(encoded.ids):
            raise EncodingError("Literal field emitted a reserved control token")
        return list(encoded.ids)

    def render_spans(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        *,
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[Span]:
        """Render the official Jinja chat template and split into control/literal spans."""
        if not messages:
            raise EncodingError("At least one message is required")

        prefix = "LITERAL_SLOT_" + uuid.uuid4().hex + "_"
        if prefix in json.dumps([messages, list(tools)], ensure_ascii=False):
            raise EncodingError("Template slot collision")

        slots: dict[str, str] = {}
        optional_slots: set[str] = set()

        def make_slot(value: str, *, quoted: bool = False) -> Any:
            if not isinstance(value, str):
                raise EncodingError("Only text fields are supported")
            if not value:
                return value
            name = f"{prefix}{len(slots)}_END"
            if quoted:
                wrapped = {"__literal_slot__": name}
                # Support both default json.dumps spacing and compact separators.
                slots[json.dumps(wrapped)] = value
                slots[json.dumps(wrapped, separators=(",", ":"))] = value
                return wrapped
            slots[name] = value
            return name

        prepared = deepcopy(messages)
        user_indices = [i for i, m in enumerate(prepared) if m.get("role") == "user"]
        last_user = max(user_indices) if user_indices else -1

        for idx, message in enumerate(prepared):
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise EncodingError("Each message must include a non-empty role")

            # Apply outer whitespace trimming before slot replacement so
            # whitespace-only content follows the template's empty-content branch.
            raw_content = extract_text(message.get("content"))
            message["content"] = make_slot(raw_content.strip())

            if role == "assistant":
                raw_reasoning_val = message.get("reasoning_content")
                if raw_reasoning_val is None:
                    raw_reasoning_val = message.get("reasoning")
                raw_reasoning = extract_text(raw_reasoning_val).strip()
                if self.reasoning_history == "last-user" and idx < last_user:
                    raw_reasoning = ""
                reasoning_slot = make_slot(raw_reasoning)
                message["reasoning_content"] = reasoning_slot
                if "reasoning" in message:
                    message["reasoning"] = reasoning_slot
                if reasoning_slot and idx < last_user:
                    optional_slots.add(reasoning_slot)

            for call in message.get("tool_calls") or []:
                fn = call.get("function", call)
                fn["name"] = make_slot(fn["name"])
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as exc:
                        raise EncodingError("Tool call arguments must be valid JSON") from exc
                if not isinstance(args, dict):
                    raise EncodingError("Tool call arguments must decode to a JSON object")
                encoded_args: dict[str, str] = {}
                for key, value in args.items():
                    raw_val = (
                        value
                        if isinstance(value, str)
                        else json.dumps(value, ensure_ascii=False, allow_nan=False)
                    )
                    # Tool arguments keep exact whitespace without stripping.
                    encoded_args[make_slot(key)] = make_slot(raw_val)
                fn["arguments"] = encoded_args

        prepared_tools = [
            make_slot(json.dumps(tool, ensure_ascii=False), quoted=True) for tool in tools
        ]

        template_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
        if self.reasoning_history == "preserve":
            template_kwargs.setdefault("preserve_thinking", True)
        template_kwargs.pop("tokenize", None)
        template_kwargs.pop("add_generation_prompt", None)
        template_kwargs.pop("tools", None)

        call_kwargs: dict[str, Any] = {
            "tokenize": False,
            **template_kwargs,
        }
        if prepared_tools:
            call_kwargs["tools"] = prepared_tools

        rendered = self.tokenizer.apply_chat_template(
            prepared, add_generation_prompt=False, **call_kwargs
        )
        prompted = self.tokenizer.apply_chat_template(
            prepared, add_generation_prompt=True, **call_kwargs
        )
        if not prompted.startswith(rendered):
            raise EncodingError("Chat template generation prompt rewrites conversation history")
        prefill = prompted[len(rendered) :]
        full_text = prompted if add_generation_prompt else rendered

        if slots:
            slot_pattern = re.compile(
                "(" + "|".join(re.escape(k) for k in sorted(slots, key=len, reverse=True)) + ")"
            )
            pieces = slot_pattern.split(full_text)
        else:
            pieces = [full_text]

        prefill_pattern = re.compile("(" + re.escape(prefill) + ")") if prefill else None
        result: list[Span] = []
        seen_names: list[str] = []

        # Group quoted aliases back to their canonical slot name for integrity verification.
        canonical_name: dict[str, str] = {}
        for key in slots:
            match = re.search(re.escape(prefix) + r"\d+_END", key)
            canonical_name[key] = match.group(0) if match else key

        for piece in pieces:
            if not piece:
                continue
            if piece in slots:
                seen_names.append(canonical_name[piece])
                result.append(Span(text=slots[piece], control=False))
                continue
            if prefill_pattern is not None:
                for sub in prefill_pattern.split(piece):
                    if sub:
                        result.append(Span(text=sub, control=True, prefill=sub == prefill))
            else:
                result.append(Span(text=piece, control=True))

        expected_names = set(canonical_name.values())
        seen_set = set(seen_names)
        if len(seen_names) != len(seen_set) or (expected_names - seen_set - optional_slots):
            raise EncodingError("Chat template dropped or duplicated a payload field")

        return result

    def encode(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        *,
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Encode a conversation into token IDs without turning literal text into control tokens."""
        added = self.tokenizer.added_tokens_encoder
        ids: list[int] = []
        buffer = ""

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            encoded = self.ordinary.encode(buffer, add_special_tokens=False)
            if self.reserved.intersection(encoded.ids):
                raise EncodingError("Literal field emitted a reserved control token")
            ids.extend(encoded.ids)
            buffer = ""

        for span in self.render_spans(
            messages,
            tools,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        ):
            if span.control and self.markers is not None:
                pieces = self.markers.split(span.text)
            else:
                pieces = [span.text]

            for piece in pieces:
                if not piece:
                    continue
                if span.control and piece in added:
                    flush()
                    ids.append(added[piece])
                else:
                    buffer += piece

            if span.prefill:
                # Match the real inference prefill boundary so prompt tokens
                # never merge across assistant generation openings.
                flush()

        flush()
        return ids

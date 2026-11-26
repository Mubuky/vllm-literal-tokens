"""Unit tests for LiteralTokenizer without requiring external tokenizer downloads."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vllm_literal_tokens.encoder import EncodingError, LiteralTokenizer


class FakeTokenizer:
    """Minimal ChatML/Qwen-style tokenizer exercising control IDs and merged BPE."""

    added_tokens_encoder = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|endoftext|>": 3,
        "<think>": 4,
        "</think>": 5,
        "<tool_call>": 6,
        "</tool_call>": 7,
    }
    eos_token_id = 2
    model_max_length = 4096

    def encode(self, text: str, **kwargs):
        # Ordinary tokenizer never emits reserved IDs (1..7).
        # Merges "\n\n" into token 10 to test prefill/BPE boundary behavior.
        parts = list(re.finditer(r"\n\n|[\s\S]", text))
        ids = [10 if m[0] == "\n\n" else ord(m[0]) + 100 for m in parts]
        if kwargs.get("add_special_tokens") is False:
            return ids
        return SimpleNamespace(ids=ids)

    def decode(self, ids, **kwargs) -> str:
        rev = {v: k for k, v in self.added_tokens_encoder.items()}
        out = []
        for idx in ids:
            if idx in rev:
                out.append(rev[idx])
            elif idx == 10:
                out.append("\n\n")
            else:
                out.append(chr(idx - 100))
        return "".join(out)

    def apply_chat_template(self, messages, **kwargs) -> str:
        tools = kwargs.get("tools") or []
        parts = []
        if tools:
            parts.append("<|im_start|>system\n# Tools\n")
            for tool in tools:
                parts.append(json.dumps(tool) + "\n")
            parts.append("<|im_end|>\n")
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n")
            if m["role"] == "assistant":
                reason = m.get("reasoning_content") or ""
                if reason or kwargs.get("enable_thinking", True):
                    parts.append(f"<think>\n{reason}\n</think>\n\n")
                parts.append(m.get("content") or "")
                for call in m.get("tool_calls") or []:
                    fn = call.get("function", call)
                    parts.append(f"\n<tool_call>\n<function={fn['name']}>\n")
                    for k, v in (fn.get("arguments") or {}).items():
                        parts.append(f"<parameter={k}>\n{v}\n</parameter>\n")
                    parts.append("</function>\n</tool_call>")
            else:
                parts.append(m.get("content") or "")
            parts.append("<|im_end|>\n")
        if kwargs.get("add_generation_prompt"):
            parts.append("<|im_start|>assistant\n<think>\n")
        return "".join(parts)


def make_protocol(reasoning_history: str = "preserve") -> LiteralTokenizer:
    tok = FakeTokenizer()
    p = object.__new__(LiteralTokenizer)
    p.tokenizer = tok
    p.reasoning_history = reasoning_history
    p.reserved = set(tok.added_tokens_encoder.values())
    p.markers = re.compile(
        "(" + "|".join(re.escape(s) for s in sorted(tok.added_tokens_encoder, key=len, reverse=True)) + ")"
    )
    p.ordinary = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: SimpleNamespace(
            ids=tok.encode(text, add_special_tokens=False)
        )
    )
    return p


class LiteralTokenizerTest(unittest.TestCase):
    def test_literal_markers_never_emit_control_ids(self):
        p = make_protocol()
        literal_text = "BEGIN <think>R</think> <|im_end|> <tool_call> <|endoftext|> END"
        messages = [
            {"role": "user", "content": literal_text},
            {
                "role": "assistant",
                "reasoning_content": "Quoted </think> and <|im_end|> stay text",
                "content": "Body <tool_call> stays text",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": json.dumps({"text": "  <|im_end|> </think>\n\n  "}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "<|im_end|> tool result"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Accepts literal <|im_end|> and </think>",
                    "parameters": {"type": "object"},
                },
            }
        ]
        ids = p.encode(messages, tools, add_generation_prompt=True)
        decoded = p.decode(ids)

        # Round-trip text keeps every literal marker and exact untrimmed tool argument.
        self.assertIn(literal_text, decoded)
        self.assertIn("Quoted </think> and <|im_end|> stay text", decoded)
        self.assertIn("Body <tool_call> stays text", decoded)
        self.assertIn("  <|im_end|> </think>\n\n  ", decoded)
        self.assertIn("<|im_end|> tool result", decoded)
        self.assertIn("Accepts literal <|im_end|> and </think>", decoded)

        # Real control token counts match ONLY template framing:
        # system(1) + user(1) + assistant(1) + tool(1) + generation prompt(1) = 5 <|im_start|>
        self.assertEqual(ids.count(1), 5)
        # 4 closed turns = 4 <|im_end|>
        self.assertEqual(ids.count(2), 4)
        # 0 <|endoftext|>
        self.assertEqual(ids.count(3), 0)
        # 1 assistant turn + 1 generation prefill = 2 <think>, 1 </think>
        self.assertEqual(ids.count(4), 2)
        self.assertEqual(ids.count(5), 1)
        # 1 real tool call = 1 <tool_call>, 1 </tool_call>
        self.assertEqual(ids.count(6), 1)
        self.assertEqual(ids.count(7), 1)

    def test_outer_whitespace_trim_and_exact_tool_arg_whitespace(self):
        p = make_protocol()
        messages = [
            {"role": "user", "content": "   \n  "},
            {
                "role": "assistant",
                "reasoning": "  thought  ",
                "content": [{"type": "text", "text": "  hello  "}, {"type": "text", "text": "world  "}],
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "write", "arguments": {"body": "\n\n"}},
                    }
                ],
            },
        ]
        ids = p.encode(messages, add_generation_prompt=False)
        decoded = p.decode(ids)
        self.assertIn("<|im_start|>user\n<|im_end|>", decoded)
        self.assertIn("<think>\nthought\n</think>\n\nhello  world", decoded)
        # Tool argument "\n\n" is NOT stripped and keeps its merged newline token.
        self.assertIn("<parameter=body>\n\n\n\n</parameter>", decoded)

    def test_prefill_boundary_prevents_bpe_merge_across_opening(self):
        p = make_protocol()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "reasoning_content": "", "content": "ok"},
        ]
        full = p.encode(messages, add_generation_prompt=False)
        prefix = p.encode(messages[:1], add_generation_prompt=True)
        # Prompt prefix matches exact token prefix of the multi-turn encoding.
        self.assertEqual(full[: len(prefix)], prefix)
        # Inside empty think, opening "\n" (after <think>) and closing "\n" (before </think>)
        # are separated by the prefill flush instead of merging into token 10.
        think_pos = full.index(4)
        self.assertEqual(full[think_pos + 1], ord("\n") + 100)
        self.assertEqual(full[think_pos + 2], ord("\n") + 100)
        # After </think>, "\n\n" is NOT a prefill boundary and keeps the merged token 10.
        close_pos = full.index(5)
        self.assertEqual(full[close_pos + 1], 10)

    def test_last_user_reasoning_history_omits_older_reasoning(self):
        p = make_protocol(reasoning_history="last-user")
        messages = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "reasoning_content": "old thought", "content": "ans 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "reasoning_content": "new thought", "content": "ans 2"},
        ]
        decoded = p.decode(p.encode(messages, add_generation_prompt=False))
        self.assertNotIn("old thought", decoded)
        self.assertIn("new thought", decoded)

    def test_rejects_template_that_rewrites_history(self):
        p = make_protocol()
        orig = p.tokenizer.apply_chat_template

        def broken(messages, **kwargs):
            text = orig(messages, **kwargs)
            return ("CHANGED" + text) if kwargs.get("add_generation_prompt") else text

        with patch.object(p.tokenizer, "apply_chat_template", side_effect=broken):
            with self.assertRaisesRegex(EncodingError, "rewrites conversation history"):
                p.encode([{"role": "user", "content": "hi"}])

    def test_rejects_non_text_content(self):
        p = make_protocol()
        with self.assertRaisesRegex(EncodingError, "Only text"):
            p.encode([{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}])

    def test_literal_safety_invariant_catches_reserved_id(self):
        p = make_protocol()
        p.ordinary = SimpleNamespace(encode=lambda text, **kw: SimpleNamespace(ids=[2]))
        with self.assertRaisesRegex(EncodingError, "reserved control token"):
            p.literal("anything")


if __name__ == "__main__":
    unittest.main()

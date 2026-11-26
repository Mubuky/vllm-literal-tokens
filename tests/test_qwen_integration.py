"""Integration tests against a real Qwen tokenizer and installed vLLM engine.

Skipped automatically unless ``VLLM_LITERAL_TEST_TOKENIZER`` points to a local
Qwen3+ model/tokenizer directory inside a vLLM environment.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
import unittest

from vllm_literal_tokens import LiteralChatMiddleware, LiteralTokenizer


class QwenVLLMIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.environ.get("VLLM_LITERAL_TEST_TOKENIZER")
        if not model_path:
            raise unittest.SkipTest("Set VLLM_LITERAL_TEST_TOKENIZER to run live vLLM tests")
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        cls.protocol = LiteralTokenizer(cls.tokenizer)

    def test_literal_markers_avoid_reserved_ids_and_roundtrip(self):
        for marker in ("</think>", "<think>", "<tool_call>", "</tool_call>", "<|im_end|>", "<|endoftext|>"):
            ids = self.protocol.literal(marker)
            self.assertFalse(self.protocol.reserved.intersection(ids))
            self.assertEqual(self.protocol.decode(ids), marker)

    def test_ordinary_conversation_matches_native_template_exactly(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "reasoning_content": "Thinking", "content": "normal XML <node>x</node>\n\nnext"},
        ]
        self.assertEqual(
            self.protocol.encode(messages, add_generation_prompt=False),
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, preserve_thinking=True
            ),
        )

    def test_vllm_online_renderer_and_token_id_parser(self):
        import vllm_literal_tokens.qwen_parser  # noqa: F401
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.parser import ParserManager
        from vllm.renderers.online_renderer import OnlineRenderer

        mw = LiteralChatMiddleware(app=None, tokenizer=self.tokenizer)
        wire = mw.prepare_request(
            {
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Quote </think> <|im_end|> literally"}],
            }
        )
        expected_ids = list(wire["kv_transfer_params"]["prompt_token_ids"])
        request = ChatCompletionRequest.model_validate(wire)

        renderer = SimpleNamespace(tokenizer=self.tokenizer, get_tokenizer=lambda: self.tokenizer)
        config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3"),
            model="qwen3",
            multimodal_config=None,
            enable_prompt_embeds=False,
            max_model_len=32768,
        )
        online = OnlineRenderer(
            config,
            renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="string",
            enable_auto_tools=True,
            tool_parser="qwen_token_ids",
            reasoning_parser="qwen_token_ids",
        )
        _, inputs = asyncio.run(online.render_chat(request))
        self.assertEqual(inputs[0]["prompt_token_ids"], expected_ids)

        parser = ParserManager.get_parser("qwen_token_ids", "qwen_token_ids", True)(self.tokenizer)
        parser.adjust_request(request)
        self.assertFalse(request.skip_special_tokens)
        self.assertTrue(request.include_stop_str_in_output)

        out_ids = (
            self.protocol.literal("Quoted <|im_end|> stays text.")
            + self.protocol.control("</think>")
            + self.protocol.literal("\n\nBody <function=echo> literal")
            + self.protocol.control("<|im_end|>")
        )
        text = self.tokenizer.decode(out_ids, skip_special_tokens=request.skip_special_tokens)
        reasoning, content, calls = parser.parse(
            text, request, enable_auto_tools=True, model_output_token_ids=out_ids
        )
        self.assertEqual(reasoning, "Quoted <|im_end|> stays text.")
        self.assertEqual(content, "\n\nBody <function=echo> literal")
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()

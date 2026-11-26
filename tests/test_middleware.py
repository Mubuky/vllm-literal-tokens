"""Unit tests for LiteralChatMiddleware and CLI helpers."""
from __future__ import annotations

import asyncio
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from vllm_literal_tokens import ContextLengthError, LiteralChatMiddleware, parser_plugin_path
from vllm_literal_tokens.cli import main as cli_main
from vllm_literal_tokens.middleware import _model_from_argv

from tests.test_encoder import make_protocol


def run_asgi(app, scope, body: bytes = b""):
    sent = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)

    asyncio.run(app(scope, receive, send))
    return sent


class MiddlewareTest(unittest.TestCase):
    def make_middleware(self, app, *, max_model_len=4096, reasoning_alias=False, gen_cfg=None):
        mw = LiteralChatMiddleware(
            app,
            max_model_len=max_model_len,
            reasoning_content_alias=reasoning_alias,
            generation_config=gen_cfg or {"eos_token_id": [2, 3]},
        )
        mw.protocol = make_protocol()
        return mw

    def test_injects_prompt_ids_and_alignment_flags(self):
        captured_requests = []
        native_response = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "qwen3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning": "Quoted <|im_end|> kept.",
                        "content": "OK",
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                    "stop_reason": 2,
                    "token_ids": [111, 107, 2],
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        }

        async def inner_app(scope, receive, send):
            raw = (await receive())["body"]
            captured_requests.append(json.loads(raw))
            body = json.dumps(native_response).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"x-request-id", b"req-123")],
                }
            )
            await send({"type": "http.response.body", "body": body})

        mw = self.make_middleware(inner_app, reasoning_alias=True)
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": "Say <|im_end|> and </think>"}],
            "temperature": 0.6,
        }
        sent = run_asgi(
            mw,
            {"type": "http", "path": "/v1/chat/completions", "headers": []},
            json.dumps(payload).encode("utf-8"),
        )

        self.assertEqual(len(captured_requests), 1)
        upstream = captured_requests[0]
        self.assertTrue(upstream["return_token_ids"])
        self.assertTrue(upstream["include_stop_str_in_output"])
        self.assertEqual(upstream["temperature"], 0.6)
        prompt_ids = upstream["kv_transfer_params"]["prompt_token_ids"]
        self.assertEqual(
            prompt_ids,
            mw.protocol.encode(payload["messages"], add_generation_prompt=True),
        )
        # Only 1 real <|im_end|> (end of user turn), not the literal one inside user content.
        self.assertEqual(prompt_ids.count(2), 1)

        self.assertEqual(sent[0]["status"], 200)
        self.assertIn((b"x-request-id", b"req-123"), sent[0]["headers"])
        reply = json.loads(sent[1]["body"])
        self.assertEqual(reply["choices"][0]["message"]["reasoning_content"], "Quoted <|im_end|> kept.")

    def test_preserves_caller_supplied_prompt_token_ids(self):
        seen = []

        async def inner_app(scope, receive, send):
            seen.append(json.loads((await receive())["body"]))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"choices":[]}'})

        mw = self.make_middleware(inner_app)
        custom_ids = [1, 200, 201, 2]
        payload = {
            "model": "qwen3",
            "messages": [{"role": "user", "content": "ignored because IDs supplied"}],
            "kv_transfer_params": {"prompt_token_ids": custom_ids},
        }
        run_asgi(
            mw,
            {"type": "http", "path": "/v1/chat/completions", "headers": []},
            json.dumps(payload).encode("utf-8"),
        )
        self.assertEqual(seen[0]["kv_transfer_params"]["prompt_token_ids"], custom_ids)

    def test_streaming_passes_through_without_buffering(self):
        chunks = [b"data: one\n\n", b"data: [DONE]\n\n"]

        async def inner_app(scope, receive, send):
            req = json.loads((await receive())["body"])
            self.assertIn("prompt_token_ids", req["kv_transfer_params"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            for chunk in chunks:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})

        mw = self.make_middleware(inner_app)
        sent = run_asgi(
            mw,
            {"type": "http", "path": "/v1/chat/completions", "headers": []},
            json.dumps({"model": "qwen3", "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
        )
        self.assertEqual([m["body"] for m in sent[1:]], chunks)

    def test_restores_length_finish_reason_on_truncated_tool_call(self):
        async def inner_app(scope, receive, send):
            body = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                            "finish_reason": "tool_calls",
                            "stop_reason": None,
                            "token_ids": [101, 102, 103, 104],
                        }
                    ]
                }
            ).encode()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": body})

        mw = self.make_middleware(inner_app)
        sent = run_asgi(
            mw,
            {"type": "http", "path": "/v1/chat/completions", "headers": []},
            json.dumps(
                {
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "call"}],
                    "max_completion_tokens": 4,
                }
            ).encode(),
        )
        reply = json.loads(sent[1]["body"])
        self.assertEqual(reply["choices"][0]["finish_reason"], "length")

    def test_context_length_exceeded_returns_400(self):
        async def inner_app(scope, receive, send):
            raise AssertionError("Should not reach upstream app")

        mw = self.make_middleware(inner_app, max_model_len=8)
        sent = run_asgi(
            mw,
            {"type": "http", "path": "/v1/chat/completions", "headers": []},
            json.dumps({"model": "qwen3", "messages": [{"role": "user", "content": "too long prompt"}]}).encode(),
        )
        self.assertEqual(sent[0]["status"], 400)
        err = json.loads(sent[1]["body"])["error"]
        self.assertEqual(err["code"], "context_length_exceeded")
        self.assertEqual(err["type"], "invalid_request_error")

    def test_non_chat_paths_pass_through(self):
        called = []

        async def inner_app(scope, receive, send):
            called.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = self.make_middleware(inner_app)
        sent = run_asgi(mw, {"type": "http", "path": "/health", "headers": []})
        self.assertEqual(called, ["/health"])
        self.assertEqual(sent[1]["body"], b"ok")

    def test_argv_model_and_tokenizer_parsing(self):
        from vllm_literal_tokens.middleware import _cli_args_from_argv

        self.assertEqual(
            _model_from_argv(["serve", "Qwen/Qwen3-32B", "--port", "8000"]),
            ("Qwen/Qwen3-32B", None),
        )
        self.assertEqual(
            _cli_args_from_argv(
                ["--model", "/models/qwen", "--tokenizer=/models/tok", "--generation-config", "/models/gen"]
            ),
            ("/models/qwen", "/models/tok", "/models/gen"),
        )

    def test_default_chat_template_kwargs_merged_under_request_overrides(self):
        mw = LiteralChatMiddleware(
            app=None,
            default_chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "xhigh"},
        )
        mw.protocol = make_protocol()
        wire = mw.prepare_request(
            {
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"reasoning_effort": "low"},
            }
        )
        self.assertEqual(wire["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(wire["chat_template_kwargs"]["reasoning_effort"], "low")
        self.assertEqual(wire["chat_template_kwargs"]["preserve_thinking"], True)

    def test_cli_outputs_plugin_path_and_flags(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(cli_main(["plugin-path"]), 0)
        self.assertEqual(buf.getvalue().strip(), str(parser_plugin_path()))
        self.assertTrue(parser_plugin_path().is_file())

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(cli_main(["flags"]), 0)
        out = buf.getvalue().strip()
        self.assertIn("--tool-call-parser qwen_token_ids", out)
        self.assertIn("--reasoning-parser qwen_token_ids", out)
        self.assertIn("--middleware vllm_literal_tokens.LiteralChatMiddleware", out)

        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(cli_main(["flags", "--explicit-plugin"]), 0)
        explicit_out = buf.getvalue().strip()
        self.assertIn("--tool-parser-plugin", explicit_out)
        self.assertIn("--reasoning-parser-plugin", explicit_out)

    def test_app_state_discovery_extracts_max_len_and_generation_config(self):
        from vllm_literal_tokens.middleware import _extract_from_app_state

        fake_tok = SimpleNamespace(name="tok")
        fake_app = SimpleNamespace(
            state=SimpleNamespace(
                openai_serving_chat=SimpleNamespace(
                    renderer=SimpleNamespace(get_tokenizer=lambda: fake_tok),
                    model_config=SimpleNamespace(
                        tokenizer="/models/qwen3",
                        model="/models/qwen3",
                        max_model_len=131072,
                        generation_config={"eos_token_id": [151645, 151643]},
                    ),
                )
            )
        )
        tok, model_ref, max_len, gen_cfg = _extract_from_app_state(fake_app)
        self.assertIs(tok, fake_tok)
        self.assertEqual(model_ref, "/models/qwen3")
        self.assertEqual(max_len, 131072)
        self.assertEqual(gen_cfg, {"eos_token_id": [151645, 151643]})


if __name__ == "__main__":
    unittest.main()

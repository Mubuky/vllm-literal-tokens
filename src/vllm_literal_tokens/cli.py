"""Small CLI helper for discovering plugin paths and ready-to-use vLLM flags."""
from __future__ import annotations

import argparse
import shlex

from . import parser_plugin_path

PARSER_NAME = "qwen_token_ids"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vllm-literal-tokens",
        description="Helpers for launching vLLM with literal special-token protection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plugin-path", help="Print the path to the Qwen token-ID parser plugin.")
    flags_cmd = sub.add_parser(
        "flags", help="Print vLLM CLI flags to enable literal-safe tokenization and parsing."
    )
    flags_cmd.add_argument(
        "--explicit-plugin",
        action="store_true",
        help="Also include --tool-parser-plugin and --reasoning-parser-plugin paths.",
    )

    args = parser.parse_args(argv)
    if args.command == "plugin-path":
        print(parser_plugin_path())
        return 0

    flags = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        PARSER_NAME,
        "--reasoning-parser",
        PARSER_NAME,
        "--middleware",
        "vllm_literal_tokens.LiteralChatMiddleware",
    ]
    if args.explicit_plugin:
        plugin = str(parser_plugin_path())
        flags = [
            "--tool-parser-plugin",
            plugin,
            "--reasoning-parser-plugin",
            plugin,
            *flags,
        ]
    print(shlex.join(flags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

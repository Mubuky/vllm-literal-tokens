"""Keep literal special-token strings as ordinary text in vLLM Chat Completions."""
from pathlib import Path

from .encoder import EncodingError, LiteralTokenizer, Span
from .middleware import ContextLengthError, LiteralChatMiddleware

__all__ = [
    "ContextLengthError",
    "EncodingError",
    "LiteralChatMiddleware",
    "LiteralTokenizer",
    "Span",
    "parser_plugin_path",
]
__version__ = "0.1.0"


def parser_plugin_path() -> Path:
    """Return the filesystem path to the Qwen token-ID vLLM parser plugin."""
    return Path(__file__).resolve().with_name("qwen_parser.py")

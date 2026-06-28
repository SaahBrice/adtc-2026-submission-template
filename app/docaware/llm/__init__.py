"""docaware.llm — Local GGUF chat model wrapper and prompt templates."""

from .client import LLMClient, get_llm
from . import prompts

__all__ = ["LLMClient", "get_llm", "prompts"]

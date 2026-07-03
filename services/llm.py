# llm.py
"""
Cached Ollama LLM client.

get_llm(model) is keyed by model name so the model router (agentic layer)
can cheaply switch between a "small" and "large" Ollama model without
reconnecting every call.
"""

from functools import lru_cache

from langchain_community.llms.ollama import Ollama

from config import LLM_MODEL


@lru_cache(maxsize=8)
def get_llm(model: str = LLM_MODEL) -> Ollama:
    return Ollama(model=model)

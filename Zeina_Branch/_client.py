"""LLM client construction and API-key loading.

Supports dual-model architecture:
  - SPELLER model: cheap, fast word completion (Groq by default)
  - RESPONSE model: smart, contextual sentence responses (Gemini by default)

Both use OpenAI-compatible endpoints. Clients are cached, warm, and share a
TCP/TLS connection pool. See LATENCY.md §"Persistent client + connection pool".
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_CLIENT_TIMEOUT_SECONDS = 15.0
_DEFAULT_SPELLER_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_RESPONSE_MODEL = "gemini-3-flash-preview"


def _load_api_key(llm_type: str = "SPELLER") -> str:
    key = os.environ.get(f"{llm_type}_API_KEY", "").strip()
    if not key or key.endswith("-replace-me"):
        raise RuntimeError(
            f"{llm_type}_API_KEY is missing. Copy .env.example to .env and paste "
            f"your provider key ({llm_type} by default). See KEY_SETUP.md for the "
            "walkthrough."
        )
    return key


def _load_base_url(llm_type: str = "SPELLER") -> str | None:
    """Optional base URL override.

    Supports:
    - `OPENAI_API_BASE_URL` (repo convention)
    - `OPENAI_BASE_URL` (OpenAI SDK convention)

    Leave unset to use the OpenAI SDK default.
    """
    base_url = (
        os.environ.get(f"{llm_type}_API_BASE_URL") or os.environ.get(f"{llm_type}_BASE_URL") or ""
    ).strip()
    return base_url or None


@lru_cache(maxsize=2)
def get_client(llm_type: str = "SPELLER") -> OpenAI:
    """Get a cached OpenAI client for the specified LLM type (SPELLER or RESPONSE)."""
    base_url = _load_base_url(llm_type)
    client = OpenAI(
        api_key=_load_api_key(llm_type),
        base_url=base_url,
        timeout=_CLIENT_TIMEOUT_SECONDS,
    )
    if "localhost" in (base_url or ""):
        # warm up local instance
        model = _get_model_for_type(llm_type)
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "ping"}],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to warm up local LLM API: {e}") from e
    return client


def _get_model_for_type(llm_type: str = "SPELLER") -> str:
    """Load model name for the given LLM type from environment."""
    if llm_type == "RESPONSE":
        return os.getenv("RESPONSE_MODEL", _DEFAULT_RESPONSE_MODEL)
    return os.getenv("SPELLER_MODEL", _DEFAULT_SPELLER_MODEL)


def get_response(
    client: OpenAI,
    system_prompt: str,
    user_message: str,
    llm_type: str | None = None,
) -> str:
    """Call an LLM with system prompt and user message.
    
    Args:
        client: OpenAI client to use.
        system_prompt: System context for the model.
        user_message: User input to respond to.
        llm_type: Which model to use (SPELLER or RESPONSE). If None, uses SPELLER.
    
    Returns:
        The model's response text.
    """
    if llm_type is None:
        llm_type = "SPELLER"
    model = _get_model_for_type(llm_type)
    
    messages = [
        {"role": "user", "content": f"System Instruction: {system_prompt}\n\nUser Query: {user_message}"},
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        # Extract final text response
        if response.choices and response.choices[0].message.content:
            text = response.choices[0].message.content
            # Strip <think>...</think> reasoning blocks emitted by some models
            # (DeepSeek, QwQ, etc.) before the actual answer.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        raise ValueError("Received empty response from model")
        
    except Exception as e:
        raise RuntimeError(f"LLM API request failed: {e}") from e

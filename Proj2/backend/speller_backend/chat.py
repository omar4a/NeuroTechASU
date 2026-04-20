"""Brain↔ChatGPT conversational reply path.

The speller half turns brain signals into text. This half turns text into a
ChatGPT reply — closing the loop judges expect to see. Uses the same Groq-
backed OpenAI-compatible client as task5_speller_api, with its own system
prompt shaped for the demo (see `demo.py`).

Public surface:
    async ask_chat(history, *, timeout=...) -> str
        history: list of {"role": "user"|"assistant", "content": str} turns.
        returns: assistant reply, or a graceful fallback string on failure.
"""
from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache
from typing import Callable

from openai import OpenAI

from .demo import CHAT_TIMEOUT_SECONDS, DEMO_CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_FALLBACK_REPLY: str = "(ChatGPT is unreachable right now — try again.)"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.endswith("-replace-me"):
        raise RuntimeError(
            "OPENAI_API_KEY missing. The chat path uses the same key as "
            "task5_speller_api — see its KEY_SETUP.md."
        )
    base_url = (os.environ.get("OPENAI_API_BASE_URL") or "").strip() or None
    kwargs = {"api_key": api_key, "timeout": 10.0}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _sync_chat(history: list[dict]) -> str:
    client = _client()
    model = os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")
    messages = [{"role": "system", "content": DEMO_CHAT_SYSTEM_PROMPT}] + list(history)
    response = client.chat.completions.create(model=model, messages=messages)
    if not response.choices or not response.choices[0].message.content:
        return _FALLBACK_REPLY
    return response.choices[0].message.content.strip()


async def ask_chat(
    history: list[dict],
    *,
    timeout: float = CHAT_TIMEOUT_SECONDS,
    chat_fn: Callable[[list[dict]], str] | None = None,
) -> str:
    """Ask ChatGPT for a reply given conversation history. Never raises."""
    fn = chat_fn or _sync_chat
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, fn, list(history)),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("chat timed out (>%.1fs)", timeout)
        return "(That reply took too long. Let's try a shorter message.)"
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat failed: %s", exc)
        return _FALLBACK_REPLY

"""ChatGPT-reply path tests — mocked client, no network, no key required."""
from __future__ import annotations

import pytest

from speller_backend.chat import ask_chat


@pytest.mark.asyncio
async def test_ask_chat_returns_model_reply():
    history = [{"role": "user", "content": "hello"}]

    def fake(h):
        assert h == history
        return "Hi! Nice to meet you on-stage."

    reply = await ask_chat(history, chat_fn=fake)
    assert reply == "Hi! Nice to meet you on-stage."


@pytest.mark.asyncio
async def test_ask_chat_timeout_returns_graceful_fallback():
    def slow(h):
        import time
        time.sleep(1.0)
        return "never arrives"

    reply = await ask_chat([{"role": "user", "content": "x"}], timeout=0.1, chat_fn=slow)
    assert "too long" in reply.lower() or "unreachable" in reply.lower()


@pytest.mark.asyncio
async def test_ask_chat_exception_returns_graceful_fallback():
    def boom(h):
        raise RuntimeError("upstream down")

    reply = await ask_chat([{"role": "user", "content": "x"}], chat_fn=boom)
    assert "unreachable" in reply.lower() or "took too long" in reply.lower()


@pytest.mark.asyncio
async def test_ask_chat_with_multi_turn_history():
    turns = [
        {"role": "user",      "content": "hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user",      "content": "what is this?"},
    ]

    def fake(h):
        # The function must receive the full history in order.
        assert h == turns
        return "A live BCI demo."

    reply = await ask_chat(turns, chat_fn=fake)
    assert reply == "A live BCI demo."

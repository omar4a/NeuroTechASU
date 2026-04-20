"""Session state-machine tests: full cycle + auto-commit + underscore commit
+ SSVEP fallback when freq doesn't map to a prediction slot.

All tests use MockSSVEPConsumer and an injected `predict_fn` — no network,
no hardware.
"""
from __future__ import annotations

import asyncio

import pytest

from speller_backend.session import (
    SpellerSession,
    S_AWAITING_PREFIX, S_AWAITING_SSVEP, S_CLOSED,
)
from speller_backend.ssvep_consumer import MockSSVEPConsumer


def _predict_factory(words):
    def predict(*, prefix, context, sentence):
        return list(words)
    return predict


class _Recorder:
    """Captures backend->frontend messages in order."""
    def __init__(self):
        self.messages: list[dict] = []

    async def __call__(self, msg):
        self.messages.append(msg)

    def commands(self):
        return [m["command"] for m in self.messages if "command" in m]


@pytest.mark.asyncio
async def test_init_triggers_flashing():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0]),
        predict_fn=_predict_factory(["hello", "hope", "help"]),
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    # Give the p300 task a tick to start.
    await asyncio.sleep(0.01)
    assert session.state == S_AWAITING_PREFIX
    assert send.commands()[0] == "start_flashing"
    # Demo persona is layered on top of the frontend-selected context.
    assert "BR41N.IO" in session.context
    assert "casual chat with a friend" in session.context
    await session.close()


@pytest.mark.asyncio
async def test_full_cycle_autocommit_prefix_then_ssvep_pick():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0]),            # pick word index 0 ("hello")
        predict_fn=_predict_factory(["hello", "hope", "help"]),
        prefix_auto_commit=2,
        ssvep_timeout=1.0,
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)

    # Feed two P300 chars; auto-commit should fire after the second.
    session.feed_p300_char("H")
    session.feed_p300_char("E")

    # Allow the async chain to run.
    for _ in range(40):
        await asyncio.sleep(0.02)
        if session.sentence:
            break

    cmds = send.commands()
    # Expected ordering (ignoring interleaving details):
    # start_flashing, type_char(H), type_char(E), stop_flashing,
    # update_predictions, start_ssvep, stop_ssvep, backspace, type_char(h), type_char(e),
    # type_char(l), type_char(l), type_char(o), type_char(' '), start_flashing (next cycle)
    assert "start_flashing" in cmds
    assert cmds.count("type_char") >= 2 + len("hello") + 1  # prefix + selected word + space
    assert "stop_flashing" in cmds
    assert "update_predictions" in cmds
    assert "start_ssvep" in cmds
    assert "stop_ssvep" in cmds
    assert "backspace" in cmds

    # update_predictions payload matches the injected predict_fn return.
    preds = [m for m in send.messages if m.get("command") == "update_predictions"]
    assert preds and preds[-1]["words"] == ["hello", "hope", "help"]

    # backspace count equals the prefix length we committed (2).
    bsps = [m for m in send.messages if m.get("command") == "backspace"]
    assert bsps and bsps[-1]["count"] == 2

    # sentence state reflects the committed word.
    assert "hello" in session.sentence

    await session.close()
    assert session.state == S_CLOSED


@pytest.mark.asyncio
async def test_underscore_commits_prefix_early():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([12.0]),            # pick index 1
        predict_fn=_predict_factory(["aa", "bb", "cc"]),
        prefix_auto_commit=5,                       # high, so underscore is needed
        ssvep_timeout=1.0,
    )
    await session.on_event({"event": "init", "context": "food", "timestamp": 0})
    await asyncio.sleep(0.01)

    session.feed_p300_char("A")
    session.feed_p300_char("_")                     # commit now

    for _ in range(40):
        await asyncio.sleep(0.02)
        if session.sentence:
            break

    assert "bb" in session.sentence
    await session.close()


@pytest.mark.asyncio
async def test_ssvep_unmapped_freq_restarts_prefix_cycle():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([7.5]),             # no mapping; should retry prefix
        predict_fn=_predict_factory(["hello", "hope", "help"]),
        prefix_auto_commit=1,
        ssvep_timeout=1.0,
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)

    session.feed_p300_char("H")

    # Let the first cycle resolve into a restart.
    for _ in range(40):
        await asyncio.sleep(0.02)
        # After unmapped SSVEP: stop_ssvep sent, start_flashing sent again
        cmds = send.commands()
        if cmds.count("start_flashing") >= 2:
            break

    cmds = send.commands()
    assert "stop_ssvep" in cmds
    assert cmds.count("start_flashing") >= 2
    # No word should have been committed.
    assert session.sentence == ""

    await session.close()


@pytest.mark.asyncio
async def test_flash_events_are_accepted_silently():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0]),
        predict_fn=_predict_factory(["a", "b", "c"]),
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)
    # Flash events are logged but don't emit any backend command.
    before = len(send.messages)
    await session.on_event({"event": "flash", "target": "row_3", "timestamp": 1.0})
    await session.on_event({"event": "flash", "target": "col_2", "timestamp": 1.1})
    after = len(send.messages)
    assert after == before

    await session.close()


@pytest.mark.asyncio
async def test_reinit_in_non_idle_state_is_ignored():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0]),
        predict_fn=_predict_factory(["a", "b", "c"]),
        demo_context_override="plain ctx",
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)
    cmds_before = list(send.commands())
    await session.on_event({"event": "init", "context": "medical", "timestamp": 1})
    await asyncio.sleep(0.01)
    assert send.commands() == cmds_before  # no new commands emitted
    assert session.context == "plain ctx"  # unchanged
    await session.close()


@pytest.mark.asyncio
async def test_trigger_chatgpt_reply_sends_command_and_grows_history():
    send = _Recorder()

    def fake_chat(history):
        # history should contain the user sentence we just committed
        assert history[-1]["role"] == "user"
        return "Nice to meet you, live from BR41N.IO!"

    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0]),
        predict_fn=_predict_factory(["hello", "hope", "help"]),
        chat_fn=fake_chat,
        prefix_auto_commit=2,
        typewriter_interval_ms=0,            # keep the test fast
        ssvep_timeout=1.0,
        demo_context_override="demo",
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)

    # Feed enough P300 chars to commit a word
    session.feed_p300_char("H")
    session.feed_p300_char("E")
    for _ in range(40):
        await asyncio.sleep(0.02)
        if session.sentence:
            break

    reply = await session.trigger_chatgpt_reply()
    assert reply == "Nice to meet you, live from BR41N.IO!"
    assert any(
        m.get("command") == "chatgpt_reply" and m["text"] == reply
        for m in send.messages
    )
    # history now has user + assistant
    assert len(session.history) == 2
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"
    # pending sentence was consumed
    assert session.sentence == ""
    await session.close()


@pytest.mark.asyncio
async def test_trigger_chatgpt_reply_noop_when_empty():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0]),
        predict_fn=_predict_factory(["a", "b", "c"]),
        chat_fn=lambda h: "shouldn't be called",
        demo_context_override="demo",
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)
    # sentence is still empty — trigger should short-circuit
    result = await session.trigger_chatgpt_reply()
    assert result is None
    assert not any(m.get("command") == "chatgpt_reply" for m in send.messages)
    await session.close()


@pytest.mark.asyncio
async def test_typewriter_pacing_applied_between_chars():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0]),
        predict_fn=_predict_factory(["hello", "hope", "help"]),
        prefix_auto_commit=2,
        typewriter_interval_ms=20,           # small but nonzero
        ssvep_timeout=1.0,
        demo_context_override="demo",
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)

    import time
    t0 = time.monotonic()
    session.feed_p300_char("H")
    session.feed_p300_char("E")
    for _ in range(80):
        await asyncio.sleep(0.02)
        if session.sentence:
            break
    elapsed = time.monotonic() - t0

    # "hello " = 6 chars × 20ms = 120ms minimum from typewriter alone.
    # The test passes if the elapsed time is clearly longer than the mocked
    # ssvep decision + prediction (both ~instant with mocks).
    assert elapsed >= 0.10
    await session.close()


@pytest.mark.asyncio
async def test_predictions_see_chatgpt_reply_as_conversation_context():
    """After ChatGPT replies, the NEXT prediction's context should include
    the reply so the subject's second message responds coherently."""
    send = _Recorder()
    seen_contexts: list[str] = []

    def predict(*, prefix, context, sentence):
        seen_contexts.append(context)
        return ["thank", "you", "so"]

    def chat_fn(h):
        return "Welcome to BR41N.IO! How can I help?"

    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0, 10.0]),    # two successive word picks
        predict_fn=predict,
        chat_fn=chat_fn,
        prefix_auto_commit=2,
        typewriter_interval_ms=0,
        ssvep_timeout=1.0,
        demo_context_override="base-ctx",
    )
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)

    # Round 1: type prefix, commit, trigger ChatGPT reply
    session.feed_p300_char("H")
    session.feed_p300_char("I")
    for _ in range(40):
        await asyncio.sleep(0.02)
        if session.sentence:
            break
    await session.trigger_chatgpt_reply()

    # Round 2: type another prefix — its context must include the reply.
    session.feed_p300_char("T")
    session.feed_p300_char("H")
    for _ in range(40):
        await asyncio.sleep(0.02)
        if len(seen_contexts) >= 2:
            break

    # First prediction saw only the base context.
    assert seen_contexts[0] == "base-ctx"
    # Second prediction was enriched with the ChatGPT reply.
    assert "Welcome to BR41N.IO" in seen_contexts[1]
    await session.close()


@pytest.mark.asyncio
async def test_demo_context_is_injected_when_no_override():
    send = _Recorder()
    session = SpellerSession(
        send=send,
        ssvep=MockSSVEPConsumer([10.0]),
        predict_fn=_predict_factory(["a", "b", "c"]),
    )  # no demo_context_override
    await session.on_event({"event": "init", "context": "chat", "timestamp": 0})
    await asyncio.sleep(0.01)
    # The specialized BR41N.IO persona must be in the context string.
    assert "BR41N.IO" in session.context
    assert "Ain Shams" in session.context
    await session.close()

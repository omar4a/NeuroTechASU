"""Spelling session — state machine for one WebSocket connection.

Drives the cycle:
    awaiting_prefix  ->  awaiting_predictions  ->  awaiting_ssvep  ->  awaiting_prefix ...

Inputs:
    * `on_event(payload)`   — called for each frontend event (init, flash)
    * `feed_p300_char(ch)`  — called by the HTTP layer when the P300 classifier
                              decodes a character
    * SSVEPConsumer.next_prediction() — pulled internally during awaiting_ssvep

Outputs (via the `send` coroutine injected at construction):
    * backend command dicts from `protocol.py`
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from . import protocol
from .chat import ask_chat
from .contexts import resolve as resolve_context
from .demo import DEMO_SPELLER_CONTEXT, TYPEWRITER_INTERVAL_MS
from .speller_glue import get_predictions

logger = logging.getLogger(__name__)

# Frontend draws prediction box 1/2/3 flickering at 10/12/15 Hz respectively.
FREQ_TO_WORD_IDX: dict[float, int] = {10.0: 0, 12.0: 1, 15.0: 2}

# State literals
S_IDLE = "idle"
S_AWAITING_PREFIX = "awaiting_prefix"
S_AWAITING_PREDICTIONS = "awaiting_predictions"
S_AWAITING_SSVEP = "awaiting_ssvep"
S_AWAITING_REPLY = "awaiting_reply"
S_CLOSED = "closed"


class SpellerSession:
    def __init__(
        self,
        send: Callable[[dict], Awaitable[None]],
        ssvep,                   # SSVEPConsumer
        *,
        prefix_auto_commit: int = 2,
        predict_fn: Callable[..., list[str]] | None = None,
        chat_fn: Callable[[list[dict]], str] | None = None,
        ssvep_timeout: float = 30.0,
        typewriter_interval_ms: int = TYPEWRITER_INTERVAL_MS,
        demo_context_override: str | None = None,
        continuation_enabled: bool = True,
    ):
        self._send = send
        self._ssvep = ssvep
        self._prefix_auto_commit = max(1, prefix_auto_commit)
        self._predict_fn = predict_fn
        self._chat_fn = chat_fn
        self._ssvep_timeout = ssvep_timeout
        self._typewriter_s = max(0, typewriter_interval_ms) / 1000.0
        self._demo_context_override = demo_context_override
        self._continuation_enabled = continuation_enabled

        self.state: str = S_IDLE
        self.context: str = ""               # the predict_words context string
        self.prefix: str = ""                # current word prefix from P300
        self.sentence: str = ""              # pending user message composed so far
        self.predictions: list[str] = []
        # Full Brain↔ChatGPT conversation. One pair of {user, assistant} turns
        # per round-trip. Used as the history passed to ask_chat().
        self.history: list[dict] = []

        self._p300_queue: asyncio.Queue[str] = asyncio.Queue()
        self._closed = asyncio.Event()
        self._p300_task: asyncio.Task | None = None
        self._ssvep_task: asyncio.Task | None = None

    # --- frontend events -----------------------------------------------------

    async def on_event(self, payload: dict) -> None:
        event = protocol.parse_event(payload)
        if isinstance(event, protocol.InitEvent):
            await self._on_init(event)
        elif isinstance(event, protocol.FlashEvent):
            logger.debug("flash %s @ %.3f", event.target, event.timestamp)
        else:
            logger.debug("ignoring unknown event payload: %r", payload)

    async def _on_init(self, ev: protocol.InitEvent) -> None:
        if self.state != S_IDLE:
            logger.info("re-init received in state %s; ignoring", self.state)
            return
        # Demo build: we layer the specialized BR41N.IO persona on top of the
        # frontend-selected context. If the caller passed an override (tests),
        # that wins.
        frontend_ctx = resolve_context(ev.context)
        if self._demo_context_override is not None:
            self.context = self._demo_context_override
        else:
            self.context = (
                f"{DEMO_SPELLER_CONTEXT} Sub-context: {frontend_ctx}."
                if frontend_ctx else DEMO_SPELLER_CONTEXT
            )
        logger.info("init: context=%r", self.context[:80] + "…")
        await self._ssvep.start()
        await self._transition_to_prefix()

    # --- P300 injection (HTTP POST / test) -----------------------------------

    def feed_p300_char(self, char: str) -> None:
        """Thread-safe: may be called from any context."""
        try:
            self._p300_queue.put_nowait(char)
        except asyncio.QueueFull:  # our queue is unbounded, but defensive
            logger.warning("p300 queue full; dropping char %r", char)

    # --- state transitions ---------------------------------------------------

    async def _transition_to_prefix(self) -> None:
        self.state = S_AWAITING_PREFIX
        self.prefix = ""
        await self._send(protocol.start_flashing())
        if self._p300_task is None or self._p300_task.done():
            self._p300_task = asyncio.create_task(self._consume_p300())

    async def _consume_p300(self) -> None:
        """Long-lived task: pulls P300 chars off the queue whenever we're
        in awaiting_prefix state."""
        while not self._closed.is_set():
            try:
                char = await asyncio.wait_for(self._p300_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self.state == S_AWAITING_PREFIX:
                await self._on_p300_char(char)
            # chars arriving outside awaiting_prefix are silently dropped

    async def _on_p300_char(self, char: str) -> None:
        char = str(char).strip()
        if len(char) != 1:
            logger.debug("skipping non-single-char p300 input: %r", char)
            return
        if char == "_":
            # underscore cell doubles as "commit prefix now"
            if self.prefix:
                await self._transition_to_predictions()
            return
        self.prefix += char.lower()
        await self._send(protocol.type_char(char))
        if len(self.prefix) >= self._prefix_auto_commit:
            await self._transition_to_predictions()

    async def _transition_to_predictions(self) -> None:
        if self.state != S_AWAITING_PREFIX:
            return
        self.state = S_AWAITING_PREDICTIONS
        await self._send(protocol.stop_flashing())
        words = await get_predictions(
            prefix=self.prefix,
            context=self._conversation_aware_context(),
            sentence=self.sentence,
            predict_fn=self._predict_fn,
        )
        self.predictions = words
        await self._send(protocol.update_predictions(words))
        await self._send(protocol.start_ssvep())
        self.state = S_AWAITING_SSVEP
        self._ssvep_task = asyncio.create_task(self._consume_ssvep())

    async def _consume_ssvep(self) -> None:
        freq = await self._ssvep.next_prediction(timeout=self._ssvep_timeout)
        if self._closed.is_set() or self.state != S_AWAITING_SSVEP:
            return
        if freq is None:
            # Timeout: classifier didn't confidently lock on a target. This is
            # the intended escape hatch — subject looked away from the 3
            # SSVEP boxes. Fall back to P300 for a fresh prefix entry. See
            # trial&error.md on why this requires classifier reset + threshold
            # filtering upstream to actually fire on real hardware.
            logger.info("ssvep timeout (no confident pick); falling back to P300")
            await self._send(protocol.stop_ssvep())
            await self._ssvep_reset_safe()
            await self._transition_to_prefix()
            return
        idx = FREQ_TO_WORD_IDX.get(float(freq))
        if idx is None or idx >= len(self.predictions):
            logger.warning(
                "ssvep returned %r (idx=%r); cannot commit; falling back to P300",
                freq, idx,
            )
            await self._send(protocol.stop_ssvep())
            await self._ssvep_reset_safe()
            await self._transition_to_prefix()
            return
        selected = self.predictions[idx]
        await self._send(protocol.stop_ssvep())
        # Delete live-typed prefix (if any — continuation picks have no
        # prefix), then type the full selected word with a typewriter pace.
        if self.prefix:
            await self._send(protocol.backspace(len(self.prefix)))
        await self._typewrite(selected + " ")
        self.sentence = (self.sentence + " " + selected).strip() + " "
        logger.info("committed %r; sentence now %r", selected, self.sentence)
        await self._ssvep_reset_safe()
        if self._continuation_enabled and self.sentence.strip():
            # Option A: skip P300, go straight to next-word prediction seeded
            # by the growing sentence.
            await self._transition_to_continuation()
        else:
            await self._transition_to_prefix()

    async def _transition_to_continuation(self) -> None:
        """After a word commits, predict the NEXT word directly from the
        sentence-so-far — no P300. If the SSVEP classifier fails to lock in
        the next cycle (timeout / unmapped), we drop back to _transition_to_
        prefix() so the subject can type fresh letters."""
        self.state = S_AWAITING_PREDICTIONS
        self.prefix = ""
        words = await get_predictions(
            prefix="",
            context=self._conversation_aware_context(),
            sentence=self.sentence,
            predict_fn=self._predict_fn,
        )
        self.predictions = words
        await self._send(protocol.update_predictions(words))
        await self._send(protocol.start_ssvep())
        self.state = S_AWAITING_SSVEP
        self._ssvep_task = asyncio.create_task(self._consume_ssvep())

    async def _ssvep_reset_safe(self) -> None:
        """Call ssvep.reset() if the consumer exposes one. Swallow errors —
        reset is best-effort hygiene, never demo-blocking."""
        reset = getattr(self._ssvep, "reset", None)
        if reset is None:
            return
        try:
            await reset()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ssvep.reset raised: %s", exc)

    def _conversation_aware_context(self) -> str:
        """Enrich the speller context with ChatGPT's most recent reply so the
        subject's next message is predicted as a natural continuation of the
        conversation rather than as an isolated utterance.
        """
        last_assistant = next(
            (m["content"] for m in reversed(self.history) if m.get("role") == "assistant"),
            None,
        )
        if not last_assistant:
            return self.context
        return (
            f"{self.context} "
            f"The user is composing a reply to ChatGPT's last message: "
            f"\"{last_assistant}\"."
        )

    async def _typewrite(self, text: str) -> None:
        """Type a string one char at a time with a pacing delay.

        Used only for SSVEP-committed words — live P300 feedback stays instant.
        """
        for ch in text:
            await self._send(protocol.type_char(ch))
            if self._typewriter_s:
                await asyncio.sleep(self._typewriter_s)

    # --- Brain↔ChatGPT send trigger ----------------------------------------

    async def trigger_chatgpt_reply(self) -> str | None:
        """Send the current pending sentence to ChatGPT and deliver the reply
        via the `chatgpt_reply` command. Operator-driven (HTTP /send_message).

        The full flow:
          1. Cancel the in-flight SSVEP-consume task so an old, stale
             prediction can't commit a garbage word against the cleared
             sentence after the reply arrives (smoke-test bug fix).
          2. Reset the SSVEP consumer (drop stale queued predictions).
          3. Tell the frontend to stop both its active display modes.
          4. Ask ChatGPT with the full conversation history.
          5. Send the reply.
          6. Start a fresh P300 prefix cycle — the subject's next utterance
             is a new turn, not a continuation of the one just sent.

        Returns the reply string, or None if there was nothing to send.
        """
        message = self.sentence.strip()
        if not message:
            logger.info("send triggered but sentence is empty; ignoring")
            return None
        # (1) pause any in-flight cycle
        if self._ssvep_task is not None and not self._ssvep_task.done():
            self._ssvep_task.cancel()
            self._ssvep_task = None
        # (2) drop stale SSVEP queue / classifier state
        await self._ssvep_reset_safe()
        # (3) make the UI idle while we wait for ChatGPT
        await self._send(protocol.stop_ssvep())
        await self._send(protocol.stop_flashing())
        self.state = S_AWAITING_REPLY
        self.history.append({"role": "user", "content": message})
        self.sentence = ""
        # (4) ask
        reply = await ask_chat(self.history, chat_fn=self._chat_fn)
        self.history.append({"role": "assistant", "content": reply})
        # (5) deliver
        await self._send(protocol.chatgpt_reply(reply))
        logger.info("ChatGPT replied %r", reply[:80])
        # (6) new utterance starts with P300 prefix
        if not self._closed.is_set():
            await self._transition_to_prefix()
        return reply

    # --- shutdown ------------------------------------------------------------

    async def close(self) -> None:
        if self.state == S_CLOSED:
            return
        self.state = S_CLOSED
        self._closed.set()
        try:
            await self._ssvep.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ssvep.stop raised: %s", exc)
        for t in (self._p300_task, self._ssvep_task):
            if t is not None and not t.done():
                t.cancel()

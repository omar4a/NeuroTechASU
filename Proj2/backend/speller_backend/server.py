"""WebSocket + HTTP server. Single spelling session at a time.

- WebSocket on `SPELLER_WS_PORT` (default 8765). The frontend connects here.
- HTTP on `SPELLER_HTTP_PORT` (default 8766). The P300 classifier (or curl)
  POSTs {"char": "H"} to /p300/char to inject a decoded character into the
  current session.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from aiohttp import web
import websockets

from .demo import PREWARM_DELAY_SECONDS
from .session import SpellerSession
from .speller_glue import get_predictions
from .ssvep_consumer import MockSSVEPConsumer, RealSSVEPConsumer

logger = logging.getLogger(__name__)


# --- single-tenant session registry -----------------------------------------

_current_session: Optional[SpellerSession] = None
_session_lock = asyncio.Lock()


def _make_ssvep():
    mode = os.environ.get("SPELLER_SSVEP_MODE", "mock").lower()
    if mode == "real":
        logger.info("ssvep mode: real (Unicorn via LSL)")
        return RealSSVEPConsumer()
    logger.info("ssvep mode: mock (10,12,15 Hz repeating)")
    return MockSSVEPConsumer([10.0, 12.0, 15.0] * 100)


# --- WebSocket ---------------------------------------------------------------

async def _ws_handler(ws):
    global _current_session
    async with _session_lock:
        if _current_session is not None:
            logger.info("new client connected; closing prior session")
            await _current_session.close()
        session = SpellerSession(
            send=lambda msg, _ws=ws: _ws.send(json.dumps(msg)),
            ssvep=_make_ssvep(),
            prefix_auto_commit=int(os.environ.get("SPELLER_PREFIX_LENGTH", "2")),
        )
        _current_session = session

    try:
        async for raw in ws:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("non-JSON frame from client: %r", raw[:80])
                continue
            await session.on_event(payload)
    except websockets.ConnectionClosed:
        logger.info("websocket closed by client")
    finally:
        async with _session_lock:
            if _current_session is session:
                _current_session = None
        await session.close()


# --- HTTP --------------------------------------------------------------------

async def _http_p300(request: web.Request) -> web.Response:
    session = _current_session
    if session is None:
        return web.json_response({"error": "no active session"}, status=409)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "body must be JSON"}, status=400)
    char = str(body.get("char", ""))
    if len(char) != 1:
        return web.json_response(
            {"error": "char must be a single character"}, status=400
        )
    session.feed_p300_char(char)
    return web.json_response({"ok": True})


async def _http_health(request: web.Request) -> web.Response:
    return web.json_response({
        "ok": True,
        "active_session": _current_session is not None,
        "state": _current_session.state if _current_session else None,
        "sentence": _current_session.sentence if _current_session else None,
        "turns": len(_current_session.history) if _current_session else 0,
    })


async def _http_send(request: web.Request) -> web.Response:
    """Operator-driven: send the current pending sentence to ChatGPT.

    This is the Brain↔ChatGPT payoff for the demo — Omar (narrator) calls this
    when the subject's sentence is complete enough for a reply. Keeping the
    trigger operator-controlled means the subject doesn't need a punctuation
    cell on the P300 grid.
    """
    session = _current_session
    if session is None:
        return web.json_response({"error": "no active session"}, status=409)
    reply = await session.trigger_chatgpt_reply()
    if reply is None:
        return web.json_response(
            {"ok": False, "reason": "no pending sentence"}, status=409
        )
    return web.json_response({"ok": True, "reply": reply})


# --- entrypoint --------------------------------------------------------------

async def run_server() -> None:
    ws_host = os.environ.get("SPELLER_WS_HOST", "0.0.0.0")
    ws_port = int(os.environ.get("SPELLER_WS_PORT", "8765"))
    http_port = int(os.environ.get("SPELLER_HTTP_PORT", "8766"))

    ws_server = await websockets.serve(_ws_handler, ws_host, ws_port)
    logger.info("websocket: ws://%s:%d", ws_host, ws_port)

    http_app = web.Application()
    http_app.router.add_post("/p300/char", _http_p300)
    http_app.router.add_post("/send_message", _http_send)
    http_app.router.add_get("/health", _http_health)
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site = web.TCPSite(http_runner, ws_host, http_port)
    await http_site.start()
    logger.info(
        "http: http://%s:%d (POST /p300/char, POST /send_message, GET /health)",
        ws_host, http_port,
    )

    # Pre-warm the LLM client so the subject's first word is instant on stage.
    asyncio.create_task(_prewarm_llm())

    try:
        await asyncio.Future()  # run forever
    finally:
        ws_server.close()
        await ws_server.wait_closed()
        await http_runner.cleanup()


async def _prewarm_llm() -> None:
    """Fire a throwaway predict_words + chat call at startup so the first
    real user word doesn't eat Groq's cold-start latency.

    Failures here are non-fatal — the server continues even if pre-warm can't
    reach Groq (e.g., missing key in dev).
    """
    await asyncio.sleep(PREWARM_DELAY_SECONDS)
    try:
        await get_predictions(prefix="hi", context="warmup", timeout=4.0)
        logger.info("llm prewarm: predict_words ok")
    except Exception as exc:  # noqa: BLE001
        logger.info("llm prewarm: predict_words skipped (%s)", exc)
    try:
        from .chat import ask_chat
        await ask_chat(
            [{"role": "user", "content": "hi"}], timeout=4.0
        )
        logger.info("llm prewarm: chat ok")
    except Exception as exc:  # noqa: BLE001
        logger.info("llm prewarm: chat skipped (%s)", exc)

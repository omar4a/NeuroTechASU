"""Async wrapper around task5_speller_api.predict_words.

The speller API is synchronous (OpenAI SDK under the hood). We run it in
asyncio's default thread pool with a timeout so the WebSocket event loop is
never blocked waiting for the LLM.

This function NEVER raises — on timeout or API error it returns the
documented fallback list. Callers can index [0]/[1]/[2] without defending.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

FALLBACK: list[str] = ["the", "and", "of"]


async def get_predictions(
    prefix: str,
    context: str,
    sentence: str = "",
    *,
    timeout: float = 5.0,
    predict_fn: Callable[..., list[str]] | None = None,
) -> list[str]:
    """Fetch 3 word predictions. Always returns a list of length 3."""
    if predict_fn is None:
        from task5_speller_api import predict_words as predict_fn  # local import: keeps test isolation clean

    loop = asyncio.get_running_loop()
    try:
        words = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: predict_fn(prefix=prefix, context=context, sentence=sentence),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("speller_api timed out (>%.1fs); returning fallback", timeout)
        return list(FALLBACK)
    except Exception as exc:  # noqa: BLE001 - any error degrades to fallback
        logger.warning("speller_api raised (%s); returning fallback", exc)
        return list(FALLBACK)

    return _normalize(words)


def _normalize(words) -> list[str]:
    """Coerce the speller return to exactly 3 lowercase strings."""
    if not isinstance(words, list):
        logger.warning("speller_api returned non-list %r; fallback", words)
        return list(FALLBACK)
    cleaned = [str(w).strip().lower() for w in words if str(w).strip()][:3]
    while len(cleaned) < 3:
        for fb in FALLBACK:
            if fb not in cleaned:
                cleaned.append(fb)
                break
    return cleaned[:3]

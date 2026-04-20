"""Message protocol between Proj2/ui (browser) and this backend.

Frontend -> Backend (events):
    {"event": "init",  "context": str, "timestamp": float}
    {"event": "flash", "target": str,  "timestamp": float}   # "row_<n>" or "col_<n>"

Backend -> Frontend (commands):
    {"command": "start_flashing"}
    {"command": "stop_flashing"}
    {"command": "start_ssvep"}
    {"command": "stop_ssvep"}
    {"command": "type_char", "char": str}
    {"command": "update_predictions", "words": [str, str, str]}
    {"command": "backspace", "count": int}
    {"command": "chatgpt_reply", "text": str}

The `backspace` command is new: when a word is selected via SSVEP the backend
needs to delete the live-typed prefix before typing the full word. Frontend
support for it is a one-liner in app.js (`typedTextSpan` slice) that lands
separately.

The `chatgpt_reply` command lands the Brain↔ChatGPT payoff on stage — the
operator triggers `POST /send_message` and this command carries ChatGPT's
reply back to the UI for rendering in a dedicated panel.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---- command builders (backend -> frontend) --------------------------------

def start_flashing() -> dict:
    return {"command": "start_flashing"}


def stop_flashing() -> dict:
    return {"command": "stop_flashing"}


def start_ssvep() -> dict:
    return {"command": "start_ssvep"}


def stop_ssvep() -> dict:
    return {"command": "stop_ssvep"}


def type_char(ch: str) -> dict:
    if not isinstance(ch, str) or len(ch) != 1:
        raise ValueError(f"type_char expects a single character, got {ch!r}")
    return {"command": "type_char", "char": ch}


def update_predictions(words: list[str]) -> dict:
    if not isinstance(words, list) or len(words) != 3:
        raise ValueError(f"update_predictions expects exactly 3 words, got {words!r}")
    if not all(isinstance(w, str) for w in words):
        raise ValueError(f"update_predictions words must all be str, got {words!r}")
    return {"command": "update_predictions", "words": list(words)}


def backspace(count: int) -> dict:
    if not isinstance(count, int) or count < 0:
        raise ValueError(f"backspace count must be non-negative int, got {count!r}")
    return {"command": "backspace", "count": count}


def chatgpt_reply(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError(f"chatgpt_reply text must be str, got {text!r}")
    return {"command": "chatgpt_reply", "text": text}


# ---- event parsers (frontend -> backend) -----------------------------------

@dataclass(frozen=True)
class InitEvent:
    context: str
    timestamp: float


@dataclass(frozen=True)
class FlashEvent:
    target: str
    timestamp: float


def parse_event(payload: dict):
    """Parse a frontend event dict. Returns an event dataclass or None."""
    if not isinstance(payload, dict):
        return None
    ev = payload.get("event")
    if ev == "init":
        return InitEvent(
            context=str(payload.get("context", "")),
            timestamp=float(payload.get("timestamp", 0.0) or 0.0),
        )
    if ev == "flash":
        return FlashEvent(
            target=str(payload.get("target", "")),
            timestamp=float(payload.get("timestamp", 0.0) or 0.0),
        )
    return None

"""Map the frontend `context-selector` dropdown values to free-text scenario
strings that speller_api's prompt uses.

The frontend offers three canned contexts today (`chat`, `medical`, `food`).
Anything unrecognised is passed through verbatim so future UI additions work
without a backend change.
"""
from __future__ import annotations

CONTEXTS: dict[str, str] = {
    "chat":    "casual chat with a friend",
    "medical": "medical consultation with a clinician",
    "food":    "ordering food at a restaurant",
}


def resolve(key: str) -> str:
    if not key:
        return ""
    return CONTEXTS.get(key, key)

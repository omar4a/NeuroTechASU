"""Speller glue tests: async wrapper hides sync speller_api.

No task5_speller_api import is required — we inject `predict_fn` directly so
the tests run without installing the speller_api package.
"""
from __future__ import annotations

import asyncio

import pytest

from speller_backend.speller_glue import FALLBACK, get_predictions


@pytest.mark.asyncio
async def test_happy_path_returns_words():
    def predict(*, prefix, context, sentence):
        assert prefix == "he"
        assert context == "email to professor"
        assert sentence == ""
        return ["hello", "hope", "help"]

    words = await get_predictions("he", "email to professor", predict_fn=predict)
    assert words == ["hello", "hope", "help"]


@pytest.mark.asyncio
async def test_timeout_returns_fallback():
    def predict(*, prefix, context, sentence):
        import time
        time.sleep(1.0)
        return ["x", "y", "z"]

    words = await get_predictions("he", "ctx", timeout=0.1, predict_fn=predict)
    assert words == FALLBACK


@pytest.mark.asyncio
async def test_exception_returns_fallback():
    def predict(*, prefix, context, sentence):
        raise RuntimeError("simulated")

    words = await get_predictions("he", "ctx", predict_fn=predict)
    assert words == FALLBACK


@pytest.mark.asyncio
async def test_non_list_return_becomes_fallback():
    def predict(*, prefix, context, sentence):
        return "not a list"

    words = await get_predictions("he", "ctx", predict_fn=predict)
    assert words == FALLBACK


@pytest.mark.asyncio
async def test_short_list_is_padded_to_three():
    def predict(*, prefix, context, sentence):
        return ["only"]

    words = await get_predictions("he", "ctx", predict_fn=predict)
    assert len(words) == 3
    assert words[0] == "only"


@pytest.mark.asyncio
async def test_long_list_is_truncated():
    def predict(*, prefix, context, sentence):
        return ["a", "b", "c", "d", "e"]

    words = await get_predictions("he", "ctx", predict_fn=predict)
    assert words == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_case_is_lowercased():
    def predict(*, prefix, context, sentence):
        return ["HELLO", "Hope", "Help"]

    words = await get_predictions("he", "ctx", predict_fn=predict)
    assert all(w == w.lower() for w in words)

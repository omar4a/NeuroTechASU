"""SSVEP consumer tests (mock only — real requires hardware)."""
from __future__ import annotations

import asyncio

import pytest

from speller_backend.ssvep_consumer import MockSSVEPConsumer, RealSSVEPConsumer


@pytest.mark.asyncio
async def test_mock_returns_programmed_sequence():
    consumer = MockSSVEPConsumer([10.0, 12.0, 15.0])
    await consumer.start()
    assert await consumer.next_prediction(timeout=1.0) == 10.0
    assert await consumer.next_prediction(timeout=1.0) == 12.0
    assert await consumer.next_prediction(timeout=1.0) == 15.0
    # exhausted queue returns None, not hang
    assert await consumer.next_prediction(timeout=1.0) is None
    await consumer.stop()


@pytest.mark.asyncio
async def test_mock_timeout_does_not_raise():
    consumer = MockSSVEPConsumer([])
    # With an empty queue, we get None promptly rather than hanging.
    result = await consumer.next_prediction(timeout=0.1)
    assert result is None


def test_real_upstream_path_resolves_relative_to_repo():
    """Sanity: RealSSVEPConsumer expects the upstream file at a path relative
    to the repo root. We don't instantiate the real streamer (needs hardware);
    we just verify the path pointer is shaped correctly."""
    expected_suffix = ("SSVEP Protocol", "ssvep_realtime.py")
    path = RealSSVEPConsumer._UPSTREAM_PATH
    assert path.parts[-2:] == expected_suffix

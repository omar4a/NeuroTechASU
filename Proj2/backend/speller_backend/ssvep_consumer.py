"""Sources of SSVEP frequency predictions for the session state machine.

Two implementations, both exposing:
    async start()
    async stop()
    async next_prediction(timeout: float | None = None) -> float | None

- MockSSVEPConsumer:
    Preprogrammed sequence. Used by tests and during dry-run rehearsal when no
    headset is attached. `timeout=None` waits forever; with a timeout, returns
    None on expiry.

- RealSSVEPConsumer:
    Loads Omar's `SSVEP Protocol/ssvep_realtime.py` via importlib (the folder
    name has a space in it and cannot be imported directly). Instantiates the
    upstream LSLStreamer with a callback that pushes predictions onto an
    asyncio.Queue on the event loop that called `.start()`. Requires pylsl +
    scipy + scikit-learn + numpy + a live Unicorn LSL stream.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import pathlib
from collections import deque
from typing import Iterable, Protocol

logger = logging.getLogger(__name__)


class SSVEPConsumer(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def next_prediction(self, timeout: float | None = None) -> float | None: ...


class MockSSVEPConsumer:
    """Returns preprogrammed frequencies in order. Handy for tests + rehearsal."""

    def __init__(self, sequence: Iterable[float]):
        self._queue: deque[float] = deque(sequence)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def next_prediction(self, timeout: float | None = None) -> float | None:
        # Simulate a small decision delay so the state machine has something to await.
        await asyncio.sleep(0.01)
        if not self._queue:
            return None
        return self._queue.popleft()


class RealSSVEPConsumer:
    """Wraps upstream SSVEPClassifier + LSLStreamer. Real hardware required."""

    # Path resolution: this file is at NeuroTechASU/Proj2/backend/speller_backend/ssvep_consumer.py
    # Upstream source is at NeuroTechASU/SSVEP Protocol/ssvep_realtime.py
    _UPSTREAM_PATH = (
        pathlib.Path(__file__).resolve().parents[3]
        / "SSVEP Protocol"
        / "ssvep_realtime.py"
    )

    def __init__(self):
        self._queue: asyncio.Queue[float] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._streamer = None

    def _load_upstream(self):
        if not self._UPSTREAM_PATH.exists():
            raise RuntimeError(
                f"SSVEP upstream source missing at {self._UPSTREAM_PATH}. "
                "This backend expects to live inside the NeuroTechASU repo."
            )
        spec = importlib.util.spec_from_file_location(
            "ssvep_realtime_upstream", str(self._UPSTREAM_PATH)
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        mod = self._load_upstream()

        def callback(event_type: str, value) -> None:
            if event_type != "prediction":
                return
            try:
                freq = float(value)
            except (TypeError, ValueError):
                return
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(self._queue.put(freq), self._loop)

        self._streamer = mod.LSLStreamer(callback=callback)
        self._streamer.start()
        logger.info("RealSSVEPConsumer: upstream LSLStreamer started")

    async def stop(self) -> None:
        if self._streamer is not None:
            self._streamer.stop()
            self._streamer = None

    async def next_prediction(self, timeout: float | None = None) -> float | None:
        try:
            if timeout is None:
                return await self._queue.get()
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

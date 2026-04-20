"""CLI: `python -m speller_backend` or the installed `speller-backend` script."""
from __future__ import annotations

import asyncio
import logging
import os

from .server import run_server


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("SPELLER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
AzureDiGate — asyncio.Semaphore wrapper that limits concurrent Azure DI calls.

Direct Python port of the Java ``AzureDiGate`` (Semaphore + AtomicLong counters).
The semaphore cap should be set at or below the concurrency tolerance of the
Azure Document Intelligence tier in use:

  - F0 free tier  : 1 concurrent call  → max_concurrent=1
  - S0 paid tier  : ~12-15 empirically → max_concurrent=12 (safe default)

Usage::

    gate = AzureDiGate(max_concurrent=12)

    async with gate:
        result = await azure_di_client.begin_analyze_document(...)
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class AzureDiGate:
    """
    Application-scoped concurrency gate for Azure Document Intelligence.

    A single shared ``asyncio.Semaphore`` limits the number of Azure DI calls
    in flight at any moment across ALL concurrent extraction tasks.  Without
    this, N documents submitted in parallel produce N simultaneous calls,
    which triggers 403/429 responses even on the S0 paid tier.

    The gate is safe to share across coroutines on the same event loop.

    Args:
        max_concurrent: Maximum simultaneous Azure DI calls.  Mirrors
            ``pipeline.azure-di-gate.max-concurrent`` from the Java config.
    """

    def __init__(self, max_concurrent: int = 12) -> None:
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._total_acquires = 0
        self._blocked_calls = 0
        logger.info("AzureDiGate initialised — max_concurrent=%d", max_concurrent)

    async def __aenter__(self) -> "AzureDiGate":
        self._total_acquires += 1
        logger.debug(
            "AzureDiGate acquire — total_acquires=%d max=%d",
            self._total_acquires,
            self._max_concurrent,
        )
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        self._semaphore.release()
        logger.debug("AzureDiGate release")

    def record_blocked(self) -> None:
        """Increment the blocked-call counter (QUOTA-403 or exhausted 429 retries)."""
        self._blocked_calls += 1

    @property
    def total_acquires(self) -> int:
        """Total Azure DI call attempts since the gate was created."""
        return self._total_acquires

    @property
    def blocked_calls(self) -> int:
        """Total blocked calls (QUOTA-403 + exhausted 429 retries)."""
        return self._blocked_calls

    @property
    def available_permits(self) -> int:
        """Current number of free slots."""
        return self._semaphore._value  # type: ignore[attr-defined]

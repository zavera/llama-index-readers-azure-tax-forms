"""
Core extraction engine — Python port of DocumentIntelligenceService.java.

Recovery chain (identical to the Java implementation):

  Stage 0  — direct Azure DI call on original bytes
  Stage 1  — page-split into chunks of ``pages_per_chunk``, analyse in parallel
  Stage 2  — DPI reduction to 300 DPI (rasterise)
  Stage 3  — rotation block: as-is → 90° → 180° → 270°

Concurrency:
  - ``asyncio.Semaphore`` (via :class:`AzureDiGate`) limits total in-flight calls
  - ``asyncio.gather`` runs split chunks in parallel (Java: virtual threads)
  - Exponential back-off with ±20 % jitter on HTTP 429 responses
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from llama_index_readers_azure_tax_forms.gate import AzureDiGate
from llama_index_readers_azure_tax_forms.models import KvEntry, ExtractionResult, TaxFormType
from llama_index_readers_azure_tax_forms.normalizer import normalize_pairs
from llama_index_readers_azure_tax_forms import pdf_utils

logger = logging.getLogger(__name__)

TARGET_DPI = 300


@dataclass
class ExtractionConfig:
    """
    Configuration for the extraction engine.

    Mirrors the ``@Value``-injected fields in ``DocumentIntelligenceService.java``.
    """

    endpoint: str
    api_key: str
    model_id: str = "prebuilt-document"
    pages_per_chunk: int = 10
    poll_timeout_seconds: int = 120
    # 429 retry back-off
    rate_limit_max_retries: int = 5
    rate_limit_initial_delay_ms: int = 1_000
    rate_limit_max_delay_ms: int = 32_000
    # Gate
    max_concurrent: int = 12


class TaxFormExtractor:
    """
    Async extraction engine for IRS tax form documents.

    Instantiate once and reuse across many documents.  Thread-safe for
    concurrent ``asyncio.gather`` calls from the same event loop.

    Args:
        config: Extraction configuration.
        gate:   Optional pre-constructed gate.  If ``None``, a new gate is
                created using ``config.max_concurrent``.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        gate: Optional[AzureDiGate] = None,
    ) -> None:
        self._config = config
        self._gate = gate or AzureDiGate(max_concurrent=config.max_concurrent)
        self._client = DocumentAnalysisClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.api_key),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, document_id: str, pdf_bytes: bytes) -> ExtractionResult:
        """
        Run the full extraction pipeline on ``pdf_bytes``.

        Args:
            document_id: Caller-supplied identifier (file path, UUID, S3 key,
                         etc.).  Used for logging only — no PII constraint
                         since this library is not FERPA-scoped.
            pdf_bytes:   Raw PDF file bytes.

        Returns:
            :class:`ExtractionResult` — always returns, never raises.
            On complete failure ``result.is_empty`` is ``True`` and
            ``result.error`` describes the failure.
        """
        start_ms = _now_ms()
        result = ExtractionResult(document_id=document_id)

        try:
            pairs, stage = await self._analyze_bytes(pdf_bytes, document_id)
            result.entries = [
                KvEntry(key=k, value=v, confidence=c) for k, v, c in pairs
            ]
            result.stage = stage
            result.form_type = _infer_form_type(result.entries)
        except Exception as exc:
            logger.error("Extraction failed for document_id=%s: %s", document_id, exc)
            result.error = str(exc)

        result.total_ms = _now_ms() - start_ms
        result.di_calls = self._gate.total_acquires
        return result

    async def extract_many(
        self,
        documents: list[tuple[str, bytes]],
    ) -> list[ExtractionResult]:
        """
        Extract KV pairs from multiple documents concurrently.

        The shared :class:`AzureDiGate` automatically limits concurrency to
        ``max_concurrent`` regardless of how many documents are submitted.

        Args:
            documents: List of ``(document_id, pdf_bytes)`` pairs.

        Returns:
            List of :class:`ExtractionResult`, one per input document, in the
            same order as the input.
        """
        tasks = [self.extract(doc_id, pdf_bytes) for doc_id, pdf_bytes in documents]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Recovery chain (mirrors Java analyzeBytes)
    # ------------------------------------------------------------------

    async def _analyze_bytes(
        self,
        pdf_bytes: bytes,
        document_id: str,
    ) -> tuple[list[tuple[Optional[str], Optional[str], Optional[float]]], str]:
        """
        Run the 4-stage recovery chain.  Returns ``(normalised_pairs, stage_label)``.
        """
        page_count = pdf_utils.get_page_count(pdf_bytes)
        logger.debug(
            "[STAGE-TRACE] document_id=%s START — size=%dKB pages=%d",
            document_id,
            len(pdf_bytes) // 1024,
            page_count,
        )

        # ── Stage 0: direct call on original bytes ─────────────────────
        try:
            raw = await self._analyze_once(pdf_bytes)
            if raw:
                logger.debug(
                    "[STAGE-TRACE] document_id=%s STAGE-0 SUCCESS — %d KV pairs",
                    document_id,
                    len(raw),
                )
                return normalize_pairs(raw), "STAGE-0"
            logger.debug(
                "[STAGE-TRACE] document_id=%s STAGE-0 EMPTY — entering recovery chain",
                document_id,
            )
        except HttpResponseError as exc:
            if _is_quota_exhausted(exc):
                logger.error(
                    "Azure DI quota exhausted for document_id=%s — upgrade to S0 paid tier",
                    document_id,
                )
                self._gate.record_blocked()
                return [], "QUOTA-403"
            if not _is_oversize_error(exc):
                raise
            logger.warning(
                "[STAGE-TRACE] document_id=%s STAGE-0 OVERSIZE-400 — entering recovery chain",
                document_id,
            )

        # ── Stage 1: page split ────────────────────────────────────────
        logger.debug(
            "[STAGE-TRACE] document_id=%s STAGE-1 ENTER — splitting into chunks of %d",
            document_id,
            self._config.pages_per_chunk,
        )
        split_pairs = await self._analyze_split_chunks(pdf_bytes, document_id)
        if split_pairs:
            logger.debug(
                "[STAGE-TRACE] document_id=%s STAGE-1 SUCCESS — %d KV pairs",
                document_id,
                len(split_pairs),
            )
            return normalize_pairs(split_pairs), "STAGE-1"
        logger.debug(
            "[STAGE-TRACE] document_id=%s STAGE-1 EMPTY — advancing to DPI reduction",
            document_id,
        )

        # ── Stage 2: DPI reduction ────────────────────────────────────
        logger.debug(
            "[STAGE-TRACE] document_id=%s STAGE-2 ENTER — reducing to %d DPI",
            document_id,
            TARGET_DPI,
        )
        reduced = pdf_utils.reduce_dpi(pdf_bytes, TARGET_DPI)

        # ── Stage 3: rotation block ───────────────────────────────────
        logger.debug(
            "[STAGE-TRACE] document_id=%s STAGE-3 ENTER — rotation block",
            document_id,
        )
        rotation_pairs = await self._analyze_rotation_block(reduced, document_id)
        stage = "STAGE-2/3" if rotation_pairs else "STAGE-2/3-EMPTY"
        return normalize_pairs(rotation_pairs), stage

    async def _analyze_rotation_block(
        self,
        pdf_bytes: bytes,
        document_id: str,
    ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        """As-is → 90° → 180° → 270°.  Short-circuits on first non-empty result."""

        # As-is
        try:
            raw = await self._analyze_once(pdf_bytes)
            if raw:
                return raw
        except HttpResponseError as exc:
            logger.warning("STAGE-3 as-is error for document_id=%s: %s", document_id, exc)

        # Rotations
        for degrees in (90, 180, 270):
            try:
                rotated = pdf_utils.rotate_pdf(pdf_bytes, degrees)
                raw = await self._analyze_once(rotated)
                if raw:
                    logger.debug(
                        "[STAGE-TRACE] document_id=%s STAGE-3 %d° SUCCESS",
                        document_id,
                        degrees,
                    )
                    return raw
            except Exception as exc:
                logger.warning(
                    "STAGE-3 %d° failed for document_id=%s: %s",
                    degrees,
                    document_id,
                    exc,
                )

        return []

    async def _analyze_split_chunks(
        self,
        pdf_bytes: bytes,
        document_id: str,
    ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        """Split PDF into page chunks and analyse each in parallel (asyncio.gather)."""
        chunks = pdf_utils.split_by_page_count(pdf_bytes, self._config.pages_per_chunk)

        async def _analyze_chunk(
            idx: int, chunk: bytes
        ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
            try:
                result = await self._analyze_once(chunk)
                logger.debug(
                    "[STAGE-TRACE] document_id=%s STAGE-1 chunk %d/%d — %d pairs",
                    document_id,
                    idx + 1,
                    len(chunks),
                    len(result),
                )
                return result
            except Exception as exc:
                logger.error(
                    "STAGE-1 chunk %d/%d failed for document_id=%s: %s",
                    idx + 1,
                    len(chunks),
                    document_id,
                    exc,
                )
                return []

        chunk_results = await asyncio.gather(
            *[_analyze_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        )
        merged: list[tuple[Optional[str], Optional[str], Optional[float]]] = []
        for chunk_pairs in chunk_results:
            merged.extend(chunk_pairs)
        return merged

    # ------------------------------------------------------------------
    # Single Azure DI call with 429 retry back-off
    # ------------------------------------------------------------------

    async def _analyze_once(
        self,
        pdf_bytes: bytes,
    ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        """
        Single Azure DI call protected by the gate and 429 retry logic.

        Mirrors ``DocumentIntelligenceService.analyzeOnce`` exactly:
          - Acquires one semaphore permit before calling Azure.
          - Exponential back-off: initial → ×2 each retry, capped at max.
          - ±20 % jitter applied after capping to prevent thundering-herd retries.
          - ``Retry-After`` header overrides the computed delay when present.
          - Non-429 errors propagate immediately.
        """
        async with self._gate:
            delay_ms = float(self._config.rate_limit_initial_delay_ms)

            for attempt in range(self._config.rate_limit_max_retries + 1):
                try:
                    return await self._do_analyze_once(pdf_bytes)
                except HttpResponseError as exc:
                    if exc.status_code != 429:
                        raise
                    if attempt == self._config.rate_limit_max_retries:
                        logger.warning(
                            "Azure DI 429 — exhausted %d retries, propagating",
                            self._config.rate_limit_max_retries,
                        )
                        self._gate.record_blocked()
                        raise

                    sleep_ms = _retry_after_ms(exc) or delay_ms
                    jitter = sleep_ms * 0.2 * (random.random() * 2 - 1)
                    actual_ms = max(
                        1.0,
                        min(sleep_ms + jitter, self._config.rate_limit_max_delay_ms),
                    )
                    logger.info(
                        "Azure DI 429 — attempt %d/%d, sleeping %.0f ms",
                        attempt + 1,
                        self._config.rate_limit_max_retries,
                        actual_ms,
                    )
                    await asyncio.sleep(actual_ms / 1_000)
                    delay_ms = min(delay_ms * 2, self._config.rate_limit_max_delay_ms)

            raise RuntimeError("_analyze_once: unreachable")

    async def _do_analyze_once(
        self,
        pdf_bytes: bytes,
    ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        """
        Raw Azure DI call — no retry, no gate.

        Runs in a thread-pool executor so the blocking SDK call does not
        stall the event loop (Python Azure SDK is synchronous).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._call_azure_sync, pdf_bytes)

    def _call_azure_sync(
        self,
        pdf_bytes: bytes,
    ) -> list[tuple[Optional[str], Optional[str], Optional[float]]]:
        """Synchronous Azure DI invocation — called from run_in_executor."""
        import io

        poller = self._client.begin_analyze_document(
            self._config.model_id,
            pdf_bytes,
        )
        result = poller.result()
        if not result.key_value_pairs:
            return []
        return [
            (
                kvp.key.content if kvp.key else None,
                kvp.value.content if kvp.value else None,
                kvp.confidence,
            )
            for kvp in result.key_value_pairs
        ]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_oversize_error(exc: HttpResponseError) -> bool:
    return exc.status_code == 400 and "InvalidContentLength" in str(exc)


def _is_quota_exhausted(exc: HttpResponseError) -> bool:
    return exc.status_code == 403 and "Out of call volume quota" in str(exc)


def _retry_after_ms(exc: HttpResponseError) -> Optional[float]:
    """Parse the ``Retry-After`` header value (seconds) → milliseconds."""
    try:
        header = exc.response.headers.get("Retry-After")  # type: ignore[union-attr]
        if header:
            return float(header.strip()) * 1_000
    except Exception:
        pass
    return None


def _infer_form_type(entries: list[KvEntry]) -> TaxFormType:
    """
    Heuristic: detect the IRS form type from key names in the extracted pairs.
    Returns ``TaxFormType.UNKNOWN`` when no confident signal is found.
    """
    keys_lower = {e.key.lower() for e in entries if e.key}
    if any("1040" in k for k in keys_lower):
        return TaxFormType.FORM_1040
    if any("w-2" in k or "w2" in k or "employer" in k for k in keys_lower):
        return TaxFormType.W2
    if any("schedule c" in k or "profit or loss from business" in k for k in keys_lower):
        return TaxFormType.SCHEDULE_C
    if any("schedule e" in k or "supplemental income" in k for k in keys_lower):
        return TaxFormType.SCHEDULE_E
    if any("schedule k-1" in k or "partner's share" in k for k in keys_lower):
        return TaxFormType.SCHEDULE_K1
    if any("1065" in k for k in keys_lower):
        return TaxFormType.FORM_1065
    if any("1120-s" in k for k in keys_lower):
        return TaxFormType.FORM_1120S
    if any("1120" in k for k in keys_lower):
        return TaxFormType.FORM_1120
    return TaxFormType.UNKNOWN


def _now_ms() -> int:
    return int(time.monotonic() * 1_000)

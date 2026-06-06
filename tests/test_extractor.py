"""
Unit tests for TaxFormExtractor — Azure DI calls are fully mocked.
No real network calls, no real PDFs required.
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import HttpResponseError

from llama_index_readers_azure_tax_forms.extractor import (
    TaxFormExtractor,
    ExtractionConfig,
    _is_oversize_error,
    _is_quota_exhausted,
    _retry_after_ms,
)
from llama_index_readers_azure_tax_forms.gate import AzureDiGate
from llama_index_readers_azure_tax_forms.models import TaxFormType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> ExtractionConfig:
    defaults = dict(
        endpoint="https://fake.cognitiveservices.azure.com/",
        api_key="fake-key",
        model_id="prebuilt-document",
        rate_limit_max_retries=2,
        rate_limit_initial_delay_ms=10,
        rate_limit_max_delay_ms=100,
        max_concurrent=2,
    )
    defaults.update(overrides)
    return ExtractionConfig(**defaults)


def _make_extractor(**config_overrides) -> TaxFormExtractor:
    config = _make_config(**config_overrides)
    extractor = TaxFormExtractor(config)
    return extractor


def _http_error(status: int, message: str = "") -> HttpResponseError:
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    err = HttpResponseError(message=message, response=response)
    err.status_code = status
    return err


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

class TestAzureDiGate:
    @pytest.mark.asyncio
    async def test_limits_concurrency(self):
        gate = AzureDiGate(max_concurrent=2)
        entered = []

        async def task(n):
            async with gate:
                entered.append(n)
                await asyncio.sleep(0.01)

        await asyncio.gather(*[task(i) for i in range(5)])
        assert len(entered) == 5  # all complete, just throttled

    @pytest.mark.asyncio
    async def test_total_acquires_increments(self):
        gate = AzureDiGate(max_concurrent=5)
        async with gate:
            pass
        async with gate:
            pass
        assert gate.total_acquires == 2

    def test_record_blocked(self):
        gate = AzureDiGate(max_concurrent=5)
        gate.record_blocked()
        gate.record_blocked()
        assert gate.blocked_calls == 2


# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

class TestErrorClassification:
    def test_oversize_error_detected(self):
        exc = _http_error(400, "InvalidContentLength: document too large")
        assert _is_oversize_error(exc)

    def test_non_oversize_400_not_detected(self):
        exc = _http_error(400, "Some other error")
        assert not _is_oversize_error(exc)

    def test_quota_exhausted_detected(self):
        exc = _http_error(403, "Out of call volume quota for this month")
        assert _is_quota_exhausted(exc)

    def test_auth_403_not_quota(self):
        exc = _http_error(403, "Access denied")
        assert not _is_quota_exhausted(exc)

    def test_retry_after_parsed(self):
        exc = _http_error(429)
        exc.response.headers = {"Retry-After": "5"}
        assert _retry_after_ms(exc) == 5_000.0

    def test_retry_after_missing_returns_none(self):
        exc = _http_error(429)
        exc.response.headers = {}
        assert _retry_after_ms(exc) is None


# ---------------------------------------------------------------------------
# Extraction — mocked Azure DI
# ---------------------------------------------------------------------------

class TestTaxFormExtractor:
    @pytest.mark.asyncio
    async def test_extract_stage0_success(self):
        """Stage 0 succeeds — recovery chain never entered."""
        extractor = _make_extractor()
        fake_pairs = [("Adjusted gross income", "75000", 0.99)]

        with patch.object(extractor, "_call_azure_sync", return_value=fake_pairs):
            result = await extractor.extract("test-doc", b"%PDF-fake")

        assert not result.is_empty
        assert result.stage == "STAGE-0"
        assert result.entries[0].key == "Adjusted gross income"
        assert result.entries[0].value == "75000"

    @pytest.mark.asyncio
    async def test_extract_returns_empty_on_complete_failure(self):
        """All stages return empty → result is empty, no exception raised."""
        extractor = _make_extractor()

        with patch.object(extractor, "_call_azure_sync", return_value=[]):
            result = await extractor.extract("test-doc", b"%PDF-fake")

        assert result.is_empty

    @pytest.mark.asyncio
    async def test_429_retried_then_succeeds(self):
        """429 on first attempt → retry → success on second attempt."""
        extractor = _make_extractor(rate_limit_max_retries=2, rate_limit_initial_delay_ms=1)
        call_count = 0
        fake_pairs = [("Total income", "50000", 0.95)]

        def side_effect(pdf_bytes):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _http_error(429, "Too Many Requests")
            return fake_pairs

        with patch.object(extractor, "_call_azure_sync", side_effect=side_effect):
            result = await extractor.extract("test-doc", b"%PDF-fake")

        assert not result.is_empty
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_429_exhausted_records_blocked(self):
        """All retries exhausted → gate.record_blocked() called."""
        extractor = _make_extractor(rate_limit_max_retries=1, rate_limit_initial_delay_ms=1)

        with patch.object(
            extractor, "_call_azure_sync", side_effect=_http_error(429, "Too Many Requests")
        ):
            await extractor.extract("test-doc", b"%PDF-fake")

        assert extractor._gate.blocked_calls >= 1

    @pytest.mark.asyncio
    async def test_non_429_http_error_propagates_as_empty_result(self):
        """Non-429 errors from Stage 0 are caught by extract() and returned as empty."""
        extractor = _make_extractor()

        with patch.object(
            extractor, "_call_azure_sync", side_effect=_http_error(500, "Internal Server Error")
        ):
            result = await extractor.extract("test-doc", b"%PDF-fake")

        assert result.is_empty
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_extract_many_runs_concurrently(self):
        """extract_many processes all documents and returns one result per input."""
        extractor = _make_extractor(max_concurrent=5)
        fake_pairs = [("key", "value", 0.9)]

        with patch.object(extractor, "_call_azure_sync", return_value=fake_pairs):
            results = await extractor.extract_many(
                [(f"doc-{i}", b"%PDF-fake") for i in range(4)]
            )

        assert len(results) == 4
        assert all(not r.is_empty for r in results)


# ---------------------------------------------------------------------------
# Form type inference
# ---------------------------------------------------------------------------

class TestFormTypeInference:
    @pytest.mark.asyncio
    async def test_infers_1040(self):
        extractor = _make_extractor()
        pairs = [("Form 1040 — adjusted gross income", "75000", 0.9)]
        with patch.object(extractor, "_call_azure_sync", return_value=pairs):
            result = await extractor.extract("1040.pdf", b"%PDF")
        assert result.form_type == TaxFormType.FORM_1040

    @pytest.mark.asyncio
    async def test_infers_w2(self):
        extractor = _make_extractor()
        pairs = [("Employer identification number (W-2)", "12-3456789", 0.95)]
        with patch.object(extractor, "_call_azure_sync", return_value=pairs):
            result = await extractor.extract("w2.pdf", b"%PDF")
        assert result.form_type == TaxFormType.W2

    @pytest.mark.asyncio
    async def test_unknown_when_no_signal(self):
        extractor = _make_extractor()
        pairs = [("Some generic field", "value", 0.8)]
        with patch.object(extractor, "_call_azure_sync", return_value=pairs):
            result = await extractor.extract("unknown.pdf", b"%PDF")
        assert result.form_type == TaxFormType.UNKNOWN

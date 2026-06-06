"""
AzureTaxFormReader — LlamaIndex BaseReader integration.

This is the primary contribution surface for the LlamaIndex ecosystem.
It wraps the extraction engine behind the standard ``BaseReader`` interface
so any LlamaIndex RAG pipeline can ingest IRS tax forms with a single import.

Usage::

    from llama_index_readers_azure_tax_forms import AzureTaxFormReader

    reader = AzureTaxFormReader(
        endpoint="https://my-resource.cognitiveservices.azure.com/",
        api_key="...",
        max_concurrent=12,
    )

    # Single file
    docs = reader.load_data("path/to/1040.pdf")

    # Multiple files in parallel
    docs = reader.load_data(["1040.pdf", "w2.pdf", "schedule_c.pdf"])

    # From raw bytes (e.g. downloaded from S3 / blob storage)
    docs = reader.load_data_from_bytes([("1040.pdf", pdf_bytes)])

Each returned ``Document`` contains:
  - ``text``:     Pipe-delimited KV pairs  (``"Adjusted gross income | 75000"``)
  - ``metadata``: ``ExtractionResult.as_dict()``
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from llama_index_readers_azure_tax_forms.extractor import TaxFormExtractor, ExtractionConfig
from llama_index_readers_azure_tax_forms.gate import AzureDiGate
from llama_index_readers_azure_tax_forms.models import ExtractionResult
from llama_index_readers_azure_tax_forms import audit

logger = logging.getLogger(__name__)


class AzureTaxFormReader(BaseReader):
    """
    LlamaIndex reader that extracts key-value pairs from IRS tax form PDFs
    using Azure Document Intelligence.

    Supports:
      - Form 1040, W-2, Schedule C/E/K-1
      - Form 1065, 1120, 1120-S
      - Any PDF processable by the ``prebuilt-document`` Azure DI model

    The reader handles all extraction complexity internally:
      - Concurrent extraction with a configurable semaphore gate
      - 4-stage recovery chain (direct → split → DPI-reduce → rotate)
      - Exponential back-off on Azure DI 429 rate limit responses
      - Field normalisation (trailing spaces, known typos, quoted numerics)

    Args:
        endpoint:           Azure Document Intelligence resource endpoint URL.
        api_key:            Azure DI API key.
        model_id:           Azure DI model to use (default: ``prebuilt-document``).
        max_concurrent:     Maximum simultaneous Azure DI calls.  Tune based on
                            your tier: S0 paid tier → 12 is a safe default.
        pages_per_chunk:    Pages per chunk in the splitter recovery stage.
        poll_timeout_seconds: Per-call timeout for the Azure DI poller.
        rate_limit_max_retries:   Maximum 429 retry attempts per call.
        rate_limit_initial_delay_ms: Back-off initial delay (ms).
        rate_limit_max_delay_ms:     Back-off ceiling (ms).
        enable_audit_log:   Write per-document extraction audit rows to
                            ``logs/extraction-audit.log`` (default: True).
        audit_log_dir:      Directory for the audit log file.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_id: str = "prebuilt-document",
        max_concurrent: int = 12,
        pages_per_chunk: int = 10,
        poll_timeout_seconds: int = 120,
        rate_limit_max_retries: int = 5,
        rate_limit_initial_delay_ms: int = 1_000,
        rate_limit_max_delay_ms: int = 32_000,
        enable_audit_log: bool = True,
        audit_log_dir: str = "logs",
    ) -> None:
        config = ExtractionConfig(
            endpoint=endpoint,
            api_key=api_key,
            model_id=model_id,
            pages_per_chunk=pages_per_chunk,
            poll_timeout_seconds=poll_timeout_seconds,
            rate_limit_max_retries=rate_limit_max_retries,
            rate_limit_initial_delay_ms=rate_limit_initial_delay_ms,
            rate_limit_max_delay_ms=rate_limit_max_delay_ms,
            max_concurrent=max_concurrent,
        )
        self._extractor = TaxFormExtractor(config)
        audit.configure_audit_logger(
            log_dir=audit_log_dir,
            enabled=enable_audit_log,
        )

    # ------------------------------------------------------------------
    # BaseReader interface
    # ------------------------------------------------------------------

    def load_data(
        self,
        file: Union[str, Path, list[Union[str, Path]]],
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Load and extract KV pairs from one or more PDF files.

        Args:
            file:       A single file path or a list of file paths.
            extra_info: Optional metadata merged into every returned Document.

        Returns:
            List of :class:`llama_index.core.schema.Document`, one per input file.
        """
        paths: list[Path] = (
            [Path(f) for f in file]
            if isinstance(file, list)
            else [Path(file)]
        )

        pairs: list[tuple[str, bytes]] = []
        for path in paths:
            try:
                pairs.append((str(path), path.read_bytes()))
            except OSError as exc:
                logger.error("Could not read file %s: %s", path, exc)
                pairs.append((str(path), b""))

        return self._run_extraction(pairs, extra_info or {})

    def load_data_from_bytes(
        self,
        documents: list[tuple[str, bytes]],
        extra_info: dict | None = None,
    ) -> list[Document]:
        """
        Load and extract KV pairs from in-memory PDF bytes.

        Useful when documents come from blob storage, S3, a database, or
        any source that provides raw bytes rather than file paths.

        Args:
            documents:  List of ``(document_id, pdf_bytes)`` tuples.
            extra_info: Optional metadata merged into every returned Document.

        Returns:
            List of :class:`llama_index.core.schema.Document`, one per input.
        """
        return self._run_extraction(documents, extra_info or {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_extraction(
        self,
        documents: list[tuple[str, bytes]],
        extra_info: dict,
    ) -> list[Document]:
        """Run async extraction synchronously via a managed event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop (e.g. Jupyter) — use a thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, self._extractor.extract_many(documents)
                )
                results: list[ExtractionResult] = future.result()
        else:
            results = asyncio.run(self._extractor.extract_many(documents))

        audit.write_header()
        llama_docs: list[Document] = []
        for result in results:
            audit.record(result)
            llama_docs.append(_to_document(result, extra_info))
        return llama_docs


# ------------------------------------------------------------------
# Document conversion
# ------------------------------------------------------------------

def _to_document(result: ExtractionResult, extra_info: dict) -> Document:
    """
    Convert an :class:`ExtractionResult` to a LlamaIndex :class:`Document`.

    Text format: one ``key | value`` line per KV pair, suitable for embedding
    and retrieval.  Empty/None values are rendered as ``(blank)``.
    """
    lines = [
        f"{entry.key} | {entry.value if entry.value else '(blank)'}"
        for entry in result.entries
    ]
    text = "\n".join(lines) if lines else "(no key-value pairs extracted)"

    metadata = {**result.as_dict(), **extra_info}
    return Document(text=text, metadata=metadata)

"""
llama-index-readers-azure-tax-forms
====================================
LlamaIndex reader for IRS tax form documents using Azure Document Intelligence.

Supports:
  - Form 1040 (individual income tax)
  - W-2 (wage and tax statement)
  - Schedule C (business profit/loss)
  - Schedule E (supplemental income)
  - Schedule K-1 (partnership / S-corp / trust)
  - Form 1065 / 1120 / 1120-S (business returns)

Features:
  - 4-stage recovery chain: direct → page-split → DPI-reduce → rotation block
  - Exponential back-off with ±20% jitter on Azure DI 429 responses
  - Field normalisation (trailing spaces, known typos, quoted numerics)
  - Per-document extraction audit log (file-bounded, no PII in stdout)

For high-volume concurrent processing across multiple documents and data sources
(S3, blob storage, databases), contact us about the enterprise edition.
"""

from llama_index_readers_azure_tax_forms.reader import AzureTaxFormReader
from llama_index_readers_azure_tax_forms.models import KvEntry, ExtractionResult, TaxFormType

__all__ = [
    "AzureTaxFormReader",
    "KvEntry",
    "ExtractionResult",
    "TaxFormType",
]

"""
Quick smoke test — no real Azure DI credentials needed.
Mocks the Azure DI client so you can verify the full pipeline
(gate, recovery chain, normalisation, LlamaIndex Document output)
runs end to end without a network call.

Run:
    .venv/bin/python test_mock.py
"""
import asyncio
from unittest.mock import patch, MagicMock

from llama_index_readers_azure_tax_forms import AzureTaxFormReader
from llama_index_readers_azure_tax_forms.extractor import TaxFormExtractor

# Fake KV pairs that Azure DI would return for a Form 1040
FAKE_1040_PAIRS = [
    ("Form 1040 — Filing status", "Single", 0.99),
    ("Wages, salaries, tips, etc.", "82000", 0.98),
    ("Adjusted gross income", "75000", 0.97),
    ("Federal income tax withheld", "12500", 0.96),
    ("Total tax", "9800", 0.95),
]


def run():
    reader = AzureTaxFormReader(
        endpoint="https://fake.cognitiveservices.azure.com/",
        api_key="fake-key",
        max_concurrent=2,
        enable_audit_log=False,   # skip file write in this smoke test
    )

    # Patch the actual Azure DI call — return fake pairs
    with patch.object(
        reader._extractor, "_call_azure_sync", return_value=FAKE_1040_PAIRS
    ):
        docs = reader.load_data_from_bytes([
            ("1040_test.pdf", b"%PDF-1.4 fake pdf bytes"),
            ("w2_test.pdf",   b"%PDF-1.4 fake pdf bytes"),
        ])

    print(f"\n✅  {len(docs)} document(s) returned\n")
    for doc in docs:
        print("=" * 60)
        print(f"document_id : {doc.metadata['document_id']}")
        print(f"form_type   : {doc.metadata['form_type']}")
        print(f"kv_count    : {doc.metadata['kv_count']}")
        print(f"stage       : {doc.metadata['stage']}")
        print(f"total_ms    : {doc.metadata['total_ms']} ms")
        print()
        print("--- extracted text (LlamaIndex Document.text) ---")
        print(doc.text)
        print()


if __name__ == "__main__":
    run()

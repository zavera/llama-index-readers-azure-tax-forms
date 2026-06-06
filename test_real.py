"""
Real end-to-end test — requires actual Azure DI credentials.
Point it at any tax form PDF you have locally.

Usage:
    export AZURE_DI_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"
    export AZURE_DI_KEY="your-api-key"
    .venv/bin/python test_real.py path/to/your/1040.pdf
"""
import os
import sys

from llama_index_readers_azure_tax_forms import AzureTaxFormReader

ENDPOINT = os.getenv("AZURE_DI_ENDPOINT", "").strip()
API_KEY  = os.getenv("AZURE_DI_KEY", "").strip()

if not ENDPOINT or not API_KEY:
    print("❌  Set AZURE_DI_ENDPOINT and AZURE_DI_KEY environment variables first.")
    sys.exit(1)

pdf_paths = sys.argv[1:] if len(sys.argv) > 1 else []
if not pdf_paths:
    print("Usage: python test_real.py path/to/form.pdf [more.pdf ...]")
    sys.exit(1)

reader = AzureTaxFormReader(
    endpoint=ENDPOINT,
    api_key=API_KEY,
    max_concurrent=4,
    enable_audit_log=False,  # no file writes in CI
)

print(f"\n📄  Extracting {len(pdf_paths)} document(s)...\n")
docs = reader.load_data(pdf_paths)

all_passed = True
for doc in docs:
    kv_count = doc.metadata["kv_count"]
    stage    = doc.metadata["stage"]
    total_ms = doc.metadata["total_ms"]
    form     = doc.metadata["form_type"]
    doc_id   = doc.metadata["document_id"]

    status = "✅" if kv_count > 0 else "⚠️ "
    if kv_count == 0:
        all_passed = False

    print(f"{status}  {doc_id}")
    print(f"     form={form}  kv_count={kv_count}  stage={stage}  total_ms={total_ms}ms")
    print()

if not all_passed:
    print("❌  One or more documents returned 0 KV pairs.")
    sys.exit(1)

print(f"✅  All {len(docs)} document(s) extracted successfully.")

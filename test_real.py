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
    print("Usage: .venv/bin/python test_real.py path/to/form.pdf [more.pdf ...]")
    sys.exit(1)

reader = AzureTaxFormReader(
    endpoint=ENDPOINT,
    api_key=API_KEY,
    max_concurrent=4,
    enable_audit_log=True,
)

print(f"\n📄  Extracting {len(pdf_paths)} document(s)...\n")
docs = reader.load_data(pdf_paths)

for doc in docs:
    print("=" * 60)
    print(f"document_id : {doc.metadata['document_id']}")
    print(f"form_type   : {doc.metadata['form_type']}")
    print(f"kv_count    : {doc.metadata['kv_count']}")
    print(f"stage       : {doc.metadata['stage']}")
    print(f"di_calls    : {doc.metadata['di_calls']}")
    print(f"az_di_ms    : {doc.metadata['az_di_ms']} ms")
    print(f"total_ms    : {doc.metadata['total_ms']} ms")
    print()
    print("--- first 10 KV pairs ---")
    lines = doc.text.split("\n")
    for line in lines[:10]:
        print(" ", line)
    if len(lines) > 10:
        print(f"  ... and {len(lines) - 10} more")
    print()

print(f"✅  Done — {len(docs)} document(s) extracted.")
print("📋  Audit log written to logs/extraction-audit.log")

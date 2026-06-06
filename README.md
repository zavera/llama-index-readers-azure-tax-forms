# llama-index-readers-azure-tax-forms

A [LlamaIndex](https://github.com/run-llama/llama_index) reader that extracts structured key-value pairs from IRS tax form PDFs using **Azure Document Intelligence**.

## Supported Forms

| Form | Description |
|------|-------------|
| Form 1040 | Individual income tax return |
| W-2 | Wage and tax statement |
| Schedule C | Profit or loss from business |
| Schedule E | Supplemental income and loss |
| Schedule K-1 | Partner's / shareholder's share of income |
| Form 1065 | U.S. return of partnership income |
| Form 1120 / 1120-S | Corporate income tax return |

## Installation

```bash
pip install llama-index-readers-azure-tax-forms
```

## Quick Start

```python
from llama_index_readers_azure_tax_forms import AzureTaxFormReader

reader = AzureTaxFormReader(
    endpoint="https://my-resource.cognitiveservices.azure.com/",
    api_key="YOUR_AZURE_DI_KEY",
    max_concurrent=12,   # tune to your Azure DI tier
)

# Single file
docs = reader.load_data("path/to/1040.pdf")

# Multiple files — processed concurrently
docs = reader.load_data(["1040.pdf", "w2.pdf", "schedule_c.pdf"])

# From raw bytes (S3, blob storage, database, etc.)
docs = reader.load_data_from_bytes([
    ("1040.pdf", open("1040.pdf", "rb").read()),
    ("w2.pdf",   open("w2.pdf",   "rb").read()),
])

# Use in a LlamaIndex RAG pipeline
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("What is the adjusted gross income?")
```

## Document Output

Each returned `Document` contains:

- **`text`** — one `key | value` line per extracted KV pair:
  ```
  Adjusted gross income | 75,000
  Wages, salaries, tips, etc. | 80,000
  Federal income tax withheld | 12,500
  ```

- **`metadata`** — extraction details:
  ```json
  {
    "document_id": "1040.pdf",
    "form_type": "1040",
    "kv_count": 47,
    "stage": "STAGE-0",
    "di_calls": 1,
    "az_di_ms": 1823,
    "total_ms": 1902
  }
  ```

## Key Features

### Concurrency Gate
A shared `asyncio.Semaphore` limits concurrent Azure DI calls so parallel extractions never trigger 429 rate-limit responses. Tune `max_concurrent` to your tier (S0 paid tier → 12 is safe empirically).

### 4-Stage Recovery Chain
Every document goes through a recovery chain before accepting an empty result:

| Stage | What it does |
|-------|-------------|
| **Stage 0** | Direct Azure DI call on original bytes |
| **Stage 1** | Split PDF into page chunks, analyse in parallel |
| **Stage 2** | Re-render pages at 300 DPI (rasterise) |
| **Stage 3** | Rotation block: as-is → 90° → 180° → 270° |

### 429 Retry Back-off
Exponential back-off with ±20% jitter on Azure DI rate limit responses. Honoring `Retry-After` header when present.

### Field Normalisation
Corrects known Azure DI output quirks:
- Trailing spaces in key names
- Known field-name typos
- `>` separator characters
- Quoted numeric values

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent` | 12 | Max simultaneous Azure DI calls |
| `pages_per_chunk` | 10 | Pages per chunk in Stage 1 split |
| `poll_timeout_seconds` | 120 | Per-call Azure DI timeout |
| `rate_limit_max_retries` | 5 | Max 429 retry attempts |
| `rate_limit_initial_delay_ms` | 1000 | Initial back-off delay (ms) |
| `rate_limit_max_delay_ms` | 32000 | Maximum back-off delay (ms) |
| `enable_audit_log` | True | Write extraction audit to file |

## Azure Setup

1. Create an **Azure Document Intelligence** resource (S0 paid tier recommended)
2. Copy the **endpoint URL** and **API key** from the Azure portal
3. The `prebuilt-document` model is used by default — no custom training required

## Development

```bash
git clone https://github.com/YOUR_GITHUB/llama-index-readers-azure-tax-forms
cd llama-index-readers-azure-tax-forms
poetry install
pytest
```

## License

MIT

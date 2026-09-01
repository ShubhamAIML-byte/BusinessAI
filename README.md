# Retail Intelligence Assistant

A modular Streamlit app that classifies customer emails as **product inquiries**
or **order requests**, matches products against a catalog, checks stock, and
drafts guarded customer responses (no supplier names, contacts, emails,
phone numbers, or margins ever leak into a reply).

## Files

| File | Purpose |
|---|---|
| `backend.py` | All business logic: classification, product search, order extraction/processing, response generation, and the leak-detection guardrail. UI-agnostic — reusable from a script, notebook, or API. |
| `app.py` | Streamlit frontend: catalog/email upload, batch processing view with exports, and a live chat tab. |
| `.env.example` | Template for API credentials. Copy to `.env` and fill in. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set OPENAI_API_KEY (and OPENAI_BASE_URL / OPENAI_MODEL if needed)
```

If no `OPENAI_API_KEY` is set, the app still runs — every LLM call has a
deterministic rule-based fallback (keyword classification, substring/fuzzy
product matching, and templated responses), matching the original notebook's
behavior.

## Run

```bash
streamlit run app.py
```

## Expected file formats

**Product catalog** (CSV or XLSX) — required columns:
`product_id, name, category, country, price, currency, stock, description`
Optional: `internal_notes` (used only server-side to build the guardrail's
restricted-name list; never shown to users or the LLM).

**Emails** (CSV or XLSX, for batch processing) — required columns:
`email_id, subject, message`

## Using the backend outside Streamlit

```python
import pandas as pd
from backend import load_backend_from_env, export_outputs

products_df = pd.read_csv("products.csv")
backend = load_backend_from_env(products_df)

# Single query (same as the CLI chat in the original notebook)
result = backend.process_customer_query("Do you have SKU-102 in stock?")
print(result["Generated Response"])

# Batch processing
emails_df = pd.read_csv("emails.csv")
dfs = backend.process_batch(emails_df)
export_outputs(dfs, output_dir="generated_outputs")
```

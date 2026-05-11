# Researcher Finder

A search tool for finding J-PAL-affiliated researchers by topic, name,
institution, country, sector, language, or office. The frontend is a
static page; the Python pipeline builds the index it reads.

## What's in the box

- `frontend/` — static UI (HTML/CSS/JS) that loads `profiles_index.json`
  and runs semantic + keyword search in the browser.
- `serve.py` — local dev server. Serves the frontend and proxies
  embedding + GPT synthesis calls to OpenAI so the browser never needs
  an API key.
- `src/` — Python pipeline: ingestion (Salesforce, Excel),
  scraping (web bios, OpenAlex), profile assembly, embedding, search.
- `data/` — input spreadsheets (Salesforce exports, sector mapping).
- `output/` — generated artifacts (profiles, index, embeddings, scrape
  reports). Not committed; rebuilt by the pipeline.
- `deploy/cloudflare-embed-worker.js` — optional worker so the static
  site can call OpenAI from a hosted deployment without leaking keys.
- `Makefile` — entry points for the build / embed / search loop.

## Quick start (local)

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # or put it in .env
python3 serve.py                    # http://localhost:8000
```

That serves the existing index in `output/`. To rebuild the index from
source data:

```bash
make fetch-sf      # Salesforce → output/researchers_clean.csv
make build         # profiles → document chunks
make embed         # chunks → embeddings (resumes from checkpoint)
make search Q="cash transfers in Kenya"   # CLI search
```

`make help` lists every target.

## Fetching data from online sources

The pipeline pulls from four external sources. Each one caches its raw
responses under `cache/` so re-runs are cheap and resumable.

### 1. Salesforce researcher reports → `output/researchers_clean.csv`

```bash
make fetch-sf
# = python3 -m src.ingest.fetch_salesforce_researchers
```

Authenticates via OAuth (see `src/ingest/salesforce_auth.py`), pulls two
Tabular reports — affiliates and invited — by their report IDs,
concatenates them, applies the column mapping in
`data/salesforce_column_map.json` (template at
`data/salesforce_column_map.example.json`), runs the same cleaning as
`src.ingest.clean_researchers`, and writes
`output/researchers_clean.csv`. This CSV is the seed every later step
reads from.

Required env vars (typically in `.env`):

- `SALESFORCE_REPORT_AFFILIATES_ID`, `SALESFORCE_REPORT_INVITED_ID`
- Salesforce OAuth creds — see `salesforce_auth.py` for the exact
  fields it expects.
- Optional: `SALESFORCE_API_VERSION` (default `59.0`),
  `SALESFORCE_COLUMN_MAP` to point at a non-default mapping file.

The initiative / office / end-date Excel sheet is still ingested
locally — `python3 -m src.ingest.extra_sheet` reads
`data/Report-2026-02-19-11-34-43.xlsx`.

### 2. Researcher homepages → `output/web/`

```bash
python3 -m src.scrape.scrape_homepages
```

For every researcher in `researchers_clean.csv` with a website URL, it
GETs the page (User-Agent identifying as `ResearcherScraper/1.0`,
20-second timeout, 0.6 s between requests to stay polite), caches the
raw HTML under `cache/html/<hash>.html`, runs `html_to_text` to extract
readable content, and writes a JSON record per researcher to
`output/web/`. Failures are logged so `src.scrape.make_retry_list` and
`src.scrape.retry_failed` can have another go later.

### 3. OpenAlex enrichment → `output/openalex/`

```bash
python3 -m src.ingest.openalex_enrich
```

Resolves each researcher against the OpenAlex API
(`https://api.openalex.org`), saves the author record under
`output/openalex/authors/` and recent works under
`output/openalex/works/`, and produces a summary CSV. Responses are
cached in `cache/openalex/`, requests are conservatively paced to
avoid 429s, and matching uses email, ORCID, and name+affiliation
heuristics. No API key required, but setting a contact email in the
User-Agent gets you into OpenAlex's "polite pool."

### 4. CV downloads + text extraction → `output/cv/`

```bash
python3 -m src.cv.download_and_extract_cvs
```

Downloads CV URLs from the cleaned CSV, sniffs each file's type
(PDF / DOCX / HTML), extracts text via `src.cv.extract_cv_text`, and
writes one JSON record per researcher to `output/cv/`. Raw files are
cached in `cache/cv/`. `src.cv.summarize_cv_run` produces a run report
with success/failure counts.

### Putting it together

Typical refresh from scratch:

```bash
make fetch-sf                                # Salesforce
python3 -m src.ingest.extra_sheet            # local Excel (initiatives/offices)
python3 -m src.scrape.scrape_homepages       # homepages
python3 -m src.ingest.openalex_enrich        # OpenAlex
python3 -m src.cv.download_and_extract_cvs   # CVs
python3 -m src.profile.build_profiles        # merge all of the above
make build && make embed                     # index + embeddings
```

Each step only re-fetches what's missing from its cache, so partial
re-runs are safe.

## Configuration

- `OPENAI_API_KEY` — required for embedding and synthesis.
- Salesforce env vars (used by `make fetch-sf`) live in `.env`; see
  `data/salesforce_column_map.example.json` for the column mapping
  template.
- `frontend/config.json` — optional. Set `embedApiUrl` to your
  Cloudflare worker URL for a CORS-safe production deployment;
  otherwise users supply an OpenAI key under the Settings panel.

## How search works

Each researcher is embedded once (chunks + a narrative summary). At
query time the frontend embeds the query, computes cosine similarity
against every researcher, and adds a small keyword boost when query
terms appear in high-signal fields (country, sectors, institution,
publications keyword index). Filters (country, office, language,
sector, type, name, university) are applied first to narrow the pool.

Short queries that look like an institution name (e.g. "MIT",
"Notre Dame") or a person's name route around the embedding step and
match directly against the affiliation / name fields.

Results are paginated and can be exported as CSV, Excel, or PDF.

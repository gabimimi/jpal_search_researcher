# Researcher Search Pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Prerequisites:
#   pip install -r requirements.txt
#   export OPENAI_API_KEY=sk-...
#
# Refresh researchers from Salesforce (two Tabular reports → output/researchers_clean.csv):
#   Set Salesforce env vars, then:  make fetch-sf
#   Initiative sheet still from local Excel:  python3 -m src.ingest.extra_sheet
#
# Run the full pipeline:
#   make all
#
# Or step by step:
#   make build     → build document chunks from profiles
#   make embed     → embed chunks with OpenAI (resumes from checkpoint)
#   make search Q="K-12 education"

PYTHON := python3
INDEX_DIR := output/index

.PHONY: all build embed search clean-index serve-frontend fetch-sf help

all: build embed
	@echo ""
	@echo "Pipeline complete. Run:  make search Q=\"your topic\""

## Step A: Build document chunks from profiles
build:
	$(PYTHON) -m src.index.build_documents
	@echo ""
	@cat $(INDEX_DIR)/documents_summary.txt

## Step B: Embed documents (resumes automatically)
embed:
	$(PYTHON) -m src.index.embed_documents

## Step B (custom model):
embed-large:
	$(PYTHON) -m src.index.embed_documents --model text-embedding-3-large

## Step C: Search (usage: make search Q="K-12 education")
Q ?= education
N ?= 10
search:
	$(PYTHON) -m src.index.search --query "$(Q)" --top $(N)

## Search with JSON output
search-json:
	$(PYTHON) -m src.index.search --query "$(Q)" --top $(N) --json

## Search with initiative filter
search-filtered:
	$(PYTHON) -m src.index.search --query "$(Q)" --top $(N) --filter-initiative "$(INIT)"

## Remove generated index files (keeps profiles intact)
clean-index:
	rm -f $(INDEX_DIR)/documents.jsonl \
	      $(INDEX_DIR)/documents_with_embeddings.jsonl \
	      $(INDEX_DIR)/embed_checkpoint.txt \
	      $(INDEX_DIR)/embed_meta.json \
	      $(INDEX_DIR)/documents_summary.txt

## Pull affiliates + invited reports from Salesforce → output/researchers_clean.csv
fetch-sf:
	$(PYTHON) -m src.ingest.fetch_salesforce_researchers

## Preview static UI (http://127.0.0.1:8080/frontend/)
serve-frontend:
	@echo "Open http://127.0.0.1:8080/frontend/"
	$(PYTHON) -m http.server 8080 --bind 127.0.0.1

help:
	@echo "Researcher Search Pipeline"
	@echo ""
	@echo "  make fetch-sf                     Salesforce → researchers_clean.csv (needs .env)"
	@echo "  make serve-frontend               Local HTTP server (visit /frontend/)"
	@echo "  make build                        Build chunks from profiles"
	@echo "  make embed                        Embed chunks (OpenAI, resumes)"
	@echo "  make search Q=\"K-12 education\"    Search researchers"
	@echo "  make search-json Q=\"vaccination\"  Search, output raw JSON"
	@echo "  make all                          build + embed"
	@echo "  make clean-index                  Remove index files"
	@echo ""
	@echo "Environment:"
	@echo "  OPENAI_API_KEY=sk-...  (required for embed + search)"

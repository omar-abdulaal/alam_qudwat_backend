# RAG pipeline

Turns scraped source datasets (currently `data/raw/rashidun/` and
`data/raw/companions_tier1/`) into a single, persistent, citable,
semantically searchable knowledge base backed by Postgres + pgvector, all
characters/datasets living in the same `documents`/`chunks` tables and
searchable together.

## Architecture

```
data/raw/rashidun/*_pages.jsonl  ─┐
data/raw/companions_tier1/pages/*.json ─┤
        (any dataset rag/ingestion/loader.py recognizes)
        │
        ▼
  rag/ingestion/ingest.py   (standalone CLI, run manually/via cron — never
        │                    imported by the runtime backend)
        │  loader (format auto-detected) -> cleaning -> chunker -> hashing -> embeddings
        ▼
  Postgres (documents, chunks + pgvector "embedding" column)
        │
        ▼
  rag/retrieval/retriever.py   (semantic search + metadata filters,
        │                        used by the runtime backend / API)
        ▼
  rag/generation/prompt.py     (builds a strictly-grounded LLM prompt +
                                 citation list from retrieved chunks)
```

The vector data lives entirely in Postgres. The backend never loads
embeddings into process memory at startup — every query is a normal SQL
`SELECT ... ORDER BY embedding <=> :query LIMIT k`.

## Data model

- `documents` — one row per scraped page, unique on `(book_id, page_id)`.
  Stores the untouched `raw_text` plus a `content_hash` used to detect
  whether a page actually changed on re-ingestion.
- `chunks` — one row per retrievable chunk, with its own `content_hash`,
  its `embedding` (`vector(1536)`), and denormalized source metadata
  (`character`, `caliph_name`, `book_title`, `author`, `era`, `page_id`,
  `printed_page`, `printed_volume`, `dataset_id`, `source_url`) so every
  retrieval hit is self-describing and directly citable — no joins needed.

Despite the column names (`caliph_id`/`caliph_name`/`character`), these
are dataset-agnostic "who this text is about" fields — the same columns
hold Rashidun caliph slugs (`abu_bakr`) and companions_tier1's hash-based
`character_id`s (`siyar_4e8409ebdae1aec0`) alike. Not renamed on the live
schema for cosmetics; see `rag/ingestion/loader.py`'s `SourcePage` for the
dataset-agnostic names used everywhere upstream of the DB models.

`printed_volume` and `dataset_id` are nullable — only datasets that
provide them (companions_tier1) populate them; Rashidun rows have both
`NULL`. `documents.source_content_hash` (nullable, Document-only)
preserves a source-provided content hash verbatim when the dataset
supplies one, purely for provenance — ingestion's own idempotency always
keys off `content_hash` (sha256 of `raw_text`, computed by this project
itself for every dataset, so it behaves identically whether or not the
source also happens to hash its own content).

`era` is derived from `rag/config.py`'s `Settings.era_for_page(character_id,
collection)`: an exact-character-id match against `CHARACTER_ERA_MAP`
first (Rashidun), else a `collection`-keyed fallback in
`COLLECTION_ERA_MAP` (e.g. Companions' `collection="الصحابة"` ->
`era="الصحابة"` — echoing the source's own group label rather than
inventing a specific historical period per person), else a generic
"unspecified" default. Adding a new character/book/era later is just:
point ingestion at a new dataset directory (see "Adding a new dataset"
below), add an entry to `CHARACTER_ERA_MAP`/`COLLECTION_ERA_MAP` if it
needs one, and re-run ingestion — no schema change required for a dataset
that fits the existing field set.

## Setup

1. Install dependencies (already done in this repo's `.venv`):
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and, if your
   Postgres isn't the docker-compose default, `DATABASE_URL`.
3. Provide a Postgres instance with the pgvector extension enabled:
   - **Docker**: `docker compose up -d db` (starts `pgvector/pgvector:pg16`).
   - **No Docker** (e.g. a native local Postgres install, as used in this
     repo's dev setup): create the database yourself and run
     `CREATE EXTENSION vector;` once inside it, then point `DATABASE_URL`
     at it, e.g.
     `postgresql+psycopg://postgres:PASSWORD@localhost:5432/alam_qudwat`.
   - A managed instance such as AWS RDS for PostgreSQL, Neon, or Supabase
     works the same way.
4. Apply migrations:
   ```bash
   alembic upgrade head
   ```

## Running ingestion

```bash
python -m rag.ingestion.ingest --input data/raw/rashidun -v
python -m rag.ingestion.ingest --input data/raw/companions_tier1 -v
```

`--input` accepts any dataset directory in either supported on-disk
shape — `rag/ingestion/loader.py` auto-detects which one it's looking at
(see "Adding a new dataset" below), so the CLI itself never changes.

For Rashidun: reads the four per-caliph files (`abu_bakr_pages.jsonl`,
`umar_pages.jsonl`, `uthman_pages.jsonl`, `ali_pages.jsonl`) — the
combined `rashidun_pages.jsonl` is intentionally skipped because it's a
stale artifact containing only one caliph's records; the per-caliph files
are the source of truth.

For companions_tier1: reads every `pages/*.json` file. 3 of the 1644
pages have no `character_id`/`character_name` in the source (they're
section-header/transition pages spanning multiple people, e.g. "السابقون
الأولون", not one person's biography) — these are skipped, logged, and
counted in `pages_skipped`, never force-fit into a required character
field or guessed at.

For each page: clean structural artifacts left by the scraper's HTML
extraction (bracket markers like `[[`, `]`, `[-`) — the actual historical
text is never altered — then split into ~400-token overlapping chunks
snapped to sentence/paragraph boundaries, then upsert into Postgres.
(companions_tier1's text has none of these bracket artifacts — cleaning
is a harmless no-op on it, not a dataset-specific code path.)

**Idempotency**: every document and chunk carries a sha256 content hash.
Re-running ingestion on unchanged files does a handful of cheap `SELECT`s
and exits — no duplicate rows, no embedding API calls. If a specific
page's text changes (e.g. the scraper is re-run with `--refresh` and pulls
a correction), only that page's chunks are re-embedded; everything else
is untouched. This holds per-dataset and across datasets — re-ingesting
Rashidun never touches companions_tier1 rows or vice versa (different
`(book_id, page_id)` keys). Try it:

```bash
python -m rag.ingestion.ingest --input data/raw/rashidun -v   # first run: embeds everything
python -m rag.ingestion.ingest --input data/raw/rashidun -v   # second run: "unchanged=276", 0 embedding calls
```

## Automatic ingestion on backend startup

`app/main.py` runs `rag.ingestion.ingest.ingest_missing_characters()` in a
background thread every time the backend starts (see
`app/services/rag_sync.py`) — it scans every dataset directory under
`RAG_DATA_DIR` (default `data/raw`) and ingests any character that has
**no rows at all** in `documents` yet. This is deliberately coarser than
the manual CLI above:

- **No content-hash diffing.** A character is either "in the DB" (skipped
  entirely — no read, no hash compare, no embedding call) or "not in the
  DB" (ingested in full). It will never detect that a character's source
  text changed and re-embed it — that's what `ingest --input <dir>`
  (above) is for, if you want to force a refresh of otherwise-unchanged
  content.
- **To force a character to be re-ingested** (e.g. you corrected its
  source text, or want to pick up a scraper fix), delete its rows first:
  ```bash
  python -m rag.ingestion.delete_character <character_id>
  ```
  This deletes its `documents` rows (cascading to `chunks` via the
  existing FK) after showing a preview and asking for confirmation
  (`--yes` to skip the prompt). The next startup sync — or a manual call
  to `ingest_missing_characters()` — will then see it as missing and
  ingest it fresh from the source files.
- **Non-blocking.** It runs on its own OS thread, not inside the FastAPI
  event loop — `GET /health` and every other route respond immediately
  on startup regardless of how long the sync takes or whether it's still
  running.
- **Never crashes the API.** Any failure (missing `OPENAI_API_KEY`, DB
  unreachable, a bad character) is logged and swallowed; the backend
  keeps serving everything else either way.
- Set `AUTO_INGEST_ON_STARTUP=false` to disable it (e.g. a scaled-out
  read replica instance, or local dev where you don't want an automatic
  OpenAI bill on every reload).
- **Known trade-off**: a character that only partially ingested (crashed
  mid-way) still counts as "present" and won't be retried automatically
  — matches the delete-to-update model above rather than silently trying
  to detect and fill gaps. Also, multiple backend instances started at
  the same time would each independently attempt any missing character;
  the existing `(book_id, page_id)` / `(document_id, chunk_index)` UNIQUE
  constraints prevent actual duplicate rows, but there's no distributed
  lock coordinating them — fine for a single-instance deployment, a
  known follow-up for a scaled-out one.

## Adding a new dataset

Two on-disk shapes are currently recognized by `load_source_pages()` in
`rag/ingestion/loader.py`, auto-detected from the `--input` directory:

1. **Flat JSONL**: `{name}_pages.jsonl` files directly in the directory
   (Rashidun's shape).
2. **Per-page JSON**: a `pages/*.json` subdirectory, one file per page,
   filename == `page_id` (companions_tier1's shape).

Both are normalized into the same `SourcePage` dataclass before anything
else in the pipeline sees them — `rag/ingestion/cleaning.py`,
`chunker.py`, `hashing.py`, and `ingest.py`'s diff/upsert/embed logic are
all dataset-agnostic and need no changes for a new dataset that fits
either shape. To add a dataset in a genuinely new shape, add one more
`_load_*_pages()` function to `loader.py` and one more branch in
`load_source_pages()` — nothing downstream needs to know it exists.

If a new dataset's source JSON has fields the current `SourcePage`/
`documents`/`chunks` schema doesn't carry, prefer a new nullable column
(additive Alembic migration) over a dataset-specific side table or a
generic JSONB blob — see `printed_volume`/`dataset_id`/
`source_content_hash` (added for companions_tier1) for the pattern.
Never invent a classification (era, category, character identity) the
source data doesn't actually assert — see `COLLECTION_ERA_MAP` above and
the companions_tier1 null-character-skip behavior for how this project
handles that.

## Testing retrieval

Programmatically:

```python
from rag.db.session import session_scope
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.retrieval.retriever import retrieve

embedder = OpenAIEmbeddingProvider()
with session_scope() as session:
    hits = retrieve(session, "كيف تولى أبو بكر الخلافة؟", embedder, top_k=5, character="abu_bakr")
    for h in hits:
        print(h.score, h.citation())
        print(h.text[:200])
```

`character` filters to a caliph_id (`abu_bakr`, `umar`, `uthman`, `ali`)
or, since companions_tier1 was ingested, any companion's hash-based
`character_id` (e.g. `siyar_4e8409ebdae1aec0` for أبو عبيدة بن الجراح —
look one up via `SELECT DISTINCT character, caliph_name FROM chunks
WHERE character LIKE 'siyar_%'`). `era` and `book_title` filters are also
supported and compose with the semantic search (all applied as SQL
`WHERE` clauses before the ANN sort). Omitting `character` searches
across every ingested dataset at once.

## Tests

```bash
pytest tests/ -v
```

- `test_cleaning.py`, `test_hashing.py`, `test_chunker.py` — pure unit
  tests, no DB required.
- `test_ingest_idempotency.py`, `test_retriever.py` — integration tests
  against a real, already-migrated (`alembic upgrade head`) Postgres+
  pgvector instance. They read `TEST_DATABASE_URL` (falling back to
  `DATABASE_URL`) and **skip automatically** if that database isn't
  reachable or hasn't been migrated yet, so `pytest` is always safe to
  run. They use a deterministic fake embedding provider
  (`tests/fake_embedder.py`) so no OpenAI API calls or costs are involved.
- **Isolation — real data is never at risk**: `TEST_DATABASE_URL` commonly
  falls back to the same `DATABASE_URL` used for real ingested data, so
  every test runs inside a single outer database transaction
  (`tests/conftest.py`, SQLAlchemy `join_transaction_mode="create_savepoint"`)
  that is unconditionally rolled back at teardown. Even though
  `run_ingestion()` and the tests call `session.commit()`, that only
  commits a SAVEPOINT nested inside the outer transaction — the final
  `rollback()` discards it regardless. Nothing a test does is ever
  persisted, no matter what the test/application code inside does.
  Fixture data also uses a `book_id` (`999999`) that can never collide
  with real Shamela book IDs, so assertions stay exact even though real
  committed rows are visible (read-only) inside the test transaction.
- `pytest.ini` sets `--basetemp=.pytest_tmp` so pytest's temp-file fixture
  doesn't depend on the OS temp directory being writable.

## Grounded generation (prep layer)

`rag/generation/prompt.py` builds a system+user prompt pair from a list of
`RetrievedChunk`s:

```python
from rag.generation.prompt import build_prompt, citation_list

prompt = build_prompt(question, hits)
# prompt["system"], prompt["user"] -> pass to your LLM's chat/completions call
print("\n".join(citation_list(hits)))
```

The system prompt instructs the model, in Arabic, to answer only from the
attached sources, to explicitly say when the sources don't cover the
question, to cite source numbers per claim, and to never alter quoted
historical text. No LLM call is made by this module — it's the hook point
for whichever chat API is wired in when a backend/API layer is built.

## AWS deployment notes

- Postgres + pgvector: AWS RDS for PostgreSQL (16+) supports the
  `vector` extension — provision an RDS instance, run
  `CREATE EXTENSION vector;` once, then `alembic upgrade head` from a
  deploy step.
- Ingestion: run as a scheduled ECS task / Lambda (long-running, so ECS
  Fargate task or a batch job is a better fit than Lambda) or a manual
  CI step — it is explicitly decoupled from the backend process.
- Secrets (`OPENAI_API_KEY`, `DATABASE_URL`) belong in AWS Secrets Manager
  / SSM Parameter Store, injected as environment variables — never in
  source control (`.env` is for local dev only).

## Known limitations

- The `EmbeddingProvider` default is OpenAI's `text-embedding-3-small`
  (1536 dims); changing `EMBEDDING_MODEL`/`EMBEDDING_DIM` requires a new
  Alembic migration to alter the `vector` column width and re-ingesting
  (old embeddings are a different vector space and aren't comparable).
- Chunking is token-length-bounded, not linguistically parsed (no full
  Arabic morphological/sentence-boundary model) — it uses sentence-final
  punctuation and paragraph breaks as split points, which works well for
  this scraped source but isn't a general-purpose Arabic NLP sentence
  splitter.
- Verified end-to-end against a real local Postgres 18 + pgvector
  instance: all 18 tests pass (`pytest tests/`), migrations apply
  cleanly, real ingestion embedded all 276 pages / 713 chunks via the
  OpenAI API, a second ingestion run confirmed idempotency (0 embedding
  calls, all pages reported "unchanged"), and semantic retrieval with
  `character` filtering returns correctly on-topic, citable results.

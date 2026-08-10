# Medical GenAI RAG Platform

A production-oriented, fully local Retrieval-Augmented Generation platform for
medical/workflow knowledge, designed to later be consumed as a tool by a
Medical Agentic AI system.

**Stack:** Qdrant (vector DB) · Ollama (local LLM) · Sentence Transformers
(local embeddings) · FastAPI · Python 3.11+
**No cloud LLM APIs. No OpenAI. No Pinecone. Everything runs on your machine.**

---

## Build Plan (14 phases)

This project is built incrementally. Each phase is explained, coded, and
verified before moving to the next.

| Phase | Status | Description |
|---|---|---|
| **1** | ✅ **Done (this doc)** | Environment setup |
| 2 | ⏳ Next | Qdrant setup |
| 3 | ⏳ | Ollama setup |
| 4 | ⏳ | Embedding service |
| 5 | ⏳ | Document ingestion |
| 6 | ⏳ | Chunking + metadata |
| 7 | ⏳ | Qdrant indexing |
| 8 | ⏳ | Basic retrieval |
| 9 | ⏳ | RAG pipeline |
| 10 | ⏳ | Grounding + guardrails |
| 11 | ⏳ | FastAPI |
| 12 | ⏳ | Streamlit UI |
| 13 | ⏳ | Evaluation |
| 14 | ⏳ | Future Agent integration design |

---

## Phase 1 — Environment Setup

### What we built

- Full project skeleton (`app/`, `data/`, `tests/`, `scripts/`) matching the
  target architecture, with every package pre-created as an importable Python
  module (`__init__.py` in place).
- `requirements.txt` — pinned, 100% local/open-source dependencies. No
  `openai`, no cloud SDKs.
- `.env.example` — every configurable value in the system (Qdrant, Ollama,
  embedding model, chunking, retrieval, API) in one place.
- `app/config/settings.py` — a single `pydantic-settings` `Settings` class
  that every later module will import from (`get_settings()`), so nothing
  downstream hard-codes a host, port, or model name.
- `docker-compose.yml` — Qdrant only, for now (see networking notes below).

### Why this order

Every phase after this one — the embedding service, the Qdrant client, the
Ollama service, the FastAPI app, the Streamlit UI — needs a single, reliable
source of configuration. Building and *validating* that first means later
phases are just "import settings and use it," with no scattered
`os.environ.get(...)` calls and no risk of the indexing pipeline and the
query pipeline silently using different embedding models (a common way RAG
systems silently break).

### Architecture (this phase)

```
.env.example  ──copy──>  .env
                            │
                            ▼
                 app/config/settings.py
                 (pydantic-settings, cached)
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
        Qdrant config  Ollama config  Embedding/
        (Phase 2)      (Phase 3)      Chunking config
                                       (Phase 4-6)
```

### Full project structure

Every file from the target architecture exists now, so the shape of the
whole system is visible from day one. Files not yet implemented are real,
importable Python modules containing only a docstring, a `# TODO (Phase N)`
marker, and — where a class is already named in the spec — a skeleton class
whose methods raise `NotImplementedError` until their phase lands. This
means `pytest --collect-only` and static analysis / IDE navigation work
across the whole codebase today, even though most logic doesn't exist yet.

```
medical-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                        ⏳ Phase 11   FastAPI entrypoint
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py                  ⏳ Phase 11   GET /health, /qdrant/health, /ollama/health
│   │   ├── documents.py               ⏳ Phase 11   POST /documents/ingest, /index, GET /documents/{id}
│   │   └── rag.py                     ⏳ Phase 11   POST /rag/search, /rag/query
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py                 ⏳ Phase 5    PDF/DOCX/TXT/MD/JSON loaders
│   │   ├── parser.py                  ⏳ Phase 5    cleaning / normalization
│   │   ├── chunker.py                 ⏳ Phase 6    fixed-size chunking (+ future semantic/parent-child)
│   │   └── metadata.py                ⏳ Phase 6    metadata payload extraction
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py       ⏳ Phase 4    EmbeddingService (Sentence Transformers)
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── qdrant_store.py            ⏳ Phase 2/7  QdrantStore (collection, upsert, search)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py               ⏳ Phase 8    semantic + metadata-filtered, screen-aware retrieval
│   │   └── reranker.py                ⏳ Phase 10   cross-encoder reranking
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_service.py          ⏳ Phase 3    OllamaLLMService (generate, generate_structured, health_check)
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py            ⏳ Phase 9    end-to-end orchestration
│   │   ├── context_builder.py         ⏳ Phase 9    assembles {retrieved_context}
│   │   └── grounding.py               ⏳ Phase 10   grounded/confidence/conflict detection
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── safety.py                  ⏳ Phase 10   response-level safety checks
│   │   └── prompt_injection.py        ⏳ Phase 10   detect injected instructions in retrieved chunks
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py                ⏳ Phase 8/11 pydantic request schemas
│   │   └── responses.py               ⏳ Phase 9/11 pydantic response schemas
│   │
│   └── config/
│       ├── __init__.py
│       └── settings.py                ✅ Phase 1    centralized Settings (done)
│
├── data/
│   ├── raw/                           input documents (git-ignored, .gitkeep only)
│   ├── processed/                     cleaned/parsed output (git-ignored, .gitkeep only)
│   └── sample_documents/              ⏳ Phase 5/21 synthetic demo corpus lands here
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py                 ✅ Phase 1    settings load + singleton cache (real, passing)
│   ├── test_ingestion.py              ⏳ Phase 5    (skipped placeholder)
│   ├── test_chunking.py               ⏳ Phase 6    (skipped placeholder)
│   ├── test_embeddings.py             ⏳ Phase 4    (skipped placeholder)
│   ├── test_qdrant_indexing.py        ⏳ Phase 7    (skipped placeholder)
│   ├── test_qdrant_retrieval.py       ⏳ Phase 8    (skipped placeholder)
│   ├── test_metadata_filtering.py     ⏳ Phase 8    (skipped placeholder)
│   ├── test_reranking.py              ⏳ Phase 10   (skipped placeholder)
│   ├── test_ollama_generation.py      ⏳ Phase 3    (skipped placeholder)
│   ├── test_grounding.py              ⏳ Phase 10   (skipped placeholder)
│   ├── test_citations.py              ⏳ Phase 9    (skipped placeholder)
│   ├── test_prompt_injection.py       ⏳ Phase 10   (skipped placeholder)
│   └── test_conflicting_documents.py  ⏳ Phase 10   (skipped placeholder)
│
├── scripts/
│   ├── ingest.py                      ⏳ Phase 5    CLI: raw/ -> processed/
│   └── index.py                       ⏳ Phase 7    CLI: processed/ -> Qdrant
│
├── docker-compose.yml                 ✅ Phase 1    Qdrant service (done)
├── requirements.txt                   ✅ Phase 1    pinned local/open-source deps (done)
├── .env.example                       ✅ Phase 1    all configurable values (done)
└── README.md
```

Verification for this phase: `pytest --collect-only` succeeds across the
**entire** tree right now (14 tests collected, 0 import errors) — proof
every stub file is syntactically valid and importable before any real
logic is written.

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for Qdrant)
- [Ollama](https://ollama.com) installed on the host, with at least one model
  pulled, e.g.:
  ```bash
  ollama pull llama3.1
  # or: ollama pull qwen2.5
  # or: ollama pull mistral
  ```

### Commands to run

```bash
cd medical-rag

# 1. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your local env file
cp .env.example .env
# Edit .env if needed — in particular set OLLAMA_MODEL to whatever
# you've pulled locally.

# 4. Verify configuration loads correctly
python3 -c "
from app.config.settings import get_settings
s = get_settings()
print('Qdrant URL   :', s.qdrant_url)
print('Ollama URL   :', s.ollama_base_url)
print('Ollama model :', s.ollama_model)
print('Embed model  :', s.embedding_model)
"
```

### Expected output

```
Qdrant URL   : http://localhost:6333
Ollama URL   : http://localhost:11434
Ollama model : llama3.1
Embed model  : BAAI/bge-small-en-v1.5
```

If you see these four lines with your values from `.env`, Phase 1 is
complete and verified — configuration loading works end-to-end.

### Docker networking notes (for later phases)

- **Now (Phase 1–13, app runs on host):** the app calls Qdrant at
  `http://localhost:6333` and Ollama at `http://localhost:11434`, both
  reachable directly because everything is on `localhost`.
- **If the FastAPI app is later containerized too:** `localhost` inside a
  container refers to the container itself, not the host.
  - Qdrant would instead be reached via the Docker Compose service name,
    e.g. `http://qdrant:6333`, once the app joins the same Compose network.
  - Ollama running on the host would be reached via Docker's host gateway
    (`http://host.docker.internal:11434` on Mac/Windows, or
    `http://172.17.0.1:11434` / `--network host` on Linux), or Ollama itself
    could be added as a Compose service later.
  - `QDRANT_HOST` and `OLLAMA_BASE_URL` in `.env` are exactly the two values
    you'd change — no application code changes needed, which is the point of
    centralizing config in Phase 1.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pydantic_settings` | Dependencies not installed | Run `pip install -r requirements.txt` inside the activated venv |
| `ValidationError` on `Settings()` | A value in `.env` doesn't match the expected type (e.g. text in an int field) | Check `.env` against `.env.example`; fix the offending line |
| Settings load but show *default* values, not your `.env` values | `.env` file missing or not in the working directory | Confirm `.env` exists in `medical-rag/` (run `cp .env.example .env` from that directory) and that you're running commands from there |
| `python3.11: command not found` | Python 3.11 not installed | Install via `pyenv`, `brew install python@3.11` (Mac), or your OS package manager |
| Docker Compose command not found | Docker Desktop not installed / Compose plugin missing | Install Docker Desktop, or `docker compose version` to check |

---

## Development Workflow

This section explains *how* the remaining 13 phases will actually be built,
so it's clear what to expect turn by turn.

### 1. One phase per iteration, never parallel

Each phase touches one or two modules and is fully explained, coded, and
verified before the next one starts. Concretely, every phase follows the
same five steps:

1. **Explain** what's being built and why, in the context of the phase
   before and after it.
2. **Implement** the real logic in the stub file(s) for that phase — the
   `NotImplementedError` bodies and `# TODO` markers get replaced.
3. **Wire it to configuration** — every new module reads from
   `app.config.settings.get_settings()`, never hard-codes a value.
4. **Verify** — run the corresponding real test (the matching
   `tests/test_*.py` file is un-skipped and filled in), plus a manual
   command showing expected output, exactly like the Phase 1 verification
   above.
5. **Update this README's status table** (✅ / ⏳) and folder-structure
   annotations before moving on.

### 2. Dependency order (why phases are sequenced this way)

```
Phase 1  Environment  ──┐
                         ├─> Phase 2  Qdrant   ──┐
                         ├─> Phase 3  Ollama    ──┤
                         └─> Phase 4  Embeddings ─┤
                                                   ├─> Phase 5 Ingestion
                                                   ├─> Phase 6 Chunking+Metadata
                                                   ├─> Phase 7 Indexing (needs 2,4,5,6)
                                                   ├─> Phase 8 Retrieval (needs 2,4,7)
                                                   ├─> Phase 9 RAG pipeline (needs 3,8)
                                                   ├─> Phase 10 Grounding+Guardrails (needs 9)
                                                   ├─> Phase 11 FastAPI (needs 2-10)
                                                   ├─> Phase 12 Streamlit UI (needs 11)
                                                   ├─> Phase 13 Evaluation (needs 9-11)
                                                   └─> Phase 14 Agent integration design (needs all)
```

Phases 2, 3, and 4 have no dependency on each other and could in principle
be built in any order — they're sequenced 2→3→4 simply to match the spec.
Everything from Phase 5 onward is strictly sequential because each stage
consumes the previous stage's output (loaders → chunker → embeddings →
Qdrant → retriever → pipeline → API → UI).

### 3. Each module is independently testable

Because Phase 1 centralized configuration and every module already exists
as an importable stub, each subsequent phase can be tested in isolation
without standing up the full stack:

- `EmbeddingService` (Phase 4) is testable with no Qdrant or Ollama running.
- `QdrantStore` (Phase 2) is testable once `docker compose up qdrant` is
  running, with no Ollama or embedding model involved.
- `OllamaLLMService` (Phase 3) is testable with no Qdrant involved, once a
  model is pulled locally.
- The full `RAGPipeline` (Phase 9) is the first module that requires all
  three running together.

This mirrors the spec's final architectural principle: RAG, Ollama, Qdrant,
and (later) the Agent are independently deployable and independently
testable, not one monolith.

### 4. What you'll see each turn

Going forward, each phase reply will contain: an explanation, an
architecture diagram for that slice, real code replacing that phase's stub
file(s), a `pytest` run showing the new test(s) passing, the exact commands
to reproduce it yourself, and a troubleshooting table — same format as
Phase 1 above. Nothing gets marked ✅ in the status table without a passing
verification shown in the reply.

### 5. How to steer development

- Say **"continue to Phase N"** to move forward normally.
- Say **"pause after Phase N and let me test it myself"** if you want to
  run things locally (e.g. against your own Ollama models) before
  continuing.
- Ask to revisit any earlier phase at any time — e.g. "change the
  embedding model" or "add a field to the metadata schema" — and I'll
  update that phase's files plus flag any downstream phases it affects.

---

## What's next — Phase 2

Phase 2 stands up Qdrant (via the `docker-compose.yml` already created here),
implements a `QdrantStore` wrapper with collection creation, health checks,
and vector insertion/search, and verifies it against a live Qdrant instance.

Say **"continue to Phase 2"** when you're ready, or ask questions about
anything in Phase 1 first.

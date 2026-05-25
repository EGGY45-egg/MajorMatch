# Implementation & Edit Log

Purpose
-------

This document is the single source-of-truth for implementation notes, edits, and decisions made while building MajorMatch. Use this file to record what you implemented, why, and any follow-up tasks.

How to log an entry
-------------------

When you make a change, add a new entry at the top with the following fields:

- Date (YYYY-MM-DD)
- Author (initials or name)
- Area (UI / Backend / Data / Infra / Docs / ML)
- Files changed (workspace-relative paths)
- Summary: one-line description
- Details: optional short paragraph with links and next steps

Template
--------

```
Date: 2026-05-25
Author: RV
Area: Scaffold
Files changed:
- streamlit_app.py
- api/predict.py
- api/search.py
- data/courses.csv
- requirements.txt
- scripts/embed.py
Summary: Added initial repo scaffold and app stub
Details: Created minimal Streamlit app, prediction/search stubs, and example course corpus. Next: wire real embeddings and the ML model.
```

Implementation Log
------------------

Date: 2026-05-25
Author: RV
Area: Scaffold
Files changed:
- streamlit_app.py
- api/predict.py
- api/search.py
- data/courses.csv
- requirements.txt
- scripts/embed.py
Summary: Initial scaffold added (Streamlit app, API stubs, sample data).
Details: Vertical MVP stub implemented to allow end-to-end testing without external services. Teammate will add the serialized ML model under `models/` or expose an API; embeddings will be computed and stored in PostgreSQL with `pgvector` later.

Date: 2026-05-25
Author: RV
Area: MVP
Files changed:
- api/search.py
Summary: Add embedding-backed semantic search with fallback
Details: `api/search.py` now attempts to load `sentence-transformers` and compute corpus embeddings on demand. If the library or model fails to load, the code falls back to the earlier keyword scoring method. This enables a stronger vertical MVP while keeping the repo runnable without heavy dependencies.

Date: 2026-05-25
Author: RV
Area: UI
Files changed:
- app_logic.py
- streamlit_app.py
- scripts/smoke_test.py
Summary: Reworked the Streamlit MVP into a clearer end-to-end flow
Details: The UI now guides users through profile input, career-track recommendation, and course discovery in one screen. The core logic moved into `app_logic.py` so it can be smoke-tested independently with `scripts/smoke_test.py`.

Date: 2026-05-25
Author: RV
Area: Validation
Files changed:
- app_logic.py
- streamlit_app.py
- scripts/smoke_test.py
- api/search.py
- api/predict.py
Summary: Syntax validation passed on the edited Python files
Details: The terminal smoke test was skipped by the environment, so I validated the edited Python files through the editor error checker. No syntax or static errors were reported for the current MVP slice.

Date: 2026-05-25
Author: RV
Area: Backend / UI
Files changed:
- app_logic.py
- api/ollama.py
- streamlit_app.py
Summary: Added an optional Ollama interview flow with slider fallback
Details: The app can now ask the user questions through Ollama, build a structured coding/math/design profile, and still keep the slider controls inside a debug expander. If Ollama is unavailable, the interview falls back to local prompts so the MVP stays demoable.

Date: 2026-05-25
Author: RV
Area: Backend / UI
Files changed:
- api/ollama.py
- streamlit_app.py
Summary: Auto-resolve an installed Ollama chat model instead of hardcoding a missing default
Details: The adapter now prefers installed chat models from `ollama list` and defaults to `llama3.2:1b`, which matches the model already present in your environment. The Streamlit UI also shows which chat model is being used.

Date: 2026-05-25
Author: RV
Area: Backend / UI
Files changed:
- api/ollama.py
Summary: Suppress brace-only Ollama replies and infer profile updates from user text
Details: The interview flow now extracts useful profile signals from the user’s natural-language response before calling Ollama, and it replaces malformed brace-only assistant output with a real follow-up question based on missing fields.

Date: 2026-05-25
Author: RV
Area: Backend / UI
Files changed:
- api/ollama.py
Summary: Switch Ollama interview from JSON-only output to natural chat replies
Details: The assistant now talks in plain text, while the app still infers and merges profile scores separately. This avoids brace-only replies and keeps the chat layer usable even when the model does not strictly follow a JSON format.

Date: 2026-05-25
Author: RV
Area: Backend / UI
Files changed:
- api/ollama.py
- streamlit_app.py
Summary: Remove Ollama fallbacks so debug failures surface directly
Details: The chat flow now raises errors when Ollama is unavailable or returns unusable output. The UI no longer tells the user there is a silent fallback path; it now reports that Ollama must be running for chat debugging.

Date: 2026-05-25
Author: RV
Area: Backend / Data / UI
Files changed:
- course_index.py
- api/search.py
- app_logic.py
- streamlit_app.py
- scripts/embed.py
- requirements.txt
Summary: Added pgvector-backed embeddings, semantic search, and PCA/UMAP/t-SNE projections
Details: Course records are now stored in PostgreSQL with pgvector embeddings and 2D coordinates for PCA, UMAP, and t-SNE. The Streamlit app can rebuild the index from the CSV corpus and render a projection scatter plot for any of the three methods.

Date: 2026-05-25
Author: RV
Area: Backend / Data / Infra
Files changed:
- course_index.py
- scripts/embed.py
- api/search.py
- api/ollama.py
- app_logic.py
- streamlit_app.py
Summary: Make embeddings robust across developer environments and fix import/indexing issues
Details: Attempts to enable the vector extension on the DB but falls back to storing embeddings as float[] (ARRAY(Float)) when the extension is unavailable. Similarity search now uses a Python cosine-similarity fallback (loading stored embeddings into memory) so search works without pgvector. The course indexer and embed.py were refactored so the database is created/initialized before use and the embed script can import project modules reliably.


Date: 2026-05-25
Author: RV
Area: Backend / Data / API
Files changed:
- course_index.py
- api/search.py
- scripts/embed.py
- app_logic.py
- streamlit_app.py
- docs/IMPLEMENTATION.md
Summary: Implement semantic search API and make embedding/indexing robust
Details: Implemented a stable `semantic_search()` API wrapper that validates inputs, strips embeddings from responses, and returns consistent metadata (`id`, `title`, `description`, `score`, `score_normalized`, and projection coordinates). Refactored embedding storage to use `float[]` (`ARRAY(Float)`) for portability, added a Python cosine-similarity fallback when server-side `pgvector` is unavailable, and updated the indexer (`scripts/embed.py`) to support indexing multiple CSV files under `data/`. Also fixed import issues by ensuring the embed script can import project modules and added CSV header expectations to the docs. Next: add smoke tests and UI polish for provider/fallback visibility.


Outstanding / Next Steps
-----------------------

- Replace `api/predict.py` with the teammate's serialized model or an API wrapper.
- Add `docs/ARCHITECTURE.md` and `docs/DEPLOY.md` if we finalize infra choices.

Notes
-----

- Keep entries short and chronological (newest at top).
- Link to specific files when referring to code (use workspace-relative paths).

- CSV header expectations: each CSV indexed by `scripts/embed.py` must include `title` and `description` columns (case-insensitive). Rows missing those fields are skipped; use UTF-8 encoding and standard CSV quoting.

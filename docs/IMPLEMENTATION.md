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
Area: UI / Backend
Files changed:
- app_logic.py
- streamlit_app.py
- docs/IMPLEMENTATION.md
Summary: Switch to chatbot-first UI and show tool-specific interfaces only when tools are actually invoked
Details: Reworked the Streamlit experience to match a chat-native flow similar to ChatGPT/Gemini behavior. The app now starts as a pure chatbot and does not pre-render prediction, career-context, course, or visualization sections by default. Added deterministic intent routing in `app_logic.py` so tools are called only when the user message clearly asks for those features (to reduce hallucination risk with lightweight local models). Tool-specific UIs are now conditionally rendered only when that tool was used for the latest user request.



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

Date: 2026-05-25
Author: RV
Area: UI / Backend
Files changed:
- course_index.py
- streamlit_app.py
- docs/IMPLEMENTATION.md
Summary: Show query point and highlight top-match in projection plot
Details: Added `project_courses_with_query()` to `course_index.py` to compute 2D projections that include the user's query so points share a consistent coordinate system. Updated `streamlit_app.py` to render the query point on the projection and color the top match differently (colors: course=blue, top_match=red, query=green). Also surfaced embedding model name and `pgvector` availability in the UI so users can see whether the app is using the in-Python similarity fallback or server vector ops. Note: PCA/UMAP/TSNE projections are computed on-demand and may add CPU cost for large corpora; next steps include annotating the top match, increasing query marker size, and caching projection reducers for faster UI response.

Date: 2026-05-25
Author: RV
Area: Docs / Validation
Files changed:
- tests/test_course_index.py
- tests/test_api_search.py
- tests/README.md
- requirements.txt
Summary: Add a dedicated tests directory with pytest coverage for corpus loading and semantic search
Details: Created a new `tests/` package with lightweight pytest checks that cover CSV corpus loading, the `semantic_search()` result shape, and the query projection helper. Added `pytest` to `requirements.txt` and verified the suite with `python -m pytest` inside the project virtual environment. The tests are intentionally isolated from the database by monkeypatching the core helpers so they run quickly and reliably in a capstone-sized corpus.

Date: 2026-05-25
Author: RV
Area: Backend / UI / Data
Files changed:
- api/jobs.py
- app_logic.py
- streamlit_app.py
- tests/test_jobs.py
Summary: Integrate an Adzuna-backed career context API into the user flow
Details: Added a small job-market adapter in `api/jobs.py` that queries Adzuna using `ADZUNA_APP_ID` and `ADZUNA_APP_KEY`, derives a stable career-context payload (`job_count`, salary range, top job titles, top companies), and degrades gracefully when credentials are missing. The Streamlit app now renders a dedicated career-context section immediately after the track recommendation so the user can see market data before exploring courses. Added pytest coverage for the happy path and the missing-credential fallback. Next: wire the same career-context tool into the Ollama orchestrator so the assistant can decide when to call it.

Date: 2026-05-25
Author: RV
Area: Backend / UI / ML
Files changed:
- api/ollama.py
- api/orchestrator.py
- streamlit_app.py
- tests/test_orchestrator.py
Summary: Add an Ollama tool-calling orchestrator for prediction, job context, and course search
Details: Introduced a shared Ollama `chat_completion()` helper that supports tool definitions, then added `api/orchestrator.py` to run a tool-calling conversation loop using the three main MajorMatch capabilities: career-track prediction, Adzuna job context, and semantic course search. The Streamlit app now includes a dedicated orchestrated-assistant panel so users can ask one question and let Ollama decide which tools to invoke. Added pytest coverage that simulates a multi-tool conversation and verifies the tool trace and returned artifacts. Next: refine the prompt/tool schema and consider showing orchestrator artifacts in a more compact UI card layout.



Outstanding / Next Steps
-----------------------

- Replace `api/predict.py` with the teammate's serialized model or an API wrapper.
- Add `docs/ARCHITECTURE.md` and `docs/DEPLOY.md` if we finalize infra choices.

Notes
-----

- Keep entries short and chronological (newest at top).
- Link to specific files when referring to code (use workspace-relative paths).

- CSV header expectations: each CSV indexed by `scripts/embed.py` must include `title` and `description` columns (case-insensitive). Rows missing those fields are skipped; use UTF-8 encoding and standard CSV quoting.

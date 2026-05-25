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


Outstanding / Next Steps
-----------------------

- Replace `api/predict.py` with the teammate's serialized model or an API wrapper.
- Implement embeddings using `sentence-transformers` and store in PostgreSQL + `pgvector`.
- Replace `api/search.py` with an embeddings-backed retrieval function.
- Add `docs/ARCHITECTURE.md` and `docs/DEPLOY.md` if we finalize infra choices.

Notes
-----

- Keep entries short and chronological (newest at top).
- Link to specific files when referring to code (use workspace-relative paths).

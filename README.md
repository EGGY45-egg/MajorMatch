# MajorMatch

MajorMatch is a chat-first semantic course and career pathfinder for students, advisors, and freshmen.

## What it does
- Chat assistant that answers normally when it can and uses tools only when useful.
- Ollama tool-calling orchestrator for career prediction, career context, and semantic course search.
 - Career-track prediction that accepts `selected_features` (model feature names). The front-end can open an interactive prediction UI when requested; a lightweight fallback remains optional.
- Semantic course search with PCA, UMAP, and t-SNE projections in the Streamlit UI.
- Latest tool output is shown as a single artifact panel to keep the UI clean.

## Current implementation
- `streamlit_app.py`: chat-first Streamlit app and UI rendering.
- `api/orchestrator.py`: tool-calling loop and grounded final replies.
- `api/ollama.py`: Ollama transport, including streaming support.
- `api/predict.py`: track prediction helper. The predictor expects `selected_features` (an array of feature names) or a sequence of selected features; it exposes a `use_fallback` flag to control rule-based fallbacks.
- `api/search.py` and `course_index.py`: semantic search and course projections.
- `api/jobs.py`: career-context lookup.

## Documentation
- [Project documentation](docs/PROJECT_DOCUMENTATION.md): consolidated overview of the current architecture, data flow, setup, and implementation notes.
- [Setup instructions](docs/instruction.md): dedicated local setup guide for Python, PostgreSQL, Ollama, indexing, and validation.
- [Implementation log](docs/IMPLEMENTATION.md): chronological record of changes and decisions.

## Setup
See the full setup guide in [docs/instruction.md](docs/instruction.md).

Quick start:

```powershell
pip install -r requirements.txt
python scripts/embed.py
streamlit run streamlit_app.py
```

## Environment variables
- `DATABASE_URL`: PostgreSQL connection string.
- `OLLAMA_BASE_URL`: Ollama server URL, defaults to `http://localhost:11434`.
- `OLLAMA_MODEL`: Chat model name, defaults to `llama2:latest` with fallback selection when tools are requested.
- `EMBEDDING_MODEL`: Sentence-transformer model used for course embeddings.
- `ADZUNA_APP_ID` / `ADZUNA_APP_KEY`: Optional Adzuna credentials for live career context.

## Tests
Run the core test suite with:

```powershell
$env:PYTHONPATH='.'; .\venv\Scripts\python -m pytest tests/test_orchestrator.py -q
```

## Notes
- The app uses tool calling automatically for relevant prompts such as salaries, career outlook, predictions, and course search. Structured user "profiles" have been removed — predictions are driven from selected model feature names or the interactive prediction UI.
- If `pgvector` is unavailable, the search layer falls back to a portable embedding storage/search path.
- The Streamlit UI is intentionally minimal and chat-first.


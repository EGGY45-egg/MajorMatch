# MajorMatch Setup Instructions

This guide covers the minimum setup needed to run MajorMatch locally, including PostgreSQL storage for the course index.

## 1. Prerequisites

- Python 3.10+.
- A virtual environment for the project.
- PostgreSQL 14+.
- Ollama installed locally and running if you want the chat assistant and tool-calling flow.

## 2. Install Python Dependencies

Create and activate a virtual environment, then install the requirements:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Set Up PostgreSQL

MajorMatch stores course records, embeddings, and projection coordinates in PostgreSQL.

Create a database and user if you do not already have one:

```sql
CREATE DATABASE semantic_search;
CREATE USER postgres WITH PASSWORD 'postgres';
GRANT ALL PRIVILEGES ON DATABASE semantic_search TO postgres;
```

You can also use your own database name and credentials. The app reads the connection string from `DATABASE_URL`.

Example local connection string:

```powershell
$env:DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/semantic_search"
```

## 4. PostgreSQL Extension Notes

MajorMatch tries to create the `vector` extension automatically if it is available.

- If the extension exists, you can keep using it.
- If the extension is not available, the app still works by storing embeddings as `float[]` and using a portable fallback search path.
- Because of that fallback, `pgvector` is helpful but not required for the app to run.

If you want to enable it manually, run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 5. Ollama Setup

Start Ollama before using the chat assistant:

```powershell
ollama serve
```

Optional environment variables:

- `OLLAMA_BASE_URL`: defaults to `http://localhost:11434`.
- `OLLAMA_MODEL`: defaults to `llama2:latest`, with fallback model selection when tools are requested.

If you want a model that is more likely to support tools, pull one of the local fallback models first, for example:

```powershell
ollama pull llama3.2:1b
```

## 6. Optional Career Context Credentials

If you want live job-market data, set Adzuna credentials:

```powershell
$env:ADZUNA_APP_ID = "your_app_id"
$env:ADZUNA_APP_KEY = "your_app_key"
```

Without these values, the career-context tool falls back gracefully.

## 7. Build the Course Index

MajorMatch reads course data from the `data/` folder and stores the indexed corpus in PostgreSQL.

Run the indexer after the database is ready:

```powershell
python scripts/embed.py
```

You can also point it at a specific CSV file or folder:

```powershell
python scripts/embed.py data\courses.csv
```

The CSV files must include `title` and `description` columns. Rows without those fields are skipped.

## 8. Run the App

Start the Streamlit app after the database and index are ready:

```powershell
streamlit run streamlit_app.py
```

## 9. Quick Validation

Run the main test target:

```powershell
$env:PYTHONPATH='.'; .\venv\Scripts\python -m pytest tests/test_orchestrator.py -q
```

## 10. What Gets Stored in PostgreSQL

- Course title and description.
- Embedding vector as a `float[]` column.
- 2D projection coordinates for PCA, UMAP, and t-SNE.

That database table is what powers semantic search and the projection plot in the UI.
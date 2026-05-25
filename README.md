# MajorMatch

MajorMatch is a semantic course and career pathfinder.

Current implementation
- Chat-first Ollama interview that builds a structured profile.
- PostgreSQL + pgvector-backed semantic search over the course corpus.
- PCA, UMAP, and t-SNE course projection plots in the Streamlit UI.

Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set `DATABASE_URL` to your PostgreSQL database.
4. Build the course index:

```powershell
python scripts/embed.py
```

5. Run the app:

```powershell
streamlit run streamlit_app.py
```

Environment variables
- `DATABASE_URL`: PostgreSQL connection string.
- `OLLAMA_BASE_URL`: Ollama server URL, defaults to `http://localhost:11434`.
- `OLLAMA_MODEL`: Chat model name, defaults to `llama3.2:1b`.
- `EMBEDDING_MODEL`: Sentence-transformer model used for course embeddings.


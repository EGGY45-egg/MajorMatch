# MajorMatch

Minimal scaffold for the MajorMatch GenAI app.

Quickstart
---------

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

What is included
- `streamlit_app.py`: lightweight UI to input profile and search courses
- `api/predict.py`: prediction stub (replace with teammate's model)
- `api/search.py`: keyword-based search stub (replace with embeddings + pgvector)
- `data/courses.csv`: small curated course corpus
- `scripts/embed.py`: placeholder for embedding computation

Next steps
- Implement the ML model and have the teammate add a serialized model or an API endpoint.
- Compute embeddings (use `sentence-transformers`) and store in PostgreSQL with `pgvector`.
- Replace the search stub with an embeddings-backed retrieval layer.


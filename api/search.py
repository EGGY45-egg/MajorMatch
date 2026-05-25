import csv
from typing import List, Dict
import logging

_logger = logging.getLogger(__name__)

# Optional embedding utilities (lazy-imported)
_st_model = None
_embeddings = None
_course_texts = None


def load_courses(path: str = "data/courses.csv") -> List[Dict]:
    courses = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                courses.append({"title": row.get("title", ""), "description": row.get("description", "")})
    except FileNotFoundError:
        return []
    return courses


def _init_sentence_transformer():
    global _st_model
    if _st_model is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        _logger.info("Loaded SentenceTransformer model")
    except Exception as e:
        _logger.warning("sentence-transformers unavailable or failed to load: %s", e)
        _st_model = None
    return _st_model


def _ensure_embeddings(courses: List[Dict]):
    """Compute and cache embeddings for the course corpus if model is available."""
    global _st_model, _embeddings, _course_texts
    if _embeddings is not None and _course_texts == [c.get("title", "") + " " + c.get("description", "") for c in courses]:
        return
    model = _init_sentence_transformer()
    if model is None:
        return
    texts = [c.get("title", "") + " " + c.get("description", "") for c in courses]
    try:
        emb = model.encode(texts, convert_to_numpy=True)
        _embeddings = emb
        _course_texts = texts
    except Exception as e:
        _logger.warning("Failed to compute embeddings: %s", e)
        _embeddings = None


def _keyword_search(query: str, courses: List[Dict], top_k: int = 5) -> List[Dict]:
    q = query.lower().strip()
    scored = []
    for c in courses:
        text = (c.get("title", "") + " " + c.get("description", "")).lower()
        score = 0
        if q in text:
            score += 10
        for w in q.split():
            if w in text:
                score += 1
        scored.append((score, c))
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda x: -x[0])
    return [c for score, c in scored][:top_k]


def semantic_search(query: str, courses: List[Dict], top_k: int = 5) -> List[Dict]:
    """Search courses semantically.

    If `sentence-transformers` is installed, use embedding similarity. Otherwise fall back to keyword scoring.
    """
    if not courses:
        return []

    model = _init_sentence_transformer()
    if model is None:
        return _keyword_search(query, courses, top_k)

    # ensure embeddings available
    _ensure_embeddings(courses)
    if _embeddings is None:
        return _keyword_search(query, courses, top_k)

    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        q_emb = model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, _embeddings)[0]
        idxs = list(reversed(np.argsort(sims).tolist()))
        results = []
        for i in idxs:
            if sims[i] <= 0:
                continue
            results.append(courses[i])
            if len(results) >= top_k:
                break
        return results
    except Exception as e:
        _logger.warning("Embedding search failed, falling back to keyword search: %s", e)
        return _keyword_search(query, courses, top_k)


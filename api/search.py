import csv
from typing import List, Dict

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

def semantic_search(query: str, courses: List[Dict], top_k: int = 5) -> List[Dict]:
    """Very small, local semantic search approximation using keyword scoring.
    Replace with embeddings + pgvector later.
    """
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

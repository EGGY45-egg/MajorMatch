from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from api.jobs import get_career_context
from api.predict import predict_track
from api.search import semantic_search


BASE_DIR = Path(__file__).resolve().parent
# Point to the data directory — support multiple CSV files under `data/`.
DATA_DIR = BASE_DIR / "data"
# Collect CSV files if present; callers can use `DATA_DIR` or `COURSE_CSVS`.
COURSE_CSVS = sorted([p for p in DATA_DIR.glob("*.csv")]) if DATA_DIR.exists() else []
TOOL_INTENT_KEYWORDS = {
    "career_track": [
        "recommend",
        "recommendation",
        "career track",
        "predict",
        "prediction",
        "what should i study",
        "what career",
    ],
    "career_context": [
        "job market",
        "salary",
        "salaries",
        "job count",
        "jobs",
        "demand",
        "market",
        "practical",
    ],
    "course_search": [
        "course",
        "courses",
        "class",
        "classes",
        "syllabus",
        "search",
        "find me",
        "learn",
    ],
    "visualization": [
        "visual",
        "visualize",
        "map",
        "scatter",
        "cluster",
        "plot",
        "show me the map",
        "show the map",
    ],
}





def recommend_track(features: dict | Sequence[str]) -> Dict[str, object]:
    """
    Recommend a track based on either a features dict (deprecated profile-like dict)
    or a sequence of selected feature names. For compatibility, forward to
    `predict_track()` which accepts either a dict or a sequence.
    """
    prediction = predict_track(features)
    if isinstance(prediction, dict):
        result = {
            "track": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "category": prediction.get("category"),
            "source": prediction.get("source"),
        }
        if "top_predictions" in prediction:
            result["top_predictions"] = prediction.get("top_predictions")
        return result

    track, confidence = prediction
    return {"track": track, "confidence": confidence}


def suggest_courses(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    capped_top_k = max(1, min(int(top_k), 5))
    return semantic_search(query, top_k=capped_top_k)


def suggest_career_context(track: str, location: str = "United States"):
    return get_career_context(track, location=location)


def detect_tool_intents(message: str) -> List[str]:
    lowered = (message or "").lower()
    intents: List[str] = []
    for intent, keywords in TOOL_INTENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            intents.append(intent)

    return intents


def build_search_query_from_message(message: str, fallback_track: str) -> str:
    lowered = (message or "").lower().strip()
    if not lowered:
        return build_course_query(fallback_track)

    stop_phrases = [
        "show me",
        "find me",
        "courses for",
        "classes for",
        "search for",
        "look for",
        "recommend",
        "recommendations",
        "courses",
        "classes",
    ]
    query = message.strip()
    for phrase in stop_phrases:
        if phrase in lowered:
            query = query.lower().replace(phrase, "")
    query = " ".join(query.split()).strip(" ,.")
    return query or build_course_query(fallback_track)


def build_course_query(track: str) -> str:
    track_map = {
        "Software Engineer": "programming software engineering systems web development",
        "Data Scientist": "data science statistics machine learning analytics",
        "Product Designer": "ux design human computer interaction prototyping",
    }
    return track_map.get(track, track.lower())


def summarize_matches(courses: Sequence[Dict[str, str]]) -> str:
    if not courses:
        return "No matches found yet. Try a broader search phrase."
    titles = [course.get("title", "") for course in courses[:5] if course.get("title")]
    if not titles:
        return "No course titles available in the current results."
    if len(titles) == 1:
        return f"Top match: {titles[0]}"
    return f"Top matches: {', '.join(titles)}"

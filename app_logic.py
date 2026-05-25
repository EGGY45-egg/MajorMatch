from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from api.predict import predict_track
from api.search import load_courses, semantic_search


BASE_DIR = Path(__file__).resolve().parent
COURSE_CSV = BASE_DIR / "data" / "courses.csv"


@dataclass(frozen=True)
class ProfileInput:
    coding: int
    math: int
    design: int


def recommend_track(profile: ProfileInput) -> Dict[str, object]:
    track, confidence = predict_track(
        {"coding": profile.coding, "math": profile.math, "design": profile.design}
    )
    return {"track": track, "confidence": confidence}


def suggest_courses(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    courses = load_courses(str(COURSE_CSV))
    if not query.strip():
        return courses[:top_k]
    return semantic_search(query, courses, top_k=top_k)


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
    titles = [course.get("title", "") for course in courses[:3] if course.get("title")]
    if not titles:
        return "No course titles available in the current results."
    if len(titles) == 1:
        return f"Top match: {titles[0]}"
    return f"Top matches: {', '.join(titles)}"

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from api.predict import predict_track
from api.search import load_courses, semantic_search


BASE_DIR = Path(__file__).resolve().parent
COURSE_CSV = BASE_DIR / "data" / "courses.csv"
PROFILE_FIELDS = ("coding", "math", "design")


@dataclass(frozen=True)
class ProfileInput:
    coding: int
    math: int
    design: int


def normalize_score(value: object) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(10, score))


def default_profile_values(default: int = 0) -> Dict[str, int]:
    return {field: default for field in PROFILE_FIELDS}


def coerce_profile_values(values: Dict[str, object]) -> Dict[str, int]:
    profile = default_profile_values()
    for field in PROFILE_FIELDS:
        if field in values:
            profile[field] = normalize_score(values[field])
    return profile


def merge_profile_values(base: Dict[str, object], updates: Dict[str, object]) -> Dict[str, int]:
    merged = coerce_profile_values(base)
    for field in PROFILE_FIELDS:
        if field in updates:
            merged[field] = normalize_score(updates[field])
    return merged


def profile_is_empty(profile: Dict[str, object]) -> bool:
    coerced = coerce_profile_values(profile)
    return all(coerced[field] == 0 for field in PROFILE_FIELDS)


def profile_from_sliders(coding: int, math: int, design: int) -> Dict[str, int]:
    return coerce_profile_values({"coding": coding, "math": math, "design": design})


def profile_to_text(profile: Dict[str, object]) -> str:
    coerced = coerce_profile_values(profile)
    return ", ".join(f"{field}={coerced[field]}" for field in PROFILE_FIELDS)


def missing_profile_fields(profile: Dict[str, object]) -> List[str]:
    coerced = coerce_profile_values(profile)
    return [field for field in PROFILE_FIELDS if coerced[field] == 0]


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

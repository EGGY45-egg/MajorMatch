import csv
from pathlib import Path
from typing import Dict, List

from course_index import (
    CourseIndexError,
    get_projection_points,
    list_courses,
    load_course_corpus,
    rebuild_course_index,
    search_courses,
)


def load_courses(path: str = "data/courses.csv") -> List[Dict]:
    return load_course_corpus(Path(path))


def semantic_search(query: str, courses: List[Dict] | None = None, top_k: int = 5) -> List[Dict]:
    del courses
    return search_courses(query, top_k=top_k)


def get_course_projection_points(method: str = "pca"):
    return get_projection_points(method)


def rebuild_index(path: str = "data/courses.csv") -> int:
    return rebuild_course_index(Path(path))


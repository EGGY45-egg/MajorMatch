from pathlib import Path

import numpy as np

import course_index


def test_load_course_corpus_reads_all_csvs(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "a.csv").write_text(
        "title,description\nIntro to Python,Learn Python basics\n",
        encoding="utf-8",
    )
    (data_dir / "b.csv").write_text(
        "title,description\nData Science,Work with data\n",
        encoding="utf-8",
    )

    rows = course_index.load_course_corpus(data_dir)

    assert len(rows) == 2
    assert rows[0]["title"] == "Intro to Python"
    assert rows[1]["title"] == "Data Science"


def test_project_courses_with_query_returns_query_point(monkeypatch):
    monkeypatch.setattr(
        course_index,
        "list_courses",
        lambda limit=None: [
            {
                "id": 1,
                "title": "Intro to Python",
                "description": "Learn Python basics",
                "embedding": [1.0, 0.0],
            }
        ],
    )
    monkeypatch.setattr(
        course_index,
        "compute_embeddings",
        lambda texts: np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        course_index,
        "_projection_matrix",
        lambda embeddings, method: np.asarray([[2.0, 3.0], [2.0, 3.0]], dtype=np.float32),
    )

    course_points, query_point = course_index.project_courses_with_query("Python basics", method="tsne")

    assert len(course_points) == 1
    assert course_points[0].title == "Intro to Python"
    assert query_point is not None
    assert query_point.title == "(query)"
    assert query_point.x == 2.0
    assert query_point.y == 3.0

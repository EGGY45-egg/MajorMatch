from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sqlalchemy import Float, Integer, String, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from pgvector.sqlalchemy import Vector


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_COURSE_CSV = BASE_DIR / "data" / "courses.csv"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
DATABASE_URL = os.getenv("DATABASE_URL")


class CourseIndexError(RuntimeError):
    pass


class Base(DeclarativeBase):
    pass


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
    pca_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pca_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    umap_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    umap_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tsne_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tsne_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


@dataclass(frozen=True)
class CoursePoint:
    id: int
    title: str
    description: str
    x: float
    y: float


def get_database_url() -> str:
    if not DATABASE_URL:
        raise CourseIndexError("DATABASE_URL is not set. Configure a PostgreSQL database before using retrieval or projections.")
    return DATABASE_URL


def get_engine():
    return create_engine(get_database_url(), future=True)


def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def ensure_database() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(engine)


def load_course_corpus(csv_path: Path | str = DEFAULT_COURSE_CSV) -> List[Dict[str, str]]:
    path = Path(csv_path)
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = (row.get("title") or "").strip()
            description = (row.get("description") or "").strip()
            if title and description:
                rows.append({"title": title, "description": description})
    return rows


def _load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def compute_embeddings(texts: Sequence[str]) -> np.ndarray:
    model = _load_embedding_model()
    embeddings = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def _projection_matrix(embeddings: np.ndarray, method: str) -> np.ndarray:
    method = method.lower().strip()
    if embeddings.shape[0] == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32)

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        return reducer.fit_transform(embeddings)

    if method == "umap":
        import umap

        neighbors = max(2, min(5, embeddings.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(2, min(5, embeddings.shape[0] - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    raise CourseIndexError(f"Unsupported projection method: {method}")


def rebuild_course_index(csv_path: Path | str = DEFAULT_COURSE_CSV) -> int:
    ensure_database()
    rows = load_course_corpus(csv_path)
    if not rows:
        return 0

    texts = [f"{row['title']} {row['description']}" for row in rows]
    embeddings = compute_embeddings(texts)
    projections = {
        method: _projection_matrix(embeddings, method)
        for method in ("pca", "umap", "tsne")
    }

    session_factory = get_session_factory()
    with session_factory() as session:
        session.execute(delete(Course))
        for index, row in enumerate(rows):
            course = Course(
                title=row["title"],
                description=row["description"],
                embedding=embeddings[index].tolist(),
                pca_x=float(projections["pca"][index][0]),
                pca_y=float(projections["pca"][index][1]),
                umap_x=float(projections["umap"][index][0]),
                umap_y=float(projections["umap"][index][1]),
                tsne_x=float(projections["tsne"][index][0]),
                tsne_y=float(projections["tsne"][index][1]),
            )
            session.add(course)
        session.commit()
    return len(rows)


def _get_projection_columns(method: str):
    method = method.lower().strip()
    mapping = {
        "pca": (Course.pca_x, Course.pca_y),
        "umap": (Course.umap_x, Course.umap_y),
        "tsne": (Course.tsne_x, Course.tsne_y),
    }
    if method not in mapping:
        raise CourseIndexError(f"Unsupported projection method: {method}")
    return mapping[method]


def search_courses(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    if not query.strip():
        return list_courses(limit=top_k)

    ensure_database()
    query_embedding = compute_embeddings([query])[0].tolist()
    session_factory = get_session_factory()
    with session_factory() as session:
        stmt = (
            select(Course)
            .order_by(Course.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        rows = session.execute(stmt).scalars().all()
        return [course_to_dict(course) for course in rows]


def list_courses(limit: Optional[int] = None) -> List[Dict[str, str]]:
    ensure_database()
    session_factory = get_session_factory()
    with session_factory() as session:
        stmt = select(Course)
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [course_to_dict(course) for course in rows]


def get_projection_points(method: str = "pca") -> List[CoursePoint]:
    ensure_database()
    x_col, y_col = _get_projection_columns(method)
    session_factory = get_session_factory()
    with session_factory() as session:
        stmt = select(Course.id, Course.title, Course.description, x_col, y_col).order_by(Course.id)
        rows = session.execute(stmt).all()
        points: List[CoursePoint] = []
        for row in rows:
            x_val = row[3]
            y_val = row[4]
            if x_val is None or y_val is None:
                continue
            points.append(
                CoursePoint(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    x=float(x_val),
                    y=float(y_val),
                )
            )
        return points


def course_to_dict(course: Course) -> Dict[str, str]:
    return {"title": course.title, "description": course.description}

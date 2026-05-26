from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml_model"
MODEL_PATH = MODEL_DIR / "majormatch_rf_model.pkl"
ENCODER_PATH = MODEL_DIR / "majormatch_encoder.pkl"
FEATURES_PATH = MODEL_DIR / "majormatch_features.pkl"
TRAINING_DATA_PATH = MODEL_DIR / "stud_training.csv"

PROFILE_FIELDS = ("coding", "math", "design")

FEATURE_GROUP_KEYWORDS = {
    "coding": ("coding", "computer", "software", "program", "information technology", "engineering", "electronics", "electrical", "mechanic", "research", "puzzle"),
    "math": ("math", "physics", "science", "economics", "account", "analysis", "statistics", "research"),
    "design": ("design", "drawing", "photography", "architecture", "craft", "makeup", "art", "graphic", "visual", "cartoon"),
}

LABEL_CATEGORY_KEYWORDS = {
    "Software Engineer": ("computer", "software", "engineering", "information technology", "bca", "cs", "electronics", "electrical", "mechanical", "civil"),
    "Data Scientist": ("data", "science", "statistics", "math", "physics", "economics", "analytics", "research"),
    "Product Designer": ("design", "art", "visual", "architecture", "photography", "fashion", "journalism", "graphics"),
}


@dataclass(frozen=True)
class ModelBundle:
    model: object
    encoder: LabelEncoder
    feature_columns: Sequence[str]


def _normalize_score(value: object) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(10, score))


def _normalize_profile(features: dict) -> Dict[str, int]:
    profile = {field: 0 for field in PROFILE_FIELDS}
    for field in PROFILE_FIELDS:
        if field in features:
            profile[field] = _normalize_score(features.get(field))
    return profile


def _score_bucket(score: int) -> int:
    if score >= 7:
        return 3
    if score >= 4:
        return 2
    if score > 0:
        return 1
    return 0


def _load_artifacts_from_disk() -> ModelBundle | None:
    if MODEL_PATH.exists() and ENCODER_PATH.exists() and FEATURES_PATH.exists():
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        feature_columns = list(joblib.load(FEATURES_PATH))
        if isinstance(encoder, LabelEncoder) and feature_columns:
            return ModelBundle(model=model, encoder=encoder, feature_columns=feature_columns)

    if not TRAINING_DATA_PATH.exists():
        return None

    training_frame = pd.read_csv(TRAINING_DATA_PATH)
    training_frame.columns = [str(column).strip() for column in training_frame.columns]
    if "Courses" not in training_frame.columns:
        raise ValueError("Training CSV must include a 'Courses' target column.")

    feature_columns = [column for column in training_frame.columns if column != "Courses"]
    cleaned_targets = training_frame["Courses"].astype(str).str.strip()

    encoder = LabelEncoder()
    encoded_targets = encoder.fit_transform(cleaned_targets)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(training_frame[feature_columns], encoded_targets)

    return ModelBundle(model=model, encoder=encoder, feature_columns=feature_columns)


@lru_cache(maxsize=1)
def _get_model_bundle() -> ModelBundle:
    bundle = _load_artifacts_from_disk()
    if bundle is None:
        raise FileNotFoundError("No teammate model artifacts or training CSV were found in ml_model/.")
    return bundle


def get_prediction_feature_columns() -> List[str]:
    """Return the feature columns used by the teammate classifier."""
    return list(_get_model_bundle().feature_columns)


def _build_feature_groups(feature_columns: Sequence[str]) -> Dict[str, List[str]]:
    grouped_columns: Dict[str, List[str]] = {field: [] for field in PROFILE_FIELDS}
    for column in feature_columns:
        lowered = column.lower()
        for field, keywords in FEATURE_GROUP_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                grouped_columns[field].append(column)
                break
    return grouped_columns


def _profile_to_feature_row(profile: Dict[str, int], feature_columns: Sequence[str]) -> Dict[str, int]:
    grouped_columns = _build_feature_groups(feature_columns)
    feature_row = {column: 0 for column in feature_columns}

    for field, score in profile.items():
        bucket = _score_bucket(score)
        if bucket == 0:
            continue

        matching_columns = grouped_columns.get(field, [])
        if not matching_columns:
            continue

        if bucket == 1:
            selected_columns = matching_columns[:1]
        elif bucket == 2:
            selected_columns = matching_columns[:3]
        else:
            selected_columns = matching_columns

        for column in selected_columns:
            feature_row[column] = 1

    if not any(feature_row.values()):
        fallback_field = max(profile.items(), key=lambda item: item[1])[0]
        fallback_columns = grouped_columns.get(fallback_field, [])
        if fallback_columns:
            feature_row[fallback_columns[0]] = 1

    return feature_row


def _selected_features_to_feature_row(selected_features: Sequence[str], feature_columns: Sequence[str]) -> Dict[str, int]:
    selected = {str(feature).strip() for feature in selected_features if str(feature).strip()}
    feature_row = {column: 0 for column in feature_columns}
    for column in feature_columns:
        if column in selected:
            feature_row[column] = 1

    if not any(feature_row.values()) and feature_columns:
        feature_row[feature_columns[0]] = 1

    return feature_row


def _label_to_category(label: str) -> str | None:
    normalized = (label or "").strip().lower()
    for category, keywords in LABEL_CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return None


def _rule_based_predict(profile: Dict[str, int]) -> Dict[str, object]:
    coding = profile.get("coding", 0)
    math = profile.get("math", 0)
    design = profile.get("design", 0)

    if coding >= math and coding >= design:
        label = "Software Engineer"
        confidence = coding / 10.0
    elif math >= coding and math >= design:
        label = "Data Scientist"
        confidence = math / 10.0
    else:
        label = "Product Designer"
        confidence = design / 10.0

    return {
        "label": label,
        "confidence": confidence,
        "category": _label_to_category(label),
        "source": "fallback",
        "top_predictions": [
            {
                "label": label,
                "confidence": confidence,
                "category": _label_to_category(label),
            }
        ],
    }


def _build_top_predictions(bundle: ModelBundle, model_frame: pd.DataFrame) -> List[Dict[str, object]]:
    if not hasattr(bundle.model, "predict_proba"):
        return []

    probabilities = bundle.model.predict_proba(model_frame)[0]
    class_indices = list(range(len(probabilities)))
    ranked_indices = sorted(class_indices, key=lambda index: float(probabilities[index]), reverse=True)[:3]

    top_predictions: List[Dict[str, object]] = []
    for class_index in ranked_indices:
        predicted_label = bundle.encoder.inverse_transform([int(class_index)])[0]
        top_predictions.append(
            {
                "label": predicted_label,
                "confidence": float(probabilities[class_index]),
                "category": _label_to_category(predicted_label),
            }
        )

    return top_predictions


def predict_track(features: dict | Sequence[str]) -> Dict[str, object]:
    profile = _normalize_profile(features) if isinstance(features, dict) else {field: 0 for field in PROFILE_FIELDS}

    try:
        bundle = _get_model_bundle()
        if isinstance(features, dict):
            if any(key in PROFILE_FIELDS for key in features):
                model_input = _profile_to_feature_row(profile, bundle.feature_columns)
            else:
                model_input = _selected_features_to_feature_row(features.keys(), bundle.feature_columns)
        else:
            model_input = _selected_features_to_feature_row(features, bundle.feature_columns)
        model_frame = pd.DataFrame([model_input], columns=bundle.feature_columns)

        encoded_prediction = bundle.model.predict(model_frame)[0]
        predicted_label = bundle.encoder.inverse_transform([int(encoded_prediction)])[0]

        confidence = 0.0
        top_predictions: List[Dict[str, object]] = []
        if hasattr(bundle.model, "predict_proba"):
            probabilities = bundle.model.predict_proba(model_frame)[0]
            confidence = float(max(probabilities))
            top_predictions = _build_top_predictions(bundle, model_frame)

        return {
            "label": predicted_label,
            "confidence": confidence,
            "category": _label_to_category(predicted_label),
            "source": "model",
            "top_predictions": top_predictions,
        }
    except Exception:
        return _rule_based_predict(profile)

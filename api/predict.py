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

# Legacy structured profile fields and keyword groups removed. The
# predictor now expects a sequence of selected feature names (strings)
# or a dict whose keys are selected feature names. Converting numeric
# skill scores into model features has been removed.

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


# Removed: numeric-profile normalization and profile->feature mapping.
# Callers should pass selected feature names instead of numeric scores.


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


# Removed: functions that mapped high-level skill scores into the model's
# binary feature space. Use `_selected_features_to_feature_row` instead.


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


def _rule_based_predict(selected_features: Sequence[str]) -> Dict[str, object]:
    """Simple fallback that selects a label based on keywords found in the
    selected feature names. This is intentionally simple and only used when
    `use_fallback` is enabled.
    """
    joined = " ".join([str(s).lower() for s in selected_features or []])
    for label, keywords in LABEL_CATEGORY_KEYWORDS.items():
        if any(k in joined for k in keywords):
            return {
                "label": label,
                "confidence": 0.5,
                "category": _label_to_category(label),
                "source": "fallback",
                "top_predictions": [
                    {"label": label, "confidence": 0.5, "category": _label_to_category(label)}
                ],
            }

    # Default fallback
    default = list(LABEL_CATEGORY_KEYWORDS.keys())[0]
    return {
        "label": default,
        "confidence": 0.25,
        "category": _label_to_category(default),
        "source": "fallback",
        "top_predictions": [
            {"label": default, "confidence": 0.25, "category": _label_to_category(default)}
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


def predict_track(features: dict | Sequence[str], use_fallback: bool = True) -> Dict[str, object]:
    """Predict a career track from either a numeric feature dict or a
    sequence of selected feature names.

    If `use_fallback` is True (default), a simple rule-based predictor is
    returned when model artifacts are missing or inference fails. If
    `use_fallback` is False, exceptions from model loading/inference are
    propagated to the caller.
    """
    # If a dict is provided, treat its keys as selected feature names.
    selected_features: Sequence[str]
    if isinstance(features, dict):
        selected_features = list(features.keys())
    else:
        selected_features = list(features)

    try:
        bundle = _get_model_bundle()
        model_input = _selected_features_to_feature_row(selected_features, bundle.feature_columns)
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
        if use_fallback:
            return _rule_based_predict(selected_features)
        raise

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="MajorMatch Predictive Engine", page_icon="🎓")

CLEANING_MAP = {
    'Psycology': 'Psychology',
    'Bussiness Education': 'Business Education',
    'Bussiness': 'Business',
    'Asrtology': 'Astrology',
    'Engeeniering': 'Engineering',
    'Pharmisist': 'Pharmacist',
    'Travelling': 'Traveling',
    'Listening Music': 'Listening to Music'
}

MODEL_PATH = 'majormatch_rf_model.pkl'
ENCODER_PATH = 'majormatch_encoder.pkl'
FEATURES_PATH = 'majormatch_features.pkl'
TRAINING_DATA_PATH = 'stud_training.csv'

@st.cache_resource
def build_or_load_model():
    """Load the notebook-trained model artifacts or rebuild from the original training CSV."""
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, encoder, features
    except Exception:
        df_train_raw = pd.read_csv(TRAINING_DATA_PATH)
        df_cleaned = df_train_raw.rename(columns=CLEANING_MAP)
        X = df_cleaned.drop(columns=['Courses'])
        y = df_cleaned['Courses']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)

        features = list(X.columns)
        return model, label_encoder, features

rf_model, label_encoder, feature_columns = build_or_load_model()

st.title("🎓 MajorMatch Career Predictor")
st.write("Select the skills, hobbies, and academic areas that match your profile below:")

selected_interests = st.multiselect(
    "Search or select your primary competencies:",
    options=feature_columns,
    placeholder="e.g., Coding, Mathematics, Designing..."
)

if st.button("Analyze My Track"):
    if not selected_interests:
        st.warning("Please select at least one interest or skill attribute to evaluate your profile.")
    else:
        input_vector = np.zeros(len(feature_columns))
        for interest in selected_interests:
            target_index = feature_columns.index(interest)
            input_vector[target_index] = 1

        final_sample = input_vector.reshape(1, -1)
        encoded_prediction = rf_model.predict(final_sample)
        predicted_major = label_encoder.inverse_transform(encoded_prediction)[0]

        probabilities = rf_model.predict_proba(final_sample)[0]
        confidence_score = np.max(probabilities) * 100

        st.success(f"### 🎯 Recommended Track: {predicted_major}")
        st.metric(label="Model Match Confidence", value=f"{confidence_score:.1f}%")

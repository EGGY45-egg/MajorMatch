from dataclasses import dataclass

import api.predict as predict_api


@dataclass
class FakeBundle:
    model: object
    encoder: object
    feature_columns: list[str]


class FakeModel:
    def __init__(self, predicted_class: int, probabilities: list[float]):
        self.predicted_class = predicted_class
        self.probabilities = probabilities
        self.last_input = None

    def predict(self, rows):
        self.last_input = rows
        return [self.predicted_class]

    def predict_proba(self, rows):
        self.last_input = rows
        return [self.probabilities]


class FakeEncoder:
    def __init__(self, labels: list[str]):
        self.labels = labels

    def inverse_transform(self, values):
        return [self.labels[int(values[0])]]


def test_predict_track_uses_model_bundle_and_maps_major_to_track(monkeypatch):
    fake_model = FakeModel(0, [0.84, 0.16])
    fake_encoder = FakeEncoder(["B.Tech.-Computer Science and Engineering", "BVA- Bachelor of Visual Arts"])
    fake_bundle = FakeBundle(
        model=fake_model,
        encoder=fake_encoder,
        feature_columns=["Coding", "Mathematics", "Designing", "Computer Parts", "Researching", "Solving Puzzles"],
    )

    monkeypatch.setattr(predict_api, "_load_artifacts_from_disk", lambda: fake_bundle)
    predict_api._get_model_bundle.cache_clear()

    prediction = predict_api.predict_track({"coding": 9, "math": 4, "design": 2})

    assert prediction["label"] == "B.Tech.-Computer Science and Engineering"
    assert prediction["category"] == "Software Engineer"
    assert prediction["confidence"] == 0.84
    assert fake_model.last_input is not None
    assert fake_model.last_input.iloc[0]["Coding"] == 1


def test_predict_track_falls_back_when_model_loading_fails(monkeypatch):
    monkeypatch.setattr(predict_api, "_load_artifacts_from_disk", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    predict_api._get_model_bundle.cache_clear()

    prediction = predict_api.predict_track({"coding": 9, "math": 4, "design": 2})

    assert prediction["label"] == "Software Engineer"
    assert prediction["confidence"] == 0.9
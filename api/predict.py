def predict_track(features: dict):
    """Simple rule-based predictor stub. Replace with teammate's model later."""
    coding = features.get("coding", 0)
    math = features.get("math", 0)
    design = features.get("design", 0)

    if coding >= math and coding >= design:
        return "Software Engineer", coding / 10.0
    if math >= coding and math >= design:
        return "Data Scientist", math / 10.0
    return "Product Designer", design / 10.0

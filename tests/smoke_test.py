from app_logic import recommend_track, summarize_matches


def main():
    # Use a list of selected feature names (model feature keys) to drive
    # the predictor. This mirrors how the app now passes selected features
    # instead of a structured numeric profile.
    selected = ["programming", "data structures", "algorithms"]
    recommendation = recommend_track(selected)
    assert isinstance(recommendation.get("track"), (str, type(None)))

    print("Smoke test passed")
    print(recommendation)


if __name__ == "__main__":
    main()

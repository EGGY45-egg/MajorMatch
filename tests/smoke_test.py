from app_logic import ProfileInput, build_course_query, recommend_track, suggest_courses, summarize_matches


def main():
    profile = ProfileInput(coding=9, math=4, design=2)
    recommendation = recommend_track(profile)
    assert recommendation["track"] == "Software Engineer"
    assert recommendation["confidence"] > 0

    query = build_course_query(recommendation["track"])
    assert query

    results = suggest_courses(query, top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0

    summary = summarize_matches(results)
    assert summary

    print("Smoke test passed")
    print(recommendation)
    print(summary)


if __name__ == "__main__":
    main()

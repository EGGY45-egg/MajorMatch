import api.search as search_api


def test_semantic_search_returns_normalized_metadata(monkeypatch):
    monkeypatch.setattr(
        search_api,
        "search_courses",
        lambda query, top_k=5: [
            {
                "id": 7,
                "title": "Intro to Python",
                "description": "Learn Python basics",
                "score": 0.5,
                "pca_x": 1.2,
                "pca_y": -0.4,
                "umap_x": 0.3,
                "umap_y": 0.7,
                "tsne_x": -1.1,
                "tsne_y": 2.2,
                "embedding": [1.0, 2.0],
            }
        ],
    )

    results = search_api.semantic_search("python", top_k=3)

    assert len(results) == 1
    item = results[0]
    assert item["id"] == 7
    assert item["title"] == "Intro to Python"
    assert item["description"] == "Learn Python basics"
    assert item["score"] == 0.5
    assert item["score_normalized"] == 0.75
    assert item["source"] == "db"
    assert item["pca_x"] == 1.2
    assert item["tsne_y"] == 2.2

import api.jobs as jobs_api
from app_logic import suggest_career_context


def test_get_career_context_uses_adzuna_payload(monkeypatch):
    monkeypatch.setattr(jobs_api, "ADZUNA_APP_ID", "id")
    monkeypatch.setattr(jobs_api, "ADZUNA_APP_KEY", "key")
    monkeypatch.setattr(jobs_api, "_build_query_url", lambda track, location, results_per_page=10: "https://example.test/jobs")
    monkeypatch.setattr(
        jobs_api,
        "_fetch_json",
        lambda url: {
            "count": 42,
            "results": [
                {"title": "Junior Software Engineer", "salary_min": 65000, "salary_max": 90000, "company": {"display_name": "Alpha"}},
                {"title": "Backend Developer", "salary_min": 70000, "salary_max": 110000, "company": {"display_name": "Beta"}},
            ],
        },
    )

    context = jobs_api.get_career_context("Software Engineer", location="United States")

    assert context.available is True
    assert context.job_count == 42
    assert context.salary_min == 65000
    assert context.salary_max == 110000
    assert context.top_job_titles == ["Junior Software Engineer", "Backend Developer"]
    assert context.top_companies == ["Alpha", "Beta"]
    assert context.source == "Adzuna"


def test_get_career_context_missing_credentials(monkeypatch):
    monkeypatch.setattr(jobs_api, "ADZUNA_APP_ID", "")
    monkeypatch.setattr(jobs_api, "ADZUNA_APP_KEY", "")

    context = suggest_career_context("Data Scientist")

    assert context.available is False
    assert context.source == "Adzuna"
    assert "ADZUNA_APP_ID" in (context.note or "")


def test_build_query_url_uses_search_endpoint(monkeypatch):
    monkeypatch.setattr(jobs_api, "ADZUNA_BASE_URL", "https://api.adzuna.com/v1/api/jobs")
    monkeypatch.setattr(jobs_api, "ADZUNA_COUNTRY", "us")
    monkeypatch.setattr(jobs_api, "ADZUNA_APP_ID", "id")
    monkeypatch.setattr(jobs_api, "ADZUNA_APP_KEY", "key")

    url = jobs_api._build_query_url("Software Engineer", "United States")

    assert "/us/search/1?" in url
    assert "app_id=id" in url
    assert "app_key=key" in url

# MajorMatch Project Documentation

## 1. Problem Statement

MajorMatch solves the problem of choice paralysis and poor path alignment for students who need to decide between majors, career tracks, and supporting courses. The primary users are college students, incoming freshmen, and academic advisors.

The problem matters because handbook pages, course lists, and job descriptions are usually too dense for quick decision-making. Users need a workflow that can answer a question, suggest a direction, show market context, and surface relevant courses without forcing them to jump between separate tools.

This project supports the workflow of career-track recommendation, career-context review, and course exploration in one chat-first interface.

## 2. Source Documentation

### Source Name: CAPSTONE_PROJECT_GUIDE.md

- Purpose: this documentation is written to match the required capstone project structure.
- Direct link: not present in the current repository workspace.
- Coverage: project-level requirements for problem statement and source documentation.
- Time period covered: current implementation work recorded in [docs/IMPLEMENTATION.md](IMPLEMENTATION.md), mainly 2026-05-25 to 2026-05-26.

### Source Name: [docs/IMPLEMENTATION.md](IMPLEMENTATION.md)

- Purpose: chronological implementation log and edit record.
- Direct link: [docs/IMPLEMENTATION.md](IMPLEMENTATION.md)
- Coverage: all recorded code and documentation changes for MajorMatch.
- Time period covered: 2026-05-25 through 2026-05-26 in the current log snapshot.
- Preparation or cleaning steps: entries were kept chronological and duplicated details were summarized into one project-level document for easier review.
- Missing data, limits, or quality issues: the log records implementation decisions, but it does not by itself provide a rubric-style problem statement.

### Source Name: [README.md](README.md)

- Purpose: high-level repository overview and setup instructions.
- Direct link: [README.md](README.md)
- Coverage: app summary, environment variables, setup steps, and testing notes.
- Preparation or cleaning steps: condensed from implementation details into a shorter user-facing overview.
- Missing data, limits, or quality issues: does not fully explain the project problem statement or the source documentation fields required by the capstone guide.

### Source Name: [features.md](features.md)

 - Purpose: project features and architecture summary.
- Direct link: [features.md](features.md)
- Coverage: project title, stakeholder group, problem statement, and technical architecture summary.
- Time period covered: current project framing document.
- Preparation or cleaning steps: used as a reference for stakeholder and architecture wording.
- Missing data, limits, or quality issues: does not list file-by-file implementation history.

### Source Name: [userflow.md](userflow.md)

- Purpose: user journey and workflow reference.
- Direct link: [userflow.md](userflow.md)
- Coverage: landing flow, chat entry, prediction, career context, course exploration, and refinement loop.
- Preparation or cleaning steps: translated into the workflow wording used in the problem statement.
- Missing data, limits, or quality issues: does not specify datasets or cleaning details.

### Covered Files, Datasets, and Records

- App and orchestration files: [streamlit_app.py](../streamlit_app.py), [app_logic.py](../app_logic.py), [api/orchestrator.py](../api/orchestrator.py), [api/ollama.py](../api/ollama.py), [api/predict.py](../api/predict.py), [api/search.py](../api/search.py), [api/jobs.py](../api/jobs.py), [course_index.py](../course_index.py)
- Data files: [data/business_finance.csv](../data/business_finance.csv), [data/computer_science.csv](../data/computer_science.csv), [data/creative_arts.csv](../data/creative_arts.csv), [data/engineering_it.csv](../data/engineering_it.csv), [data/health_sciences.csv](../data/health_sciences.csv), [data/social_sciences.csv](../data/social_sciences.csv)
- Model data: [ml_model/stud_training.csv](../ml_model/stud_training.csv), [ml_model/stud_testing.csv](../ml_model/stud_testing.csv)
- Test records: [tests/](../tests/)

### Time Period Covered

- Implementation log snapshot used for this documentation: 2026-05-25 to 2026-05-26.
- The source materials describe the current project state rather than a historical dataset with a fixed academic term.

### Preparation or Cleaning Steps

- Course CSVs are indexed from the `data/` folder and expected to contain `title` and `description` columns.
- Missing course rows without those fields are skipped during indexing.
- Search supports fallback behavior when `pgvector` or other optional dependencies are unavailable.
- Prediction data is loaded from the local ML CSV files and mapped into the app's track and label flow.

### Missing Data, Limits, or Quality Issues

- The repository does not currently include a checked-in `CAPSTONE_PROJECT_GUIDE.md` file.
- Career-context data depends on live Adzuna credentials and may fall back when credentials are missing.
- Semantic search quality depends on the completeness and consistency of the course CSV descriptions.
- Embedding and visualization quality may vary when fallback paths are used instead of full database vector support.

## 3. Short Project Summary

MajorMatch is a chat-first system that combines prediction, job context, and semantic course search so users can move from a question to a practical next step in one interface.
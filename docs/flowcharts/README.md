# Flowchart Showcase

This folder contains the rendered diagrams used to explain MajorMatch during the presentation.

## What to show

- [Architecture overview](rendered/architecture/full_overview.png): system components and how they connect.
- [Prediction architecture](rendered/architecture/architecture_predict.png): trained model, feature handling, and fallback path.
- [Semantic search architecture](rendered/architecture/architecture_semantic_search.png): embeddings, storage, ranking, and projection setup.
- [Adzuna job context architecture](rendered/architecture/architecture_jobs.png): query building, request flow, and parsed output.

## Execution flow diagrams

- [Orchestrator execution flow](rendered/execution/orchestrator_overall.png): how the app decides whether to call tools.
- [Semantic search execution flow](rendered/execution/execute_semantic_search.png): how a query is handled from input to ranked results.
- [Career context execution flow](rendered/execution/get_career_context.png): how the app requests and formats Adzuna results.
- [Prediction execution flow](rendered/execution/execute_prediction.png): how the app handles prediction requests, including feature selection and fallback.

## Use in slides

Use the PNG files under `rendered/` when you need presentation-ready screenshots, and keep the `.mmd` files only if you want to edit the diagrams later.
import plotly.express as px
import streamlit as st

from app_logic import (
    default_profile_values,
    detect_tool_intents,
    summarize_matches,
)
from api.ollama import chat_completion, ollama_is_available, resolve_chat_model
from api.orchestrator import OrchestratorResult, run_orchestrated_assistant
from api.predict import get_prediction_feature_columns, predict_track
from api.search import CourseIndexError, rebuild_index


def _render_tool_prediction(prediction):
    st.markdown("### Career Track Prediction")
    st.success(f"Predicted major: {prediction.get('track') or prediction.get('label') or 'Unknown'}")
    if prediction.get("category"):
        st.caption(f"Career family: {prediction['category']}")
    st.progress(min(float(prediction["confidence"]), 1.0))
    st.caption(f"Confidence score: {prediction['confidence']:.2f}")

    top_predictions = prediction.get("top_predictions") or []
    if top_predictions:
        st.markdown("Top 3 predictions")
        for index, item in enumerate(top_predictions[:3], start=1):
            label = item.get("label") or "Unknown"
            confidence = item.get("confidence")
            category = item.get("category")
            details = f"{index}. {label} ({float(confidence):.2f})" if confidence is not None else f"{index}. {label}"
            if category:
                details += f" - {category}"
            st.write(details)


def _render_prediction_tool():
    st.markdown("### Prediction Tool")
    st.caption("Choose the interests that fit you best, then run the model to get the raw major label.")

    try:
        feature_columns = get_prediction_feature_columns()
    except Exception as error:
        st.error(f"Could not load prediction features: {error}")
        return

    selected_interests = st.multiselect(
        "What are you good at?",
        options=feature_columns,
        key="prediction_selected_interests",
        placeholder="Search and select your strongest interests...",
    )

    if st.button("Analyze My Track", key="prediction_analyze_button"):
        if not selected_interests:
            st.warning("Please select at least one interest before running the prediction.")
        else:
            prediction = predict_track(selected_interests)
            if isinstance(prediction, dict):
                _render_tool_prediction(prediction)
            else:
                label, confidence = prediction
                _render_tool_prediction({"label": label, "confidence": confidence, "category": None})


def _render_tool_career_context(career_context):
    st.markdown("### Career Context")
    available = bool(career_context.get("available"))
    if available:
        job_count = career_context.get("job_count")
        salary_min = career_context.get("salary_min")
        salary_max = career_context.get("salary_max")
        salary_currency = career_context.get("salary_currency") or "USD"
        left_metric, right_metric = st.columns(2)
        with left_metric:
            st.metric("Job count", f"{job_count:,}" if job_count is not None else "N/A")
        with right_metric:
            if salary_min is not None and salary_max is not None:
                st.metric(
                    f"Salary range ({salary_currency})",
                    f"{salary_min:,} - {salary_max:,}",
                )
            else:
                st.metric(f"Salary range ({salary_currency})", "N/A")

        if career_context.get("top_job_titles"):
            st.caption("Top job titles: " + ", ".join(career_context.get("top_job_titles", [])))
        if career_context.get("top_companies"):
            st.caption("Top companies: " + ", ".join(career_context.get("top_companies", [])))
    else:
        st.info(career_context.get("note") or "Job-market context is not available yet.")


def _render_semantic_search(search_artifact):
    query = search_artifact.get("query") or ""
    results = search_artifact.get("results") or []
    projection = search_artifact.get("projection") or {}

    st.markdown("### Semantic Search")
    if query:
        st.caption(f"Query: {query}")
    st.info(summarize_matches(results))

    projection_methods = projection.get("methods") or {}
    selected_method = str(projection.get("method") or "pca").lower()
    method_labels = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}

    available_methods = [method for method in ("pca", "umap", "tsne") if projection_methods.get(method, {}).get("available")]
    if projection.get("available") and available_methods:
        default_index = available_methods.index(selected_method) if selected_method in available_methods else 0
        chosen_method = st.selectbox(
            "Projection method",
            available_methods,
            index=default_index,
            format_func=lambda method: method_labels.get(method, method.upper()),
            key="semantic_projection_method",
        )
        chosen_projection = projection_methods.get(chosen_method) or projection
        course_points = chosen_projection.get("courses") or []
        query_point = chosen_projection.get("query_point")
        if course_points and query_point:
            top_title = results[0].get("title") if results else None
            frame = {
                "title": [point["title"] for point in course_points] + [query_point["title"]],
                "description": [point["description"] for point in course_points] + [query_point["description"]],
                "x": [point["x"] for point in course_points] + [query_point["x"]],
                "y": [point["y"] for point in course_points] + [query_point["y"]],
                "kind": [
                    "top_match" if top_title and point["title"] == top_title else "course"
                    for point in course_points
                ]
                + ["query"],
            }

            figure = px.scatter(
                frame,
                x="x",
                y="y",
                color="kind",
                color_discrete_map={"course": "blue", "top_match": "red", "query": "green"},
                hover_name="title",
                hover_data={"description": True, "x": False, "y": False, "kind": True},
                title=f"{method_labels.get(chosen_method, chosen_method.upper())} projection of semantic search results",
            )
            figure.update_traces(marker=dict(size=11, opacity=0.85))
            figure.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Semantic search returned results, but the projection map is not available yet.")
    elif projection.get("available"):
        method = str(projection.get("method") or "pca").upper()
        course_points = projection.get("courses") or []
        query_point = projection.get("query_point")
        if course_points and query_point:
            top_title = results[0].get("title") if results else None
            frame = {
                "title": [point["title"] for point in course_points] + [query_point["title"]],
                "description": [point["description"] for point in course_points] + [query_point["description"]],
                "x": [point["x"] for point in course_points] + [query_point["x"]],
                "y": [point["y"] for point in course_points] + [query_point["y"]],
                "kind": [
                    "top_match" if top_title and point["title"] == top_title else "course"
                    for point in course_points
                ]
                + ["query"],
            }

            figure = px.scatter(
                frame,
                x="x",
                y="y",
                color="kind",
                color_discrete_map={"course": "blue", "top_match": "red", "query": "green"},
                hover_name="title",
                hover_data={"description": True, "x": False, "y": False, "kind": True},
                title=f"{method} projection of semantic search results",
            )
            figure.update_traces(marker=dict(size=11, opacity=0.85))
            figure.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("Semantic search returned results, but the projection map is not available yet.")
    elif projection.get("error"):
        st.info(f"Visualization unavailable: {projection['error']}")

    for course in results[:5]:
        with st.container(border=True):
            st.markdown(f"**{course.get('title', 'Untitled')}**")
            st.write(course.get("description", "No description available."))


def _render_tool_output(result: OrchestratorResult):
    if result.artifacts.get("prediction"):
        _render_tool_prediction(result.artifacts["prediction"])

    if result.artifacts.get("career_context"):
        _render_tool_career_context(result.artifacts["career_context"])

    if result.artifacts.get("semantic_search"):
        _render_semantic_search(result.artifacts["semantic_search"])


def main():
    st.set_page_config(page_title="MajorMatch", layout="wide")
    st.title("MajorMatch")
    st.caption("A semantic course and career pathfinder for students, advisors, and freshmen.")

    st.write(
        "The assistant chooses tools automatically when they are relevant and keeps the reply grounded in the result."
    )

    col_status_left, col_status_right = st.columns([1, 1])
    with col_status_left:
        ollama_ready = ollama_is_available()
        if ollama_ready:
            st.success("Ollama is available locally.")
        else:
            st.error("Ollama is not reachable. Start Ollama before using chat.")
    with col_status_right:
        resolved_model = resolve_chat_model()
        st.caption(f"Chat model in use: {resolved_model}")

    with st.expander("Maintenance", expanded=False):
        if st.button("Build / refresh course index from CSV"):
            try:
                with st.spinner("Embedding courses and writing them to PostgreSQL..."):
                    indexed_count = rebuild_index()
                st.success(f"Indexed {indexed_count} courses.")
            except CourseIndexError as error:
                st.error(str(error))
            except Exception as error:
                st.exception(error)

    st.subheader("Chat Assistant")
    st.caption("Ask naturally. Normal questions get a direct friendly reply; tools are used only when needed.")

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "content": "I am MajorMatch, an AI assistant that can help you decide what course in college or career to take. Ask me a question and I will answer directly unless a tool is useful.",
            }
        ]
    if "assistant_profile" not in st.session_state:
        st.session_state["assistant_profile"] = default_profile_values(5)
    if "assistant_tools_state" not in st.session_state:
        st.session_state["assistant_tools_state"] = {}
    if "prediction_tool_open" not in st.session_state:
        st.session_state["prediction_tool_open"] = False

    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.get("prediction_tool_open"):
        st.divider()
        _render_prediction_tool()

    user_message = st.chat_input("Ask a question about majors, careers, courses, or maps...")
    if user_message:
        st.session_state["assistant_messages"].append({"role": "user", "content": user_message})

        intents = detect_tool_intents(user_message)
        if "career_track" in intents and not any(intent in intents for intent in ("course_search", "career_context", "visualization")):
            st.session_state["prediction_tool_open"] = True
            st.session_state["assistant_messages"].append(
                {
                    "role": "assistant",
                    "content": "I opened the prediction tool below. Pick the interests that match you best, then run the analyzer.",
                }
            )
            st.rerun()

        try:
            profile_values = st.session_state["assistant_profile"]
            result = run_orchestrated_assistant(
                user_message,
                profile_values,
                location="United States",
                model=resolved_model,
                conversation_history=st.session_state["assistant_messages"],
            )

            st.session_state["assistant_profile"] = result.profile
            st.session_state["assistant_tools_state"] = result.artifacts

            assistant_reply = result.reply or "I am MajorMatch, an AI assistant that can help you decide what course in college or career to take."
            st.session_state["assistant_messages"].append({"role": "assistant", "content": assistant_reply})
        except Exception as error:
            st.session_state["assistant_messages"].append(
                {
                    "role": "assistant",
                    "content": f"I hit an issue while processing that request: {error}",
                }
            )
        st.rerun()

    tools_state = st.session_state.get("assistant_tools_state", {})
    if tools_state:
        st.divider()
        st.caption("Latest tool output")
        result = OrchestratorResult(
            reply="",
            profile=st.session_state["assistant_profile"],
            artifacts=tools_state,
            tool_trace=[],
            raw="",
        )
        _render_tool_output(result)


if __name__ == "__main__":
    main()

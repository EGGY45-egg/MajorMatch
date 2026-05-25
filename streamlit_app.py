import plotly.express as px
import streamlit as st

from app_logic import (
    default_profile_values,
    summarize_matches,
)
from api.ollama import chat_completion, ollama_is_available, resolve_chat_model
from api.orchestrator import OrchestratorResult, run_orchestrated_assistant
from api.search import CourseIndexError, rebuild_index


def _render_tool_prediction(prediction):
    st.markdown("### Career Track Prediction")
    st.success(f"Predicted track: {prediction['track']}")
    st.progress(min(float(prediction["confidence"]), 1.0))
    st.caption(f"Confidence score: {prediction['confidence']:.2f}")


def _render_tool_career_context(career_context):
    st.markdown("### Career Context")
    if career_context.available:
        left_metric, right_metric = st.columns(2)
        with left_metric:
            st.metric("Job count", f"{career_context.job_count:,}" if career_context.job_count is not None else "N/A")
        with right_metric:
            if career_context.salary_min is not None and career_context.salary_max is not None:
                st.metric(
                    f"Salary range ({career_context.salary_currency})",
                    f"{career_context.salary_min:,} - {career_context.salary_max:,}",
                )
            else:
                st.metric(f"Salary range ({career_context.salary_currency})", "N/A")

        if career_context.top_job_titles:
            st.caption("Top job titles: " + ", ".join(career_context.top_job_titles))
        if career_context.top_companies:
            st.caption("Top companies: " + ", ".join(career_context.top_companies))
    else:
        st.info(career_context.note or "Job-market context is not available yet.")


def _render_semantic_search(search_artifact):
    query = search_artifact.get("query") or ""
    results = search_artifact.get("results") or []
    projection = search_artifact.get("projection") or {}

    st.markdown("### Semantic Search")
    if query:
        st.caption(f"Query: {query}")
    st.info(summarize_matches(results))

    if projection.get("available"):
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

    for course in results:
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
    st.caption("Ask naturally. The assistant will decide when to use tools.")

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "content": "Hi, I am MajorMatch. Ask me anything about careers, salary context, or courses, and I will use tools when needed.",
            }
        ]
    if "assistant_profile" not in st.session_state:
        st.session_state["assistant_profile"] = default_profile_values(5)
    if "assistant_tools_state" not in st.session_state:
        st.session_state["assistant_tools_state"] = {}

    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input("Ask anything about your major, careers, courses, or maps...")
    if user_message:
        st.session_state["assistant_messages"].append({"role": "user", "content": user_message})

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

            assistant_reply = result.reply or "I can help with career recommendations, job context, and course exploration."
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

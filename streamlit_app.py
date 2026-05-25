import plotly.express as px
import streamlit as st

from app_logic import (
    ProfileInput,
    build_search_query_from_message,
    default_profile_values,
    recommend_track,
    suggest_career_context,
    suggest_courses,
    summarize_matches,
)
from api.ollama import chat_completion, ollama_is_available, resolve_chat_model
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


def _render_tool_courses(courses):
    st.markdown("### Course Exploration")
    st.info(summarize_matches(courses))
    for course in courses:
        with st.container(border=True):
            st.markdown(f"**{course.get('title', 'Untitled')}**")
            st.write(course.get("description", "No description available."))


def _render_tool_visualization(query, courses):
    st.markdown("### Visual Exploration")
    projection_method = st.radio("Projection method", ["pca", "umap", "tsne"], horizontal=True, key="tool_projection_method")
    try:
        from course_index import project_courses_with_query

        course_points, query_point = project_courses_with_query(query, method=projection_method)
        if course_points and query_point is not None:
            top_title = courses[0].get("title") if courses else None
            frame = {
                "title": [p.title for p in course_points] + [query_point.title],
                "description": [p.description for p in course_points] + [query_point.description],
                "x": [p.x for p in course_points] + [query_point.x],
                "y": [p.y for p in course_points] + [query_point.y],
                "kind": [
                    "top_match" if top_title and p.title == top_title else "course"
                    for p in course_points
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
                title=f"{projection_method.upper()} projection of course embeddings (query shown)",
            )
            figure.update_traces(marker=dict(size=11, opacity=0.85))
            figure.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No projection points found yet. Build the index first.")
    except CourseIndexError as error:
        st.error(str(error))
    except Exception as error:
        st.exception(error)


def _tool_label(tool_key: str) -> str:
    labels = {
        "none": "Plain chat",
        "career_track": "Career recommendation",
        "career_context": "Career context",
        "semantic_search": "Semantic search",
    }
    return labels.get(tool_key, "Plain chat")


def _compose_tool_reply(user_message: str, tool_choice: str, tools_state: dict, conversation: list, resolved_model: str) -> str:
    if tool_choice == "career_track":
        prediction = tools_state.get("prediction") or {}
        prompt = (
            f"Tool result: career_track -> track={prediction.get('track', 'unknown')}, "
            f"confidence={float(prediction.get('confidence', 0.0)):.2f}. "
            "Write one short chat response grounded only in this result."
        )
    elif tool_choice == "career_context":
        career_context = tools_state.get("career_context")
        prompt = (
            "Tool result: career_context -> "
            f"available={getattr(career_context, 'available', False)}, "
            f"track={getattr(career_context, 'track', 'unknown')}, "
            f"job_count={getattr(career_context, 'job_count', None)}, "
            f"salary_min={getattr(career_context, 'salary_min', None)}, "
            f"salary_max={getattr(career_context, 'salary_max', None)}, "
            f"top_job_titles={getattr(career_context, 'top_job_titles', [])}. "
            "Write one short chat response grounded only in this result."
        )
    elif tool_choice == "semantic_search":
        courses = tools_state.get("courses") or []
        top_titles = [course.get("title", "Untitled") for course in courses[:3]]
        query = tools_state.get("query") or user_message
        prompt = (
            "Tool result: semantic_search -> "
            f"query='{query}', top_matches={top_titles}, visualization=available. "
            "Write one short chat response that summarizes the matches and mentions the map if relevant."
        )
    else:
        prompt = ""

    if not prompt:
        return ""

    response = chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are MajorMatch. Keep the reply short, conversational, and grounded only in the tool result. "
                    "Do not invent extra findings."
                ),
            },
            *conversation,
            {"role": "user", "content": prompt},
        ],
        model=resolved_model,
        options={"temperature": 0.2},
    )
    return str(response.get("message", {}).get("content", "")).strip()


def main():
    st.set_page_config(page_title="MajorMatch", layout="wide")
    st.title("MajorMatch")
    st.caption("A semantic course and career pathfinder for students, advisors, and freshmen.")

    st.write(
        "Chat first. The assistant only shows tool results when your message clearly asks for career prediction, job-market context, course exploration, or visualization."
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
    st.caption("Choose a tool first, or leave it on plain chat.")

    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "content": "Hi, I am MajorMatch. Ask for a career recommendation, salary/job context, course suggestions, or a learning map.",
            }
        ]
    if "assistant_profile" not in st.session_state:
        st.session_state["assistant_profile"] = default_profile_values(5)
    if "assistant_tools_state" not in st.session_state:
        st.session_state["assistant_tools_state"] = {}
    if "assistant_selected_tool" not in st.session_state:
        st.session_state["assistant_selected_tool"] = "none"

    tool_options = ["none", "career_track", "career_context", "semantic_search"]
    tool_choice = st.selectbox(
        "Tool mode",
        tool_options,
        index=tool_options.index(st.session_state["assistant_selected_tool"])
        if st.session_state["assistant_selected_tool"] in tool_options
        else 0,
        format_func=_tool_label,
        key="assistant_selected_tool",
    )
    st.caption(f"Selected tool: {_tool_label(tool_choice)}")

    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input("Ask anything about your major, careers, courses, or maps...")
    if user_message:
        st.session_state["assistant_messages"].append({"role": "user", "content": user_message})
        tools_state = {
            "used_tools": [],
            "prediction": None,
            "career_context": None,
            "courses": None,
            "visualization": None,
            "query": None,
        }

        try:
            profile_values = st.session_state["assistant_profile"]
            profile = ProfileInput(
                coding=profile_values["coding"],
                math=profile_values["math"],
                design=profile_values["design"],
            )

            conversation = [
                {
                    "role": "system",
                    "content": (
                        "You are MajorMatch, a concise student assistant. Keep responses short and practical. "
                        "When a tool is selected, answer using the tool result and keep the response grounded in that result."
                    ),
                },
                *st.session_state["assistant_messages"],
            ]
            recommendation = None
            if tool_choice in ("career_track", "career_context", "semantic_search"):
                recommendation = recommend_track(profile)
                tools_state["prediction"] = recommendation
                tools_state["used_tools"].append("career_track")

            if tool_choice == "career_context":
                if recommendation is None:
                    recommendation = recommend_track(profile)
                    tools_state["prediction"] = recommendation
                    if "career_track" not in tools_state["used_tools"]:
                        tools_state["used_tools"].append("career_track")
                career_context = suggest_career_context(str(recommendation["track"]))
                tools_state["career_context"] = career_context
                tools_state["used_tools"].append("career_context")

            if tool_choice == "semantic_search":
                if recommendation is None:
                    recommendation = recommend_track(profile)
                    tools_state["prediction"] = recommendation
                    if "career_track" not in tools_state["used_tools"]:
                        tools_state["used_tools"].append("career_track")
                query = build_search_query_from_message(user_message, fallback_track=str(recommendation["track"]))
                tools_state["query"] = query
                courses = suggest_courses(query, top_k=5)
                tools_state["courses"] = courses
                tools_state["used_tools"].append("course_search")
                tools_state["visualization"] = {"enabled": True}
                tools_state["used_tools"].append("visualization")

            st.session_state["assistant_tools_state"] = tools_state

            if tool_choice == "none":
                llm_response = chat_completion(conversation, model=resolved_model, options={"temperature": 0.2})
                assistant_reply = str(llm_response.get("message", {}).get("content", "")).strip()
            else:
                assistant_reply = _compose_tool_reply(user_message, tool_choice, tools_state, conversation, resolved_model)

            if not assistant_reply:
                assistant_reply = "I can help with career recommendations, job context, course search, and visual maps."
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
    used_tools = tools_state.get("used_tools", [])

    if used_tools:
        st.divider()
        st.caption("Latest tool results")

        prediction = tools_state.get("prediction")
        if prediction and "career_track" in used_tools:
            _render_tool_prediction(prediction)

        career_context = tools_state.get("career_context")
        if career_context and "career_context" in used_tools:
            _render_tool_career_context(career_context)

        courses = tools_state.get("courses")
        if courses and "course_search" in used_tools:
            _render_tool_courses(courses)

        if "visualization" in used_tools:
            _render_tool_visualization(tools_state.get("query") or "", tools_state.get("courses") or [])

        st.caption("Change the tool picker if you want a different result on the next message.")


if __name__ == "__main__":
    main()

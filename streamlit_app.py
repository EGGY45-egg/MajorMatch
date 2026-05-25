import streamlit as st
import plotly.express as px

from app_logic import (
    coerce_profile_values,
    default_profile_values,
    ProfileInput,
    build_course_query,
    missing_profile_fields,
    profile_from_sliders,
    profile_is_empty,
    profile_to_text,
    recommend_track,
    suggest_career_context,
    suggest_courses,
    summarize_matches,
)
from api.ollama import interview_profile, ollama_is_available, resolve_chat_model
from api.search import CourseIndexError, get_course_projection_points, rebuild_index

def main():
    st.set_page_config(page_title="MajorMatch", layout="wide")
    st.title("MajorMatch")
    st.caption("A semantic course and career pathfinder for students, advisors, and freshmen.")

    col_intro, col_status = st.columns([2, 1])
    with col_intro:
        st.subheader("Explore a direction, then see matching courses")
        st.write(
            "Start with your skills, get a track recommendation, then use that recommendation to discover relevant courses."
        )
    with col_status:
        from course_index import get_embedding_model_name, server_supports_vector

        model_name = get_embedding_model_name()
        pgvector_available = server_supports_vector()

        st.metric("Embedding model", model_name)
        if pgvector_available:
            st.success("pgvector available — SQL vector ops will be used when enabled")
        else:
            st.info("pgvector not available — using in-Python similarity fallback")

    ollama_ready = ollama_is_available()
    resolved_model = resolve_chat_model()
    if ollama_ready:
        st.info("Ollama is available locally and can interview the user.")
    else:
        st.error("Ollama is not reachable right now. Start Ollama before using the chat flow; errors will surface directly for debugging.")
    st.caption(f"Chat model in use: {resolved_model}")

    st.divider()
    index_controls, index_status = st.columns([1, 1])
    with index_controls:
        if st.button("Build / refresh course index from CSV"):
            try:
                with st.spinner("Embedding courses and writing them to PostgreSQL + pgvector..."):
                    indexed_count = rebuild_index()
                st.success(f"Indexed {indexed_count} courses.")
            except CourseIndexError as error:
                st.error(str(error))
            except Exception as error:
                st.exception(error)
    with index_status:
        st.caption("The index must be built once before semantic search and projections can render.")

    st.divider()

    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("1. Talk to Ollama")

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Tell me what you are strongest at so I can build your profile. You can answer naturally, like 'I am best at coding and okay with math'.",
                }
            ]
        if "chat_profile" not in st.session_state:
            st.session_state["chat_profile"] = default_profile_values()
        if "chat_complete" not in st.session_state:
            st.session_state["chat_complete"] = False
        if "debug_profile" not in st.session_state:
            st.session_state["debug_profile"] = default_profile_values(5)
        if "prefer_debug_sliders" not in st.session_state:
            st.session_state["prefer_debug_sliders"] = False

        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_message = st.chat_input("Talk to Ollama about your strengths, interests, or weaknesses...")
        if user_message:
            st.session_state["chat_messages"].append({"role": "user", "content": user_message})
            result = interview_profile(
                st.session_state["chat_messages"],
                st.session_state["chat_profile"],
            )
            st.session_state["chat_profile"] = coerce_profile_values(result.profile)
            st.session_state["chat_complete"] = bool(result.complete)
            st.session_state["chat_messages"].append({"role": "assistant", "content": result.reply})
            st.rerun()

        active_chat_profile = coerce_profile_values(st.session_state["chat_profile"])
        st.success("Chat profile built" if not profile_is_empty(active_chat_profile) else "Chat profile not built yet")
        st.json(active_chat_profile)

        with st.expander("Debug sliders", expanded=False):
            st.checkbox(
                "Prefer debug sliders for the active recommendation",
                key="prefer_debug_sliders",
            )
            with st.form("debug_profile_form", clear_on_submit=False):
                coding = st.slider("Coding", 0, 10, st.session_state["debug_profile"]["coding"])
                math = st.slider("Math", 0, 10, st.session_state["debug_profile"]["math"])
                design = st.slider("Design", 0, 10, st.session_state["debug_profile"]["design"])
                debug_submitted = st.form_submit_button("Save debug profile")

            if debug_submitted:
                st.session_state["debug_profile"] = profile_from_sliders(coding, math, design)
                st.toast("Debug profile saved")

            if st.button("Reset Ollama chat"):
                st.session_state["chat_messages"] = [
                    {
                        "role": "assistant",
                        "content": "Tell me what you are strongest at so I can build your profile. You can answer naturally, like 'I am best at coding and okay with math'.",
                    }
                ]
                st.session_state["chat_profile"] = default_profile_values()
                st.session_state["chat_complete"] = False
                st.rerun()

        active_profile = (
            st.session_state["debug_profile"]
            if st.session_state["prefer_debug_sliders"]
            else active_chat_profile
        )
        if profile_is_empty(active_profile):
            active_profile = st.session_state["debug_profile"]

        profile = ProfileInput(
            coding=active_profile["coding"],
            math=active_profile["math"],
            design=active_profile["design"],
        )
        recommendation = recommend_track(profile)

        st.subheader("Current structured profile")
        st.caption(profile_to_text(active_profile))
        if missing_profile_fields(active_profile):
            st.caption(f"Still missing: {', '.join(missing_profile_fields(active_profile))}")

        st.success(f"Predicted track: {recommendation['track']}")
        st.progress(min(float(recommendation["confidence"]), 1.0))
        st.caption(f"Confidence score: {recommendation['confidence']:.2f}")

        st.subheader("Career context")
        career_context = suggest_career_context(str(recommendation["track"]))
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
            if career_context.query_url:
                st.caption(f"Source: {career_context.source} | Query URL prepared")
        else:
            st.info(career_context.note or "Job-market context is not available yet.")

        st.subheader("Suggested course search")
        auto_query = build_course_query(str(recommendation["track"]))
        query = st.text_input("Search courses", value=auto_query)

        try:
            search_results = suggest_courses(query, top_k=5)
            st.session_state["last_query"] = query
            st.session_state["last_results"] = search_results
        except CourseIndexError as error:
            st.error(str(error))
            st.session_state["last_results"] = []
        except Exception as error:
            st.exception(error)
            st.session_state["last_results"] = []

    with right:
        st.subheader("2. What to explore next")
        results = st.session_state.get("last_results", [])

        st.info(summarize_matches(results))

        if results:
            for course in results:
                with st.container(border=True):
                    st.markdown(f"**{course['title']}**")
                    st.write(course["description"])
        else:
            st.warning("No courses are available yet. Check the corpus or try a broader query.")

        st.subheader("Course projection map")
        projection_method = st.radio("Projection method", ["pca", "umap", "tsne"], horizontal=True)
        try:
            from course_index import project_courses_with_query

            # Compute projection coordinates for all courses together with the current query
            course_points, query_point = project_courses_with_query(query, method=projection_method)
            if course_points:
                # Determine top match title if available
                top_title = None
                last_results = st.session_state.get("last_results", [])
                if last_results:
                    top_title = last_results[0].get("title")

                frame = {
                    "title": [p.title for p in course_points] + [query_point.title],
                    "description": [p.description for p in course_points] + [query_point.description],
                    "x": [p.x for p in course_points] + [query_point.x],
                    "y": [p.y for p in course_points] + [query_point.y],
                    "kind": ["course" for _ in course_points] + ["query"],
                }

                # Mark top match
                kinds = []
                for p in course_points:
                    if top_title and p.title == top_title:
                        kinds.append("top_match")
                    else:
                        kinds.append("course")
                kinds.append("query")
                frame["kind"] = kinds

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
                # Make query marker larger
                # Plotly express doesn't allow per-point sizes easily without a column; use marker updates by filtering traces
                figure.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(figure, use_container_width=True)
            else:
                st.info("No projection points found yet. Build the index first.")
        except CourseIndexError as error:
            st.error(str(error))
        except Exception as error:
            st.exception(error)

        st.subheader("Demo flow")
        st.write(
            "1. Adjust the profile sliders. 2. Get a track recommendation. 3. Review the course matches on the right."
        )

    st.divider()
    st.subheader("How this is testable")
    st.write(
        "The core recommendation and search helpers live outside Streamlit in `app_logic.py`, and the Ollama adapter is isolated in `api/ollama.py`, so the flow can be smoke-tested without a full browser session."
    )

if __name__ == '__main__':
    main()

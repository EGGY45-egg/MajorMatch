import streamlit as st

from app_logic import (
    ProfileInput,
    build_course_query,
    recommend_track,
    suggest_courses,
    summarize_matches,
)

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
        st.metric("Current mode", "Demo MVP")
        st.write("Everything here is wired for a clean first demo and easy testing.")

    st.divider()

    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("1. Tell us about yourself")
        with st.form("profile_form", clear_on_submit=False):
            coding = st.slider("Coding", 0, 10, 5)
            math = st.slider("Math", 0, 10, 5)
            design = st.slider("Design", 0, 10, 5)
            submitted = st.form_submit_button("Get recommendation")

        profile = ProfileInput(coding=coding, math=math, design=design)
        recommendation = recommend_track(profile)

        if submitted or "last_recommendation" not in st.session_state:
            st.session_state["last_recommendation"] = recommendation
            st.session_state["last_profile"] = profile

        current = st.session_state["last_recommendation"]
        st.success(f"Predicted track: {current['track']}")
        st.progress(min(float(current["confidence"]), 1.0))
        st.caption(f"Confidence score: {current['confidence']:.2f}")

        st.subheader("Suggested course search")
        auto_query = build_course_query(str(current["track"]))
        query = st.text_input("Search courses", value=auto_query)

        search_results = suggest_courses(query, top_k=5)
        st.session_state["last_query"] = query
        st.session_state["last_results"] = search_results

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

        st.subheader("Demo flow")
        st.write(
            "1. Adjust the profile sliders. 2. Get a track recommendation. 3. Review the course matches on the right."
        )

    st.divider()
    st.subheader("How this is testable")
    st.write(
        "The core recommendation and search helpers live outside Streamlit in `app_logic.py`, so they can be smoke-tested without launching the UI."
    )

if __name__ == '__main__':
    main()

import streamlit as st
from api.predict import predict_track
from api.search import semantic_search, load_courses

def main():
    st.set_page_config(page_title="MajorMatch", layout="wide")
    st.title("MajorMatch — Semantic Course & Career Pathfinder")

    st.sidebar.header("Profile")
    coding = st.sidebar.slider("Coding", 0, 10, 5)
    math = st.sidebar.slider("Math", 0, 10, 5)
    design = st.sidebar.slider("Design", 0, 10, 5)

    if st.sidebar.button("Predict career track"):
        track, score = predict_track({"coding": coding, "math": math, "design": design})
        st.success(f"Predicted track: {track} — score {score:.2f}")

    st.header("Semantic course search")
    query = st.text_input("Search course descriptions (natural language)")

    courses = load_courses()
    if query:
        results = semantic_search(query, courses)
        if results:
            for r in results:
                st.markdown(f"**{r['title']}** {r['description']}")
        else:
            st.info("No matches found; try different keywords.")
    else:
        st.info("Enter a query to search the curated course corpus.")

if __name__ == '__main__':
    main()

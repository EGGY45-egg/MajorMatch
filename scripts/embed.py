"""Build the PostgreSQL + pgvector course index from the CSV corpus."""

from course_index import rebuild_course_index


def main():
    count = rebuild_course_index()
    print(f"Indexed {count} courses into PostgreSQL + pgvector.")


if __name__ == "__main__":
    main()

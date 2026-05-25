"""Build the PostgreSQL + pgvector course index from the CSV corpus.

This script lives in `scripts/` but imports modules from the project root.
Ensure the project root is on `sys.path` so `course_index` can be imported
when invoked as a script (e.g. `python scripts/embed.py`).
"""

from pathlib import Path
import sys

# Add project root to sys.path so sibling modules are importable when running
# this script from the repo root or from any working directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from course_index import rebuild_course_index


def main():
    count = rebuild_course_index()
    print(f"Indexed {count} courses into PostgreSQL + pgvector.")


if __name__ == "__main__":
    main()

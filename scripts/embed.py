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

from course_index import rebuild_course_index, DEFAULT_COURSE_CSV


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    # If the user passed a path, use it; otherwise use the data directory.
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = data_dir

    # Informative message
    if target.is_dir():
        csvs = sorted(target.glob("*.csv"))
        print(f"Indexing CSV files in: {target} ({len(csvs)} files found)")
    else:
        print(f"Indexing single CSV file: {target}")

    count = rebuild_course_index(csv_path=target)
    if count:
        print(f"Indexed {count} courses into PostgreSQL + pgvector.")
    else:
        print("Indexed 0 courses. Check that CSV files exist and have 'title' and 'description' columns.")


if __name__ == "__main__":
    main()

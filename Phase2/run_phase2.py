# Phase2/run_phase2.py

import subprocess
import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = BASE_DIR / "src/eval/run_eval.py"
SCORE_SCRIPT = BASE_DIR / "src/eval/score_eval.py"
EXTRACT_SCRIPT = BASE_DIR / "extract_for_grading.py"

EVAL_RUN_DIR = BASE_DIR / "logs/eval_runs"
RESULTS_DIR = BASE_DIR / "logs/evaluation_results"


def find_latest_eval_file(version: str):
    version_dir = EVAL_RUN_DIR / version
    if not version_dir.exists():
        raise RuntimeError(f"No eval directory found for version: {version}")

    eval_files = sorted(version_dir.glob("*__eval.json"))
    if not eval_files:
        raise RuntimeError("No evaluation logs found.")

    return eval_files[-1]


def find_latest_csv(version: str):
    result_files = sorted(
        RESULTS_DIR.glob(f"{version}__*.csv")
    )
    if not result_files:
        return None
    return result_files[-1]


def print_summary(csv_path: Path):
    import csv

    grounded = []
    citation = []
    usefulness = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("manual_groundedness"):
                grounded.append(float(row["manual_groundedness"]))
            if row.get("manual_citation_score"):
                citation.append(float(row["manual_citation_score"]))
            if row.get("manual_usefulness"):
                usefulness.append(float(row["manual_usefulness"]))

    def avg(lst):
        return round(sum(lst)/len(lst), 2) if lst else 0

    print("\n=== Summary Metrics ===")
    print("Groundedness avg:", avg(grounded))
    print("Citation score avg:", avg(citation))
    print("Usefulness avg:", avg(usefulness))
    print("=======================\n")


def main():
    print("\n=== Phase 2 Full Evaluation Pipeline ===\n")

    # Import version dynamically
    from run_query import PROMPT_VERSION

    # 1️⃣ Run evaluation queries
    print("Step 1: Running evaluation queries...")
    subprocess.run([sys.executable, "-m", "src.eval.run_eval"], check=True,)

    # 2️⃣ Locate latest eval file
    latest_eval = find_latest_eval_file(PROMPT_VERSION)
    print(f"Latest evaluation log: {latest_eval.name}")

    # 3️⃣ Score evaluation
    print("Step 2: Scoring evaluation...")
    subprocess.run(
        [
            sys.executable,
            str(SCORE_SCRIPT),
            "--eval_path",
            str(latest_eval),
        ],
        check=True,
    )

    # 4️⃣ Extract clean grading file
    print("Step 3: Extracting grading input...")
    subprocess.run(
        [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--eval_path",
            str(latest_eval),
        ],
        check=True,
    )

    # 5️⃣ Print summary metrics (if manual scores exist)
    latest_csv = find_latest_csv(PROMPT_VERSION)
    if latest_csv:
        print_summary(latest_csv)

    print("=== Phase 2 Pipeline Complete ===\n")


if __name__ == "__main__":
    main()
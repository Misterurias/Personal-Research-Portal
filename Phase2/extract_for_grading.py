import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--eval_path", required=True)
args = parser.parse_args()

EVAL_PATH = Path(args.eval_path)
OUTPUT_PATH = EVAL_PATH.parent / "grading_input.json"


# =========================
# Extract answer from stdout
# =========================

def extract_answer(stdout: str):
    if not stdout or "=== Answer ===" not in stdout:
        return ""

    answer_part = stdout.split("=== Answer ===", 1)[1]

    if "Saved log:" in answer_part:
        answer_part = answer_part.split("Saved log:", 1)[0]

    return answer_part.strip()


# =========================
# Extract structured citations
# =========================

def extract_structured_citations(text):
    pattern = r"\([A-Za-z0-9_]+,\s*[A-Za-z0-9_]+_chunk_\d+\)"
    return re.findall(pattern, text)


# =========================
# Main
# =========================

def main():
    with open(EVAL_PATH, "r") as f:
        data = json.load(f)

    clean_runs = []

    for entry in data:
        run_id = entry.get("id", "")
        run_type = entry.get("type", "")
        query = entry.get("query", "")
        stdout = entry.get("stdout", "")

        answer = extract_answer(stdout)
        citations = extract_structured_citations(answer)

        clean_runs.append({
            "id": run_id,
            "type": run_type,
            "query": query.strip(),
            "answer": answer.strip(),
            "structured_citations": citations
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(clean_runs, f, indent=2)

    print(f"Grading input saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
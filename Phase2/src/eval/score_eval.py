import argparse
import json
import re
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--eval_path", required=True)
args = parser.parse_args()

EVAL_PATH = Path(args.eval_path)
PROMPT_VERSION = EVAL_PATH.parent.name
OUTPUT_PATH = Path("logs/evaluation_results") / f"{PROMPT_VERSION}__evaluation_scaffold.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Extraction Logic
# =========================

def extract_answer_from_stdout(stdout: str):
    """
    Extracts the model answer from the stdout field.
    Looks for the '=== Answer ===' marker.
    """
    if not stdout or "=== Answer ===" not in stdout:
        return ""

    # Split at answer marker
    answer_part = stdout.split("=== Answer ===", 1)[1]

    # Remove trailing Saved log section if present
    if "Saved log:" in answer_part:
        answer_part = answer_part.split("Saved log:", 1)[0]

    return answer_part.strip()


# =========================
# Citation Detection
# =========================

def has_structured_citation(text):
    # Matches: (SkillFormation2026, SkillFormation2026_chunk_018)
    pattern = r"\([A-Za-z0-9_]+,\s*[A-Za-z0-9_]+_chunk_\d+\)"
    return bool(re.search(pattern, text))


def count_structured_citations(text):
    pattern = r"\([A-Za-z0-9_]+,\s*[A-Za-z0-9_]+_chunk_\d+\)"
    return len(re.findall(pattern, text))


def has_fabricated_reference_style(text):
    """
    Detects bibliography-style references like:
    * DevExperienceGenAI2025: Brandebusemeyer et al. (2026)
    """
    bullet_style = r"\*\s+[A-Za-z0-9_]+:"
    apa_style = r"\b[A-Z][a-zA-Z]+ et al\. \(\d{4}\)"
    return bool(re.search(bullet_style, text) or re.search(apa_style, text))


def mentions_insufficient_evidence(text):
    return "insufficient evidence" in text.lower()


# =========================
# Main
# =========================

def main():
    with open(EVAL_PATH, "r") as f:
        data = json.load(f)

    rows = []

    for entry in data:
        query_id = entry.get("id", "")
        query_type = entry.get("type", "unknown")
        stdout = entry.get("stdout", "")

        answer = extract_answer_from_stdout(stdout)

        num_citations = count_structured_citations(answer)
        correct_format = has_structured_citation(answer)
        fabricated_format = has_fabricated_reference_style(answer)
        insufficient_flag = mentions_insufficient_evidence(answer)
        answer_length = len(answer.split())

        rows.append({
            "query_id": query_id,
            "query_type": query_type,
            "num_structured_citations": num_citations,
            "has_correct_citation_format": correct_format,
            "has_fabricated_reference_style": fabricated_format,
            "mentions_insufficient_evidence": insufficient_flag,
            "answer_word_count": answer_length,
            "manual_groundedness": "",
            "manual_citation_score": "",
            "manual_usefulness": "",
            "failure_tag": "",
            "notes": ""
        })

    with open(OUTPUT_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scaffold saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
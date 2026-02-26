import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from run_query import PROMPT_VERSION

EVAL_LOG_DIR = Path("logs/eval_runs") / PROMPT_VERSION

QUERY_SET_PATH = Path("src/eval/query_set.json")
RUN_QUERY_SCRIPT = Path("run_query.py")

def ensure_dirs():
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

def load_queries():
    with open(QUERY_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_single_query(query_obj):
    query_id = query_obj["id"]
    query_text = query_obj["query"]

    print(f"\n=== Running {query_id} ===")
    cmd = ["python", str(RUN_QUERY_SCRIPT), query_text]

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "id": query_id,
        "type": query_obj["type"],
        "query": query_text,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

def main():
    ensure_dirs()
    queries = load_queries()

    results = []
    for q in queries:
        result = run_single_query(q)
        results.append(result)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-")
    out_path = EVAL_LOG_DIR / f"{timestamp}__eval.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
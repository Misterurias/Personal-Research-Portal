import json
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "Phase2"))

from run_query import run_query_structured, PROMPT_VERSION

# -------------------------
# Setup
# -------------------------

EVAL_DIR = Path("logs/phase3_eval") / PROMPT_VERSION
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CITATION_RE = re.compile(r"\(([^,\)]+),\s*([^\)]+)\)")

QUESTIONS = [
    {"id": "Q1", "type": "productivity", "query": "Does GitHub Copilot increase developer productivity?"},
    {"id": "Q2", "type": "productivity", "query": "What empirical evidence measures productivity gains from AI coding tools?"},
    {"id": "Q3", "type": "productivity", "query": "How do AI coding assistants affect task completion time?"},
    {"id": "Q4", "type": "quality", "query": "What impact do AI coding assistants have on code quality?"},
    {"id": "Q5", "type": "quality", "query": "What quality metrics are used to evaluate AI-generated code?"},
    {"id": "Q6", "type": "quality", "query": "Is AI-assisted code more error-prone than human-written code?"},
    {"id": "Q7", "type": "technical_debt", "query": "What evidence links AI coding tools to technical debt accumulation?"},
    {"id": "Q8", "type": "technical_debt", "query": "Do AI coding assistants increase long-term maintenance burden?"},
    {"id": "Q9", "type": "experience", "query": "How do AI coding tools affect junior versus senior developers?"},
    {"id": "Q10", "type": "experience", "query": "Are experienced developers more effective when using AI assistants?"},
]

results = []

# -------------------------
# Run Evaluation
# -------------------------

for q in QUESTIONS:
    print(f"\nRunning {q['id']} — {q['query']}")
    
    r = run_query_structured(q["query"], top_k=6)

    answer_text = r["answer"]

    # Extract inline citations
    inline_citations = [
        {"source_id": m.group(1).strip(), "chunk_id": m.group(2).strip()}
        for m in CITATION_RE.finditer(answer_text)
    ]

    # Extract references section
    references_section = None
    if "References:" in answer_text:
        references_section = answer_text.split("References:", 1)[1].strip()

    results.append({
        "id": q["id"],
        "type": q["type"],
        "query": q["query"],
        "answer": answer_text,
        "inline_citations": inline_citations,
        "references_section_raw": references_section,
        "citations_valid": r["citations_valid"],
        "references_consistent": r["references_consistent"],
        "insufficient": answer_text.lower().startswith("insufficient"),
        "retrieved_chunks": [
            {
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "rank": chunk["rank"]
            }
            for chunk in r["retrieved"]
        ],
        "prompt_version": r["prompt_version"],
    })

# -------------------------
# Summary Metrics
# -------------------------

total = len(results)
valid = sum(r["citations_valid"] for r in results)
consistent = sum(r["references_consistent"] for r in results)
insufficient = sum(r["insufficient"] for r in results)

summary = {
    "total_queries": total,
    "citation_valid_count": valid,
    "reference_consistent_count": consistent,
    "insufficient_count": insufficient,
    "citation_valid_rate": round(valid / total, 3),
    "reference_consistent_rate": round(consistent / total, 3),
    "insufficient_rate": round(insufficient / total, 3),
    "prompt_version": PROMPT_VERSION,
}

print("\n=== Evaluation Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# -------------------------
# Save Results
# -------------------------

timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-")
out_path = EVAL_DIR / f"{timestamp}__phase3_eval.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "summary": summary,
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved evaluation to {out_path}")
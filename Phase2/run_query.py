import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunk_metadata.json"

# Retrieval defaults
DEFAULT_TOP_K = 5

# Model + prompt versioning (for reproducibility + logs)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3"
PROMPT_VERSION = "v2.5-testing_system_rules_prompt1"  # UPDATED
# PROMPT_VERSION = "v3-system-test"  # UPDATED

# LOG_DIR = Path("logs/runs")  / PROMPT_VERSION
LOG_DIR = BASE_DIR / "logs" / "runs" / PROMPT_VERSION

CITATION_RE = re.compile(r"\(([^,\)]+),\s*([^\)]+)\)")


# -------------------------
# Trust behavior (tightened) --> Phase 2
# -------------------------
# SYSTEM_RULES = """You are a research assistant answering ONLY from the provided evidence chunks.

# Hard rules:
# - Use ONLY the evidence below. Do not use outside knowledge.
# - Do NOT invent citations or references. Do NOT use bibliography/APA-style references.
# - Every major claim MUST have an inline citation in the exact format (source_id, chunk_id).
# - Only cite chunk IDs that appear in the Allowed citations list.
# - If the evidence does not support the answer, output EXACTLY:
# Insufficient evidence in the current corpus.
# Then provide 2–4 bullets describing what to search/add to the corpus.

# Output requirements:
# - Write a concise answer in 1–3 paragraphs (unless insufficient evidence).
# - End with "References:" and list ONLY the (source_id, chunk_id) pairs you used.
# - The "References:" list must contain ONLY citations that also appear inline in the answer.
# """

# # SYSTEM RULES PROMPT 1
SYSTEM_RULES = """You are a research assistant answering ONLY from the provided evidence chunks.

Hard rules:
- Use ONLY the evidence below. Do not use outside knowledge.
- Do NOT invent citations or references. Do NOT use bibliography/APA-style references.
- Every major claim MUST have an inline citation in the exact format (source_id, chunk_id).
- Only cite chunk IDs that appear in the Allowed citations list.
- If the evidence is partial or mixed, answer conservatively and explicitly describe uncertainty.
- Only output "Insufficient evidence..." if no relevant evidence exists at all.

Output requirements:
- Write a concise answer in 1–3 paragraphs (unless insufficient evidence).
- End with "References:" and list ONLY the (source_id, chunk_id) pairs you used.
- The "References:" list must contain ONLY citations that also appear inline in the answer.
"""

# SYSTEM RULES PROMPT 2
# SYSTEM_RULES = """You are a research assistant answering ONLY from the provided evidence chunks.

# Hard rules:
# - Use ONLY the evidence below. Do not use outside knowledge.
# - Do NOT invent citations or references.
# - Every major claim MUST have an inline citation in EXACT format: (source_id, chunk_id)
# - Only cite chunk IDs that appear in the Allowed citations list.
# - If the evidence is partial or mixed, answer conservatively and explicitly describe uncertainty.
# - Only output EXACTLY:
# Insufficient evidence in the current corpus.
# if no relevant evidence exists at all.

# Output format requirements (STRICT):
# 1. Write a concise answer in 1–3 paragraphs.
# 2. After the answer, output a blank line.
# 3. Then output EXACTLY this line:
# References:
# 4. On the next lines, list ONLY the unique citation pairs you used,
#    one per line, in EXACT format:
# (source_id, chunk_id)
# 5. Do NOT add bullets, numbers, dashes, or extra commentary.
# 6. Do NOT repeat citations.
# 7. The References list MUST match the inline citations exactly.
# """

# SYSTEM RULES PROMPT 3
# SYSTEM_RULES = """You are a research assistant answering ONLY from the provided evidence chunks.

# Grounding constraints:
# - Use ONLY the provided evidence.
# - Do NOT use outside knowledge.
# - Do NOT invent citations.
# - Inline citations must use EXACT format: (source_id, chunk_id).
# - Only cite pairs listed in the Allowed citations section.

# Answering behavior:
# - Synthesize findings clearly.
# - If evidence is mixed, explicitly describe uncertainty.
# - If no relevant evidence exists, output EXACTLY:
# Insufficient evidence in the current corpus.

# Before writing the final answer:
# - Ensure every major claim has an inline citation.
# - Ensure citations are drawn only from the allowed list.

# Output format:
# Write the answer in 1–3 concise paragraphs.

# After the answer, include a section labeled exactly:
# References:

# Under References:, list each unique (source_id, chunk_id) that appears inline.
# Each citation must appear exactly once.
# Do not include additional commentary in the References section.
# """

# -------------------------
# Enhancement: Query rewrite
# -------------------------
REWRITE_RULES = """Rewrite the user's question into a single, clearer search query for retrieval.

Rules:
- Keep it to ONE sentence.
- Preserve key entities, metrics, and constraints (years, author names, methods).
- Do NOT answer the question.
- Do NOT add new facts.
- Output ONLY the rewritten query text (no quotes, no bullets).
"""


def load_index_and_metadata() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing chunk metadata: {META_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    emb = model.encode([query], show_progress_bar=False)
    return np.array(emb).astype("float32")


def retrieve_top_k(
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    query_emb: np.ndarray,
    k: int,
) -> List[Dict[str, Any]]:
    # FAISS IndexFlatL2 returns squared L2 distances (lower is better)
    distances, indices = index.search(query_emb, k)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances, indices):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        results.append(
            {
                "rank": len(results) + 1,
                "distance_l2": float(dist),
                "source_id": c["source_id"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
            }
        )
    return results


def build_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
    allowed = [f"({r['source_id']}, {r['chunk_id']})" for r in retrieved]

    evidence_blocks = []
    for r in retrieved:
        evidence_blocks.append(
            f"[{r['chunk_id']} | {r['source_id']}]\n{r['text'].strip()}\n"
        )

    prompt = (
        f"{SYSTEM_RULES}\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Allowed citations (you may ONLY cite these):\n"
        + "\n".join(allowed)
        + "\n\n"
        f"Evidence:\n"
        + "\n".join(evidence_blocks)
    )
    return prompt


def call_ollama(prompt: str, temperature: float = 0.0) -> str:
    """
    Calls `ollama run llama3` and returns stdout.
    Uses stdin for prompt to avoid shell escaping issues.
    """
    base_cmd = ["ollama", "run", OLLAMA_MODEL_NAME]
    cmd_with_temp = base_cmd + ["--temperature", str(temperature)]

    try:
        proc = subprocess.run(
            cmd_with_temp,
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip()
    except subprocess.CalledProcessError:
        proc = subprocess.run(
            base_cmd,
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip()


def rewrite_query(original_query: str) -> str:
    prompt = f"{REWRITE_RULES}\n\nUser question:\n{original_query.strip()}\n"
    rewritten = call_ollama(prompt, temperature=0.0).strip()
    # Guardrail: if it returns something weird/empty, fall back
    if not rewritten or len(rewritten.split()) < 3:
        return original_query
    return rewritten


def extract_citations(text: str) -> List[Tuple[str, str]]:
    return [(m.group(1).strip(), m.group(2).strip()) for m in CITATION_RE.finditer(text)]


def validate_citations(
    citations: List[Tuple[str, str]],
    retrieved: List[Dict[str, Any]],
) -> Tuple[bool, List[Tuple[str, str]]]:
    allowed_pairs = set((r["source_id"], r["chunk_id"]) for r in retrieved)
    invalid = [c for c in citations if c not in allowed_pairs]
    return (len(invalid) == 0), invalid


def references_list(answer: str) -> List[Tuple[str, str]]:
    """
    Extract citations that appear in the "References:" section (if present).
    """
    if "References:" not in answer:
        return []
    ref_part = answer.split("References:", 1)[1]
    return extract_citations(ref_part)


def answer_is_insufficient(answer: str) -> bool:
    return answer.strip().lower().startswith("insufficient evidence in the current corpus.")


def enforce_reference_consistency(
    answer: str,
    inline_citations: List[Tuple[str, str]],
) -> Tuple[bool, str]:
    """
    Ensure:
    - If not insufficient evidence: there is a References: section
    - References list only contains citations that appear inline
    - Every inline citation appears in References
    """
    if answer_is_insufficient(answer):
        return True, answer  # no refs required by the template (but allowed)

    if "References:" not in answer:
        return False, "Missing References: section."

    refs = references_list(answer)
    inline_set = set(inline_citations)
    refs_set = set(refs)

    if not refs:
        return False, "References: section is empty."

    extra = refs_set - inline_set
    missing = inline_set - refs_set

    if extra:
        return False, f"References contains citations not used inline: {sorted(extra)}"
    if missing:
        return False, f"Inline citations missing from References: {sorted(missing)}"

    return True, answer


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(payload: Dict[str, Any]) -> Path:
    ensure_dirs()
    ts = payload["timestamp"].replace(":", "-")
    out_path = LOG_DIR / f"{ts}__run.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--show_evidence", action="store_true", help="Print retrieved chunks (preview)")
    parser.add_argument("--no_rewrite", action="store_true", help="Disable query rewriting enhancement")
    args = parser.parse_args()

    original_query = args.query.strip()
    top_k = args.top_k

    # Load retrieval assets
    index, chunks = load_index_and_metadata()
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # Enhancement: rewrite query (can disable)
    rewritten_query: Optional[str] = None
    retrieval_query = original_query
    if not args.no_rewrite:
        rewritten_query = rewrite_query(original_query)
        retrieval_query = rewritten_query

    # Retrieve
    q_emb = embed_query(embedder, retrieval_query)
    retrieved = retrieve_top_k(index, chunks, q_emb, top_k)

    if not retrieved:
        raise RuntimeError("No retrieval results. Check your index and metadata files.")

    # Build prompt + generate
    prompt = build_prompt(original_query, retrieved)
    answer = call_ollama(prompt, temperature=0.0)

    # Validate citations; retry once if it violates rules
    citations = extract_citations(answer)
    valid, invalid = validate_citations(citations, retrieved)

    # Additional trust: enforce references consistency
    refs_ok, refs_issue_or_answer = enforce_reference_consistency(answer, citations)

    if (not valid) or (not refs_ok):
        repair_prompt = (
            f"{SYSTEM_RULES}\n\n"
            f"Fix the answer to fully comply with the rules.\n"
            f"- Invalid citations found: {invalid}\n"
            f"- Reference consistency issue: {refs_issue_or_answer if not refs_ok else 'none'}\n"
            f"Rewrite the answer using ONLY Allowed citations.\n\n"
            f"Question:\n{original_query}\n\n"
            f"Allowed citations:\n"
            + "\n".join([f"({r['source_id']}, {r['chunk_id']})" for r in retrieved])
            + "\n\n"
            f"Evidence:\n"
            + "\n".join(
                [f"[{r['chunk_id']} | {r['source_id']}]\n{r['text'].strip()}\n" for r in retrieved]
            )
        )
        answer = call_ollama(repair_prompt, temperature=0.0)
        citations = extract_citations(answer)
        valid, invalid = validate_citations(citations, retrieved)
        refs_ok, refs_issue_or_answer = enforce_reference_consistency(answer, citations)

    # Print results
    print("\n=== Retrieved Chunks ===")
    for r in retrieved:
        print(f"{r['rank']}. {r['chunk_id']} ({r['source_id']})  distance_l2={r['distance_l2']:.4f}")

    if rewritten_query:
        print("\n=== Query Rewrite (Enhancement) ===")
        print(f"Original:  {original_query}")
        print(f"Rewritten: {rewritten_query}")

    if args.show_evidence:
        print("\n=== Evidence Previews ===")
        for r in retrieved:
            preview = (r["text"][:400] + "...") if len(r["text"]) > 400 else r["text"]
            print(f"\n[{r['chunk_id']} | {r['source_id']}]\n{preview}")

    print("\n=== Answer ===\n")
    print(answer)

    if not valid:
        print("\n⚠️ Warning: Answer still contains invalid citations:", invalid)
    if not refs_ok:
        print("\n⚠️ Warning: Reference consistency issue:", refs_issue_or_answer)

    # Log
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_payload = {
        "timestamp": now,
        "query": original_query,
        "retrieval_query": retrieval_query,
        "query_rewritten": bool(rewritten_query),
        "rewritten_query": rewritten_query,
        "top_k": top_k,
        "retrieved": [
            {
                "rank": r["rank"],
                "source_id": r["source_id"],
                "chunk_id": r["chunk_id"],
                "distance_l2": r["distance_l2"],
                "text_preview": (r["text"][:300] + "...") if len(r["text"]) > 300 else r["text"],
            }
            for r in retrieved
        ],
        "answer": answer,
        "citations_found": citations,
        "citations_valid": valid,
        "invalid_citations": invalid,
        "references_consistent": refs_ok,
        "references_issue": None if refs_ok else str(refs_issue_or_answer),
        "embed_model": EMBED_MODEL_NAME,
        "ollama_model": OLLAMA_MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
    }
    out_path = write_log(log_payload)
    print(f"\nSaved log: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--show_evidence", action="store_true")
    parser.add_argument("--no_rewrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_query_structured(
        query=args.query,
        top_k=args.top_k,
        no_rewrite=args.no_rewrite,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # CLI pretty print
    print("\n=== Retrieved Chunks ===")
    for r in result["retrieved"]:
        print(f"{r['rank']}. {r['chunk_id']} ({r['source_id']})  distance_l2={r['distance_l2']:.4f}")

    if result["query_rewritten"]:
        print("\n=== Query Rewrite (Enhancement) ===")
        print(f"Original:  {result['query']}")
        print(f"Rewritten: {result['rewritten_query']}")

    print("\n=== Answer ===\n")
    print(result["answer"])

    print(f"\nSaved log under logs/runs/{PROMPT_VERSION}")

def run_query_structured(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    no_rewrite: bool = False,
) -> Dict[str, Any]:
    """
    Programmatic entry point for Phase 3.
    Returns structured dict instead of printing.
    """

    original_query = query.strip()

    # Load retrieval assets
    index, chunks = load_index_and_metadata()
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    rewritten_query: Optional[str] = None
    retrieval_query = original_query

    if not no_rewrite:
        rewritten_query = rewrite_query(original_query)
        retrieval_query = rewritten_query

    q_emb = embed_query(embedder, retrieval_query)
    retrieved = retrieve_top_k(index, chunks, q_emb, top_k)

    if not retrieved:
        raise RuntimeError("No retrieval results.")

    prompt = build_prompt(original_query, retrieved)
    answer = call_ollama(prompt, temperature=0.2)

    citations = extract_citations(answer)
    valid, invalid = validate_citations(citations, retrieved)
    refs_ok, refs_issue_or_answer = enforce_reference_consistency(answer, citations)

    if (not valid) or (not refs_ok):
        repair_prompt = (
            f"{SYSTEM_RULES}\n\n"
            f"Fix the answer to fully comply with the rules.\n"
            f"- Invalid citations found: {invalid}\n"
            f"- Reference consistency issue: {refs_issue_or_answer if not refs_ok else 'none'}\n"
            f"Rewrite the answer using ONLY Allowed citations.\n\n"
            f"Question:\n{original_query}\n\n"
            f"Allowed citations:\n"
            + "\n".join([f"({r['source_id']}, {r['chunk_id']})" for r in retrieved])
            + "\n\n"
            f"Evidence:\n"
            + "\n".join(
                [f"[{r['chunk_id']} | {r['source_id']}]\n{r['text'].strip()}\n" for r in retrieved]
            )
        )
        answer = call_ollama(repair_prompt, temperature=0.0)
        citations = extract_citations(answer)
        valid, invalid = validate_citations(citations, retrieved)
        refs_ok, refs_issue_or_answer = enforce_reference_consistency(answer, citations)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    result = {
        "timestamp": now,
        "query": original_query,
        "retrieval_query": retrieval_query,
        "query_rewritten": bool(rewritten_query),
        "rewritten_query": rewritten_query,
        "top_k": top_k,
        "answer": answer,
        "citations": [
            {"source_id": s, "chunk_id": c}
            for (s, c) in citations
        ],
        "citations_valid": valid,
        "invalid_citations": invalid,
        "references_consistent": refs_ok,
        "retrieved": retrieved,
        "embed_model": EMBED_MODEL_NAME,
        "ollama_model": OLLAMA_MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
    }

    # Still log like Phase 2
    write_log(result)

    return result

if __name__ == "__main__":
    main()

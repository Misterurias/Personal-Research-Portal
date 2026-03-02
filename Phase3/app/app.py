import json
import uuid
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Import Phase2 RAG engine
sys.path.append(str(Path(__file__).resolve().parents[2] / "Phase2"))
from run_query import run_query_structured

BASE_DIR = Path(__file__).resolve().parents[1]
THREADS_DIR = BASE_DIR / "threads"
ARTIFACTS_DIR = BASE_DIR / "outputs" / "artifacts"
EXPORTS_DIR = BASE_DIR / "outputs" / "exports"

for d in [THREADS_DIR, ARTIFACTS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide")
st.title("Personal Research Portal (PRP) – Phase 3")

tab1, tab2, tab3 = st.tabs(["Ask", "Threads", "Artifacts"])

# -------------------------
# Helper functions
# -------------------------

def save_thread(data):
    thread_id = data["thread_id"]
    path = THREADS_DIR / f"{thread_id}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_threads():
    threads = []
    for p in THREADS_DIR.glob("*.json"):
        threads.append(json.loads(p.read_text(encoding="utf-8")))
    return sorted(threads, key=lambda x: x["timestamp"], reverse=True)

# -------------------------
# ASK TAB
# -------------------------

with tab1:
    query = st.text_input("Enter research question")

    top_k = st.slider("Top K Retrieval", 3, 10, 5)

    if st.button("Run Query") and query:
        with st.spinner("Running retrieval + generation..."):
            result = run_query_structured(query, top_k=top_k)

        thread = {
            "thread_id": str(uuid.uuid4())[:8],
            "timestamp": result["timestamp"],
            "query": result["query"],
            "answer": result["answer"],
            "retrieved": result["retrieved"],
            "citations": result["citations"],
            "prompt_version": result["prompt_version"],
            "embed_model": result["embed_model"],
            "ollama_model": result["ollama_model"],
            "query_rewritten": result["query_rewritten"],
            "rewritten_query": result["rewritten_query"],
            "citations_valid": result["citations_valid"],
            "references_consistent": result["references_consistent"],
        }

        save_thread(thread)
        st.success(f"Thread saved: {thread['thread_id']}")

        # -------------------------
        # Answer
        # -------------------------

        st.subheader("Answer")
        st.write(thread["answer"])

        # -------------------------
        # System Metadata
        # -------------------------

        st.subheader("System Metadata")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Prompt Version:", thread["prompt_version"])
            st.write("Embedding Model:", thread["embed_model"])
            st.write("LLM:", thread["ollama_model"])

        with col2:
            st.write("Query Rewritten:", thread["query_rewritten"])
            if thread["query_rewritten"]:
                st.write("Rewritten Query:", thread["rewritten_query"])

        if thread["citations_valid"] and thread["references_consistent"]:
            st.success("🟢 Grounding Verified")
        else:
            st.error("🔴 Grounding Issues Detected")

        # -------------------------
        # Retrieved Evidence
        # -------------------------

        st.subheader("Retrieved Evidence")
        for r in thread["retrieved"]:
            with st.expander(f"{r['chunk_id']} ({r['source_id']})"):
                st.write(r["text"])

# -------------------------
# THREADS TAB
# -------------------------

with tab2:
    threads = load_threads()

    if not threads:
        st.info("No threads yet.")
    else:
        for t in threads:
            with st.expander(f"{t['timestamp']} – {t['query'][:80]}"):
                st.write("Answer:")
                st.write(t["answer"])

                if t.get("citations_valid") and t.get("references_consistent"):
                    st.success("🟢 Grounding Verified")
                else:
                    st.error("🔴 Grounding Issues Detected")

# -------------------------
# ARTIFACTS TAB
# -------------------------

with tab3:
    threads = load_threads()

    if not threads:
        st.info("Run a query first.")
    else:
        thread_map = {f"{t['timestamp']} – {t['query'][:50]}": t for t in threads}
        selected = st.selectbox("Select thread", list(thread_map.keys()))
        t = thread_map[selected]

        rows = []
        for r in t["retrieved"]:
            rows.append({
                "claim": t["query"],
                "evidence_snippet": r["text"][:250] + "...",
                "citation": f"({r['source_id']}, {r['chunk_id']})",
                "confidence": "Med",
                "notes": "Retrieved supporting chunk"
            })

        df = pd.DataFrame(rows)
        st.dataframe(df)

        if st.button("Export Artifact"):
            csv_path = ARTIFACTS_DIR / f"{t['thread_id']}_evidence_table.csv"
            md_path = EXPORTS_DIR / f"{t['thread_id']}_evidence_table.md"

            df.to_csv(csv_path, index=False)

            md_content = f"# Evidence Table\n\nQuery: {t['query']}\n\n"
            md_content += df.to_markdown(index=False)

            md_path.write_text(md_content, encoding="utf-8")

            st.success("Exported:")
            st.code(str(csv_path))
            st.code(str(md_path))
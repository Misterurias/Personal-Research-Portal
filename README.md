# **Personal Research Portal (PRP) — AI Coding Assistant Impact Study**

Author: Jorge Urias  
Course: AI Model Development  
Phase: 2 — Ground the Research Domain  

---

# 📌 Research Domain

Impact of AI coding assistants (e.g., GitHub Copilot, CodeWhisperer, ChatGPT) on:

- Developer productivity
- Code quality
- Skill formation
- Technical debt
- Long-term maintainability

---

# 🎯 Main Research Question

> How do AI coding assistants affect productivity, code quality, and long-term software engineering outcomes?

---

# 🧠 Phase 2 Objective

Build a **research-grade Retrieval-Augmented Generation (RAG)** system with:

- Structured citations
- Traceable chunk-level evidence
- Logged retrieval + generation
- Reproducible evaluation pipeline
- Trust safeguards (no fabricated citations)

---

# 🚀 Quick Start (5-Minute Grader Path)

## 1️⃣ Clone the Repository

```bash
git clone <repo_url>
cd Phase2
```

# 2️⃣ Create Python Environment
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All dependencies are pinned for reproducibility.

# 3️⃣ Install Local LLM (Ollama Required)

This project uses a local LLM via Ollama for fully reproducible generation.

Install Ollama:
https://ollama.com

Pull the required model:

```
ollama pull llama3
```

Verify it works:

```
ollama run llama3
```

The system assumes Ollama is running at:

```
http://localhost:11434
```

# 4️⃣ Run Full Evaluation Pipeline
```
python run_phase2.py
```

This single command will:

* Run all 20 evaluation queries
* Retrieve top-k evidence chunks
* Generate answers with structured citations
* Save full logs
* Score evaluation runs
* Print summary metrics

# 📊 Evaluation Design

Evaluation query set includes:

* 10 Direct questions
* 5 Synthesis (multi-hop) questions
* 5 Edge / ambiguity cases

Metrics:

* Groundedness (1–4)
* Citation correctness (1–4)
* Usefulness (1–4)

Evaluation logs stored in:

```
logs/eval_runs/
logs/evaluation_results/
```

# 🧾 Corpus

The system uses 15–30 research sources including:

* Peer-reviewed papers
* Technical reports
* Empirical studies

Each source includes required metadata:

* source_id
* title
* authors
* year
* source_type
* venue
* url_or_doi
* raw_path
* processed_path
* tags
* relevance_note

Metadata file:

```
data/data_manifest.csv
```

Raw artifacts stored under:

```
data/raw/
```

# 🔍 RAG Architecture

**1.** Ingest PDFs → parse and clean

**2.** Section-aware chunking

**3.** Embedding using sentence-transformers

**4.** FAISS vector index

**5.** Top-k retrieval

**6.** Local LLM generation (Ollama)

**7.** Structured citations: (source_id, chunk_id)

**8.** Logging of:

* Query
* Retrieved chunks
* Prompt version
* Model output

# 🛡️ Trust Safeguards Implemented

* No fabricated citations allowed
* Strict abstention when insufficient evidence
* Structured citation enforcement
* Logged chunk IDs for traceability
* Evaluation explicitly flags invalid citation patterns

# 🔧 Enhancement Implemented (Phase 2 Requirement)

Enhancement: Trust tightening + structured citation enforcement

Improvements from baseline:

* Reduced fabricated references
* Stronger abstention on missing evidence
* Citation format normalization
* Evaluation pipeline automation

📁 Repository Structure
```
Phase2/
  README.md
  requirements.txt
  AI_USAGE_LOG.md
  run_query.py
  run_phase2.py
  data/
    raw/
    processed/
    data_manifest.csv
  src/
    ingest/
    rag/
    eval/
  logs/
    eval_runs/
    evaluation_results/
  report/
    Phase2_Report.md
```

# 🧪 Reproducibility Notes

* All Python dependencies pinned in `requirements.txt`
* Local LLM specified and documented
* All evaluation logs saved
* Raw research artifacts stored locally
* Single-command evaluation pipeline

# ⚠️ Common Failure Modes Addressed

* Fabricated citations → prevented
* Incorrect citation formatting → flagged
* Overconfident answers → guarded
* Missing evidence → explicit abstention

# 📄 Deliverables Included

* Baseline RAG implementation
* Enhancement implementation
* Data manifest
* Evaluation logs (machine-readable)
* Evaluation report (3–5 pages)
* AI usage disclosure
* Reproducible one-command pipeline

# 📚 AI Usage Disclosure

See:

```AI_USAGE_LOG.md```

# 🏁 Phase 2 Acceptance Test Checklist

* ✔ One command runs full evaluation
* ✔ Answers contain structured citations
* ✔ Citations resolve to real source text
* ✔ Logs saved
* ✔ At least one enhancement implemented
* ✔ Evaluation report includes failure cases

# 👨‍💻 Author

Jorge Urias

AI Model Development — Personal Research Portal Project

Carnegie Mellon University
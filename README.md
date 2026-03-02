**Personal Research Portal (PRP) --- AI Coding Assistant Impact Study**
=====================================================================

Author: Jorge Urias\
Course: AI Model Development\
Phases: 2 (Research-Grade RAG) + 3 (Research Portal Product)

* * * * *

📌 Research Domain
==================

Impact of AI coding assistants (e.g., GitHub Copilot, CodeWhisperer, ChatGPT) on:

-   Developer productivity

-   Code quality

-   Skill formation

-   Technical debt

-   Long-term maintainability

* * * * *

🎯 Main Research Question
=========================

> How do AI coding assistants affect productivity, code quality, and long-term software engineering outcomes?

* * * * *

🧠 Phase Overview
=================

Phase 2 --- Research-Grade RAG
----------------------------

-   Structured citations

-   Traceable chunk-level evidence

-   Logged retrieval + generation

-   Reproducible evaluation pipeline

-   Trust safeguards (no fabricated citations)

Phase 3 --- Research Portal Product
---------------------------------

-   Streamlit-based research interface

-   Threaded research sessions

-   Exportable artifacts

-   Evaluation harness for product-level testing

-   Persistent logging and traceability

* * * * *

🚀 Full Project Setup (Grader Instructions)
===========================================

1️⃣ Clone Repository
--------------------
```
git clone https://github.com/Misterurias/Personal-Research-Portal.git
cd PersonalResearchProject
```
* * * * *

2️⃣ Create Python Environment
-----------------------------
```
python -m venv .venv\
source .venv/bin/activate\
pip install -r requirements.txt
```
All dependencies are pinned for reproducibility.

* * * * *

3️⃣ Install Local LLM (Ollama Required)
---------------------------------------

This project uses a fully local LLM via Ollama for reproducible generation.

Install Ollama:

<https://ollama.com>

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
* * * * *

🖥️ Phase 3 --- Running the Research Portal (Streamlit App)
=========================================================

The main product interface is the Streamlit portal.

From the project root:
```
streamlit run Phase3/app/app.py
```
This launches the research portal in your browser.

* * * * *

What the Streamlit Portal Does
------------------------------

-   Accepts natural-language research questions

-   Retrieves top-k evidence chunks

-   Generates grounded answers with structured citations

-   Maintains threaded research sessions

-   Logs retrieval and generation artifacts

-   Supports exportable outputs

* * * * *

📁 Where Phase 3 Artifacts Are Stored
-------------------------------------

### 🔹 Threads (Research Conversations)
```
Phase3/threads/
```
Each JSON file represents a saved research session.

* * * * *

### 🔹 Snapshots (UI or Research State Snapshots)
```
Phase3/snapshots/
```
Used for capturing reproducible portal states.

* * * * *

### 🔹 Outputs (Exports / Generated Artifacts)
```
Phase3/outputs/
```
Contains:

-   Exported responses

-   Structured research outputs

-   Generated artifacts

* * * * *

### 🔹 Logs (Portal + Evaluation Logging)
```
Phase3/logs/
```

Includes:

-   Retrieval traces

-   Generation outputs

-   Prompt versions

-   Evaluation runs

* * * * *

🧪 Phase 3 Evaluation Harness
=============================

Phase 3 includes a product-level evaluation script.

To run the full Phase 3 evaluation:

```
python Phase3/phase3_eval.py
```
This script:

-   Runs predefined evaluation questions
-   Tests citation formatting
-   Checks grounding behavior
-   Logs outputs
-   Stores evaluation artifacts

* * * * *

📊 Phase 3 Evaluation Logs Location
-----------------------------------

```
Phase3/logs/phase3_eval/
```

Each evaluation run is versioned by prompt version.

* * * * *

🧾 Corpus & Data (Phase 2 Foundation)
=====================================

Corpus metadata:

```
Phase2/data/data_manifest.csv
```

Raw research artifacts:

```
Phase2/data/raw/
```

Processed chunks:

```
Phase2/data/processed/
```

FAISS index and embeddings built during ingestion.

* * * * *

🔍 RAG Architecture (Phases 2 & 3)
==================================

1.  PDF ingestion and cleaning

2.  Section-aware chunking

3.  Embedding with sentence-transformers

4.  FAISS vector index

5.  Top-k retrieval

6.  Local LLM generation (Ollama)

7.  Structured citations: (source_id, chunk_id)

8.  Logging of:

    -   Query

    -   Retrieved chunks

    -   Prompt version

    -   Model output

* * * * *

🛡️ Trust Safeguards
====================

-   No fabricated citations allowed

-   Strict abstention when insufficient evidence

-   Structured citation enforcement

-   Logged chunk IDs for traceability

-   Citation regex validation in evaluation

-   Versioned prompt control

* * * * *

📁 Full Repository Structure (Phases 2 + 3)
===========================================
```
Phase2/\
  run_phase2.py\
  run_query.py\
  data/\
  src/\
  logs/\
  report/

Phase3/\
  app/\
    app.py\
  phase3_eval.py\
  threads/\
  snapshots/\
  outputs/\
  logs/\
  README.md
```

* * * * *

🧪 Reproducibility Notes
========================

-   All dependencies pinned

-   Local LLM specified

-   All evaluation runs logged

-   Artifacts persisted to disk

-   No external APIs required

-   Fully reproducible on local machine

* * * * *

⚠️ Common Issues
================

If Streamlit does not launch:

-   Ensure `.venv` is activated

-   Ensure `requirements.txt` is installed

-   Ensure Ollama is running

If model fails:

```
ollama list
```

Confirm `llama3` is installed.

* * * * *

📄 Deliverables Included
========================

-   Research-grade RAG system (Phase 2)

-   Trust-enhanced RAG architecture

-   Streamlit research portal (Phase 3)

-   Evaluation harness (Phase 3)

-   Logs and traceable artifacts

-   Reports (Phase 2 + Phase 3)

-   AI usage disclosure

* * * * *

📚 AI Usage Disclosure
======================

See:

```
Phase2/AI_USAGE_LOG.md
```
* * * * *

👨‍💻 Author
============

Jorge Urias\
AI Model Development --- Personal Research Portal\
Carnegie Mellon University
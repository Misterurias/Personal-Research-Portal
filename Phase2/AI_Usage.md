AI Usage Disclosure: Phase 2 (Research-Grade RAG)
==================================================

Overview
--------

AI tools were used during Phase 2 as development assistants and productivity aids. All design decisions, system architecture, evaluation criteria, and final outputs were reviewed, modified, and validated manually by the project author. The author retains full responsibility for correctness, citations, and compliance with project requirements.

* * * * *

1\. Code Development Assistance
-------------------------------

**Tool Used:** ChatGPT (OpenAI)\
**Purpose:**

-   Debugging Python scripts (e.g., run_eval.py, score_eval.py, run_phase2.py)

-   Clarifying Git branch tracking behavior

-   Suggesting improvements to logging structure and evaluation pipelines

-   Assisting in writing reproducibility scripts

**Manual Modifications:**

-   All generated code was reviewed and adapted to fit the project structure.

-   Logging paths, evaluation schema, and versioning conventions were manually aligned with Phase 2 requirements.

-   Retrieval logic, citation formatting rules, and trust guardrails were implemented and verified manually.

-   Errors and environment-specific fixes were resolved independently.

* * * * *

2\. RAG Design and Enhancement Strategy
---------------------------------------

**Tool Used:** ChatGPT\
**Purpose:**

-   Brainstorming enhancements (e.g., structured citations, trust behaviors, evaluation scoring)

-   Discussing evaluation metrics and failure modes

-   Reviewing improvement ideas (e.g., query decomposition, citation validation)

**Manual Decisions:**

-   Selection of FAISS as vector store

-   Chunking strategy (size, overlap, section awareness)

-   Citation formatting standard (source_id + chunk_id)

-   Evaluation query set design

-   Definition of groundedness, citation correctness, and usefulness rubric

-   Implementation of trust behavior (explicit "insufficient evidence" handling)

* * * * *

3\. Documentation & Report Drafting
-----------------------------------

**Tool Used:** ChatGPT\
**Purpose:**

-   Drafting README structure

-   Formatting evaluation report outline

-   Producing AI usage disclosure draft

-   Structuring Markdown templates

**Manual Modifications:**

-   All content was edited to reflect actual implementation.

-   Metrics, results, and failure case descriptions are based solely on system outputs and manual evaluation.

-   All claims in the final report are grounded in the project's evaluation logs.

* * * * *

4\. What AI Was NOT Used For
----------------------------

AI was **not** used to:

-   Generate or fabricate corpus sources

-   Modify source documents

-   Alter retrieval results

-   Generate citations without validation

-   Bypass reproducibility requirements

All evaluation scores were manually assigned according to the Phase 2 rubric.

All retrieved citations resolve to real text stored locally under `data/raw/` and `data/processed/`.

* * * * *

5\. Responsibility Statement
----------------------------

The author confirms that:

-   All retrieval results and citations originate from the project's local corpus.

-   All evaluation metrics and scoring reflect manual review.

-   AI-assisted suggestions were critically evaluated before integration.

-   The final system meets reproducibility and trust requirements independent of AI tools.

* * * * *

6\. Summary of AI Role in Phase 2
---------------------------------

AI was used as:

-   A development assistant

-   A debugging aid

-   A structural editor

AI was not used as:

-   An automated research engine

-   A citation generator

-   An evaluator

*Conversations with ChatGPT can be found in `Phase2/AI_Usage/Phase2AIUsageDiscussion.pdf`*

All intellectual responsibility and verification remain with the author.
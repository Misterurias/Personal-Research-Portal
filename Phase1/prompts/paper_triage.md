# Task: Paper Triage

## Intent
Summarize a research paper on AI-assisted software engineering in a structured, research-oriented format. This task tests whether models can accurately extract key elements of a paper without hallucinating or over-interpreting results.

## Inputs
- A single academic paper or report (PDF text or excerpt) related to AI tools in software engineering.

## Required Output Structure
The output must contain exactly the following five fields:
1. Contribution
2. Method
3. Data
4. Findings
5. Limitations


## Prompt A: Baseline

You are given an academic paper about AI tools in software engineering.

Summarize the paper using the following fields:
- Contribution
- Method
- Data
- Findings
- Limitations


## Prompt B: Structured with Guardrails

You are analyzing an academic paper about AI-assisted tools in software engineering.

Produce a structured summary using **exactly** the following five fields:
1. Contribution
2. Method
3. Data
4. Findings
5. Limitations

### Constraints / Guardrails
- Base all statements strictly on the content of the provided paper.
- Do not infer or speculate beyond what is explicitly supported by the text.
- If the paper does not clearly specify information for a field, write: **“Not specified in the paper.”**
- Be concise, factual, and neutral in tone.
- Do not introduce external knowledge or assumptions.

Output only the five fields listed above.


## Why These Constraints Exist
- **Exact fields:** Ensures outputs are comparable across runs and models.
- **“Not specified” requirement:** Prevents hallucination when information is missing.
- **No speculation:** Tests whether the model can resist filling gaps with plausible-sounding but unsupported claims.
- **Concise, neutral tone:** Encourages research-style summarization rather than narrative explanation.

## Failure Modes to Watch For
- Hallucinated methods or datasets.
- Overstated findings not supported by the paper.
- Vague or generic limitations not grounded in the text.
- Ignoring the required structure.

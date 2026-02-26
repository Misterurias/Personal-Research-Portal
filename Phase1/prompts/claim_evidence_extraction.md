# Task: Claim–Evidence Extraction

## Intent
Extract concrete claims from research papers and pair them with direct supporting evidence. This task explicitly tests grounding and citation faithfulness.

## Inputs
- A research paper or excerpt related to AI tools and software engineering.
- Each paper is associated with a known `source_id`.

## Required Output Structure
A table with exactly five rows and the following columns:
| Claim | Evidence Snippet | Citation (source_id, chunk_id) |


## Prompt A: Baseline

From the provided paper, extract five key claims about the impact of AI tools on software engineering.

For each claim, include:
- The claim
- A supporting quote or snippet from the text
- A citation

Present your answer in a table.


## Prompt B: Structured with Guardrails

You are extracting claims and evidence from a research paper about AI-assisted software engineering.

Produce a table with **exactly five rows** and the following columns:

| Claim | Evidence Snippet | Citation (source_id, chunk_id) |

### Constraints / Guardrails
- Each claim must be directly supported by the provided evidence snippet.
- Evidence snippets must be copied verbatim or closely paraphrased from the text.
- Citations must use the format: **(source_id, chunk_id)**.
- Do not invent claims, evidence, or citations.
- If the paper does not contain enough information to extract five well-supported claims, include fewer claims and explicitly state **“Insufficient evidence for additional claims.”**
- Do not rely on external knowledge.


## Why These Constraints Exist
- **Fixed table format:** Enables consistent evaluation across models and prompts.
- **Direct evidence requirement:** Forces grounding and discourages abstract or generalized claims.
- **Explicit citation format:** Tests whether models can correctly attribute claims to source material.
- **Insufficient evidence rule:** Rewards uncertainty handling rather than hallucination.

## Failure Modes to Watch For
- Claims that are broader than the evidence supports.
- Fabricated or vague citations.
- Evidence snippets that do not actually support the stated claim.
- Reworded “common knowledge” instead of paper-specific findings.

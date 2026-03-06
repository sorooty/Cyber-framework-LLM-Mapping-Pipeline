# RiskHunter – Copilot Instructions

## Project Context

RiskHunter is a French AI-augmented GRC (Governance, Risk & Compliance) SaaS platform. It helps organizations centrally manage audits, risk cartographies, internal controls, compliance analyses, and documentary evidence through automated AI pipelines.

**Current primary module:** Referential Mapping Pipeline — automatically mapping security requirements across compliance frameworks (CIS v8, NIST SP 800-53, ISO 27001, NIS2, DORA, SOC2) to propagate compliance status across frameworks and reduce audit effort.

---

## Tech Stack

- **Python** (primary language)
- **pandas** — multi-sheet Excel/CSV/JSON parsing, NaN handling, deduplication
- **Sentence Transformers** (`all-MiniLM-L6-v2`) or **OpenAI** (`text-embedding-3-small`) for vectorization
- **NumPy / SciPy sparse matrices** — cosine similarity computation, compliance propagation adjacency matrix
- **LLM providers** (OpenAI, Mistral, or local models) — structured JSON output for requirement analysis
- **scikit-learn** — when additional ML utilities are needed

---

## Architecture: Referential Mapping Pipeline

### Data Model

Each normalized requirement has the shape:
```python
{
  "id": str,           # e.g. "CIS-1.1", "ISO-A.8.1"
  "framework": str,    # "CIS_v8" | "NIST_800-53" | "ISO_27001" | "NIS2" | "DORA" | "SOC2"
  "chapter": str,
  "title": str,
  "description": str,
  "tags": list[str]
}
```

### Pipeline Stages

1. **Ingestion** (`ingestion/`) — Parse Excel/JSON/CSV sources into normalized requirement records. Handle multi-sheet workbooks with pandas; drop NaN rows, deduplicate on `id`.

2. **Vectorization** (`vectorization/`) — Embed `title + " " + description` per requirement. Compute cosine similarity matrix; filter candidates at configurable threshold (default `0.7`). Avoid brute-force O(n²) by pre-filtering with FAISS or top-k cosine on sparse candidates.

3. **LLM Analysis** (`llm/`) — For each candidate pair, call LLM with a structured prompt. Output must be parseable JSON:
   ```json
   {
     "coverage_A_to_B": 0.0,
     "coverage_B_to_A": 0.0,
     "confidence": 0.0,
     "relation_type": "equivalence | A_covers_B | B_covers_A | partial | none"
   }
   ```

4. **Compliance Propagation** (`propagation/`) — Build adjacency matrix (SciPy sparse). Transitive inference rule: if A is compliant and `coverage_A_to_B >= 0.8`, infer B is probably compliant with a confidence decay factor.

5. **Output** (`output/`) — Export:
   - `relations.json` — sourced mapping relations
   - `matrix.csv` / `matrix.xlsx` — compliance coverage matrix
   - `graph.graphml` — for visualization tools (e.g., Gephi, Neo4j)

---

## Key Conventions

### LLM Prompts
- Always enforce JSON output via system prompt or `response_format={"type": "json_object"}` (OpenAI).
- Prompts assess **bidirectional** coverage independently (`coverage_A_to_B` ≠ `coverage_B_to_A`).
- Validate parsed LLM output against a Pydantic model before use.

### Thresholds (configurable via config)
| Parameter | Default | Meaning |
|---|---|---|
| `similarity_threshold` | `0.7` | Min cosine sim to consider a candidate pair |
| `propagation_threshold` | `0.8` | Min `coverage_A_to_B` for transitive compliance |
| `confidence_decay` | `0.9` | Multiplier per propagation hop |

### Traceability
- Every mapping relation must carry `source` (embedding-based, LLM-assessed) and `confidence` fields.
- Never overwrite a human-validated relation with an automated one; mark automated relations with `"auto": true`.

### Framework Identifiers
Use these canonical keys throughout the codebase:
`CIS_v8`, `NIST_800-53`, `ISO_27001`, `NIS2`, `DORA`, `SOC2`

### Module Structure
Trello backlog modules map to top-level directories:
`audit/`, `risks/`, `questionnaires/`, `rag/`, `security/`, `gdpr/`, `api/`, `voice/`

---

## Modules Overview

| Module | Purpose |
|---|---|
| `referential_mapping/` | Cross-framework compliance mapping pipeline (current priority) |
| `audit/` | Processing voluminous JSON audit outputs, generating summaries |
| `risks/` | Risk cartography generation from client documents (LLM prompt engineering) |
| `questionnaires/` | Automated questionnaire analysis |
| `rag/` | RAG pipeline for documentary evidence retrieval |
| `gdpr/` | GDPR compliance analysis |
| `api/` | Integrations with external GRC tools |
| `voice/` | Voice AI agents |

---

## Development Priorities (Roadmap)

1. **Quick wins** — CIS v8 ↔ ISO 27001 mapping (MVP, leverages existing Excel data)
2. **Foundational builds** — Generalize pipeline to all 5 frameworks; build propagation engine
3. **Long-term** — Full SaaS integration, voice agents, real-time compliance dashboards

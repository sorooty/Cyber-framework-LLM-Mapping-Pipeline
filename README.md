# RiskHunter — Referential Mapping Pipeline

> Automatically map security requirements across any two compliance frameworks using semantic search and LLM verification.

---

## What it does

Given two security frameworks (e.g. **CIS Controls v8** and **NIST CSF v2**), this pipeline identifies which requirements correlate with each other — and how strongly — without requiring any manual review.

It works as a **funnel**: start with every possible pair, eliminate the irrelevant ones fast and cheap, then verify the promising ones intelligently.

```
Ref A (153 requirements) × Ref B (106 requirements) = 16,218 combinations
        │
  [Step 1]  Parse & normalize YAML sources
        │
  [Step 2]  Semantic similarity on titles → keep top candidates   ~4%
        │
  [Step 3]  Correlation heatmap (visual diagnostic)
        │
  [Step 4]  LLM scoring on title + description → final verdict    ~1%
        │
  Excel export  (clear mapping: "Req A.1 ↔ Req B.6 — equivalence, score 0.87")
```

---

## Pipeline steps

### Step 1 — Parse & Normalize
Reads YAML source files and produces a clean, unified list of requirements — each with an `id`, `title`, `description`, and `tags`. Tags (e.g. IG1/IG2 for CIS, function prefix for NIST) are joined here. Results are cached to avoid re-parsing on subsequent runs.

### Step 2 — Semantic Similarity (titles only)
Encodes all requirement titles using `all-MiniLM-L6-v2` (local, free, no API call). Computes cosine similarity across all pairs and retains the **top-k per requirement** above a configurable threshold. This step alone eliminates ~96% of irrelevant combinations.

### Step 3 — Correlation Matrix
Generates a heatmap of the full similarity matrix — requirements from Ref A on rows, Ref B on columns. **Purely visual and indicative**: useful for spotting correlation clusters and validating that Step 2 retained the right candidates.

### Step 4 — LLM Verification (title + description)
The only step that uses an LLM (`gpt-4o-mini`). For each candidate pair, submits both `title + description` and asks the model to score:
- `coverage_A_to_B` — to what extent A satisfies B
- `coverage_B_to_A` — to what extent B satisfies A
- `confidence` — model's confidence
- `relation_type` — `equivalence` | `A_covers_B` | `B_covers_A` | `partial` | `no_link`

High-scoring pairs go through a **second LLM pass** (double verification). Scores from both passes are averaged for robustness. Results are exported to a clear Excel file and a JSON file.

---

## Project structure

```
RiskHunter/
├── app.py                          # Streamlit UI — interactive demo
├── pipeline_standalone.py          # Single-file pipeline (all logic, no package dependency)
├── pipeline_sandbox.ipynb          # Notebook — run the pipeline step by step
├── requirements.txt
├── .env                            # OPENAI_API_KEY (local only, gitignored)
├── files/
│   ├── survey.ts                   # Structural schema for all frameworks
│   └── surveys/                    # Source YAML files (40+ frameworks)
└── referential_mapping/            # Original modular package
    ├── pipeline.py                 # CLI entrypoint
    ├── config.py                   # All thresholds & parameters
    ├── models.py                   # Shared data models
    ├── schemas.py                  # FrameworkSchema + SCHEMAS registry
    ├── survey_validator.py         # Validates parsed counts against survey.ts
    ├── ingestion/
    │   ├── ingester.py             # Step 1 orchestrator (with caching)
    │   ├── registry.py             # Framework discovery (list_frameworks)
    │   └── adapters/
    │       ├── generic_yaml.py     # Universal YAML parser
    │       ├── cis_v8.py           # CIS Controls v8 YAML parser
    │       └── nist_csf_v2.py      # NIST CSF v2 YAML parser
    ├── semantic/
    │   └── similarity.py           # Step 2 — embeddings + cosine similarity
    ├── visualization/
    │   └── heatmap.py              # Step 3 — seaborn heatmap
    └── llm/
        └── scorer.py               # Step 4 — async LLM scoring + Excel export
```

> **`pipeline_standalone.py`** consolidates the entire `referential_mapping/` package into a single importable file. `app.py` uses it exclusively.

---

## Quickstart

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit app
```bash
streamlit run app.py
```

### Run via CLI (standalone)
```bash
# Steps 1–3 only (no LLM, no API key needed)
python pipeline_standalone.py \
  --ref-a files/surveys/cis-controls-v8-1.yaml --adapter-a cis_v8 \
  --ref-b files/surveys/nistCsfV2.yaml          --adapter-b nist_csf_v2 \
  --skip-step4

# Full pipeline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python pipeline_standalone.py \
  --ref-a files/surveys/cis-controls-v8-1.yaml --adapter-a cis_v8 \
  --ref-b files/surveys/nistCsfV2.yaml          --adapter-b nist_csf_v2
```

### Cache & re-runs
```bash
--force-step1   # re-parse source files
--force-step2   # recompute embeddings
--force-step4   # re-score with LLM
```

### Available adapters
| Name | Framework |
|---|---|
| `cis_v8` | CIS Controls v8 |
| `nist_csf_v2` | NIST CSF v2 |
| `generic` | Any other YAML in `files/surveys/` |

---

## Configuration

All thresholds are defined at the top of `pipeline_standalone.py` (and mirrored in `referential_mapping/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (no API key) |
| `SEMANTIC_THRESHOLD` | `0.40` | Min cosine score to keep a pair (Step 2) |
| `TOP_K` | `5` | Max candidates per requirement (Step 2) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM used for scoring (Step 4) |
| `LLM_CONFIRM_THRESHOLD` | `0.75` | Min LLM score to trigger double verification |
| `LLM_CONCURRENCY` | `5` | Parallel LLM calls (asyncio semaphore) |

---

## Adding a new framework

1. Add your YAML file to `files/surveys/`
2. The `generic` adapter handles most YAML structures automatically — try it first
3. If the structure is non-standard, add a custom `load(path)` function in `pipeline_standalone.py` and register it in `ADAPTER_LOADERS`

---

## References

| # | Auteurs / Source | Titre | Lien |
|---|---|---|---|
| 1 | Nature (2025) | Two-stage LLM approach for entity mapping | [↗](https://www.nature.com/articles/s41598-025-16213-z) |
| 2 | EMNLP 2025 | Hybrid LLM & Embedding for Semantic Attribute Mapping | [↗](https://aclanthology.org/2025.emnlp-industry.120.pdf) |
| 3 | arXiv 2025 | AutoPK: Hybrid Similarity + LLM | [↗](https://arxiv.org/html/2510.00039v1) |
| 4 | journal-isi (2024) | NLP for Regulatory Document Interconnections | [↗](https://journal-isi.org/index.php/isi/article/download/861/436) |
| 5 | Semantic Web Journal | RP-Match: Automatic Regulation Mapping | [↗](https://www.semantic-web-journal.net/sites/default/files/swj312.pdf) |
| 6 | RegNLP 2025 | Regulatory QA using Generative AI | [↗](https://aclanthology.org/2025.regnlp-1.16.pdf) |
| 7 | Frontiers (2025) | Ontology-Based Regulatory Document Analysis | [↗](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2025.1575913/full) |
| 8 | IEEE TSE (2023) | NLP-Based Automated Compliance Checking | [↗](https://dl.acm.org/doi/abs/10.1109/TSE.2023.3288901) |
| 9 | arXiv (2025) | Hybrid Retrieval for Hallucination Mitigation | [↗](https://arxiv.org/html/2504.05324v1) |
| 10 | arXiv (2025) | RAG Architectures for Policy Documents | [↗](https://arxiv.org/html/2601.15457v1) |
| 11 | Reimers & Gurevych (2019) | Sentence-BERT | [↗](https://arxiv.org/abs/1908.10084) |
| 12 | SBERT.net | STS Documentation | [↗](https://sbert.net/examples/sentence_transformer/training/sts/README.html) |
| 13 | Milvus (2026) | Sentence Transformers — common mistakes | [↗](https://milvus.io/ai-quick-reference/what-are-common-mistakes-that-could-lead-to-poor-results-when-using-sentence-transformer-em) |
| 14 | PMC (2025) | Semantic Similarity on Long Texts | [↗](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453783/) |
| 15 | SCITEPRESS (2025) | Ontology-Based System Requirements | [↗](https://www.scitepress.org/Papers/2025/132105/132105.pdf) |
| 16 | model-engineering.info (2024) | Requirements-to-Code Traceability | [↗](https://model-engineering.info/publications/papers/ER24-Requirements2Code.pdf) |
| 17 | intuitem (2024) | AI-Assisted Compliance Mapping | [↗](https://intuitem.com/ai-assisted-mapping) |


---
*by Sensey*

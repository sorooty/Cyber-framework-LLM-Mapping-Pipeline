# RiskHunter — Referential Mapping Pipeline

> Automatically map security requirements across compliance frameworks using semantic search and LLM verification.

---

## What it does

Given any set of security frameworks (e.g. **CIS Controls v8**, **NIST CSF v2**, **ISO 27001**, **NIS2**, **DORA**, **SOC2**), this pipeline identifies which requirements correlate with each other — and how strongly — without requiring manual review.

It works as a **funnel**: start with every possible pair across all frameworks, eliminate irrelevant ones fast and cheaply with local embeddings, then verify the promising candidates intelligently with an LLM.

```
N frameworks → C(N,2) pairs of frameworks, each pair processed through:
        │
  [Step 1]  Parse & normalize YAML sources
        │
  [Step 2]  Semantic similarity (local embeddings) → keep top candidates   ~4%
        │
  [Step 3]  Correlation heatmap (visual diagnostic)
        │
  [Step 4]  LLM scoring on title + description → final verdict             ~1%
        │
  Excel export (color-coded by relation type) + global JSON
```

---

## Pipeline steps

### Step 1 — Parse & Normalize
Reads YAML source files and produces a clean, unified list of requirements — each with an `id`, `title`, `description`, and `tags`. A single generic adapter (`load_generic_yaml`) handles all frameworks via `FrameworkSchema`, which defines the ID prefix per framework (e.g. `CIS-`, `NIST-`). Results are cached as JSON to avoid re-parsing on subsequent runs.

### Step 2 — Semantic Similarity
Encodes all requirement titles using `paraphrase-multilingual-MiniLM-L12-v2` (local, free, no API call — supports both French and English). Computes cosine similarity across all pairs and retains the **top-k per requirement** above a configurable threshold. This step alone eliminates ~96% of irrelevant combinations. The embedding model is cached at module level to avoid reloading between runs.

Pairs are filtered by both a minimum cosine score (`SEMANTIC_THRESHOLD = 0.40`) and a rank constraint (`TOP_K = 5`), ensuring that only genuinely close requirements reach the LLM.

### Step 3 — Correlation Matrix
Generates a heatmap of the full similarity matrix — requirements from framework A on rows, framework B on columns. Purely visual and diagnostic: useful for spotting correlation clusters and validating that Step 2 retained the right candidates.

### Step 4 — LLM Verification (title + description)
The only step that calls an external API (`gpt-4o-mini`). For each candidate pair, submits both `title + description` and asks the model to score:
- `coverage_A_to_B` — to what extent A satisfies B
- `coverage_B_to_A` — to what extent B satisfies A (assessed independently — never averaged with the above)
- `confidence` — model's confidence in the scores
- `relation_type` — `equivalence` | `A_covers_B` | `B_covers_A` | `partial` | `no_link`
- `justification` — a concise 1–2 sentence explanation

High-scoring pairs (`≥ LLM_CONFIRM_THRESHOLD`) go through a **second LLM pass** for double verification; scores from both passes are averaged. LLM output is validated with Pydantic before use. API calls are batched (`LLM_BATCH_SIZE = 10` pairs per call) and parallelized with asyncio (`LLM_CONCURRENCY = 20`).

**Adaptive thresholds**: relation classification thresholds automatically adjust based on the cosine score distribution of the current dataset pair, preventing over-classification when two frameworks share closely related vocabulary (e.g. ISO 27001 ↔ NIS2).

---

## Project structure

```
RiskHunter/
├── app.py                          # Streamlit UI - interactive demo
├── pipeline_standalone.py          # Single-file pipeline (all logic, no package dependency)
├── notebook.py                     # Marimo notebook (run the pipeline step by step) - DEPRECATED (kept for reference)
├── requirements.txt
├── .env                            # OPENAI_API_KEY (local only, gitignored)
├── files/
│   ├── survey.ts                   # Structural schema for all frameworks (expected counts)
│   └── surveys/                    # Source YAML files (40+ frameworks)
├── docs/                           # Internal documentation & analysis
└── referential_mapping/            # Original modular package (kept for reference)
    ├── pipeline.py                 # CLI entrypoint
    ├── config.py                   # All thresholds & parameters
    ├── models.py                   # Shared data models
    ├── schemas.py                  # FrameworkSchema + SCHEMAS registry
    ├── survey_validator.py         # Validates parsed counts against survey.ts
    ├── ingestion/
    │   ├── ingester.py             # Step 1 orchestrator (with caching)
    │   ├── registry.py             # Framework discovery (list_frameworks)
    │   └── adapters/
    │       └── generic_yaml.py     # Universal YAML adapter for all frameworks
    ├── semantic/
    │   └── similarity.py           # Step 2 — embeddings + cosine similarity
    ├── visualization/
    │   └── heatmap.py              # Step 3 — seaborn heatmap
    └── llm/
        └── scorer.py               # Step 4 — async LLM scoring + exports
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
  --ref-a files/surveys/cis-controls-v8-1.yaml --adapter-a generic \
  --ref-b files/surveys/nistCsfV2.yaml          --adapter-b generic \
  --skip-step4

# Full pipeline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python pipeline_standalone.py \
  --ref-a files/surveys/cis-controls-v8-1.yaml --adapter-a generic \
  --ref-b files/surveys/nistCsfV2.yaml          --adapter-b generic
```

### Cache & re-runs
```bash
--force-step1   # re-parse source files
--force-step2   # recompute embeddings
--force-step4   # re-score with LLM
```

---

## Configuration

All thresholds are defined at the top of `pipeline_standalone.py` (and mirrored in `referential_mapping/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local multilingual embedding model (FR/EN, no API key) |
| `SEMANTIC_THRESHOLD` | `0.40` | Min cosine score to keep a pair (Step 2) |
| `TOP_K` | `5` | Max candidates per requirement (Step 2) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM used for scoring (Step 4) |
| `LLM_CONFIRM_THRESHOLD` | `0.75` | Min LLM score to trigger double verification |
| `LLM_CONCURRENCY` | `20` | Parallel LLM calls (asyncio semaphore) |
| `LLM_BATCH_SIZE` | `10` | Requirement pairs per API call |
| `LLM_MIN_CONFIDENCE` | `0.40` | Min LLM confidence to keep a relation (Step 5 cleanup) |
| `THRESHOLD_EQUIVALENCE` | `0.85` | Min bidirectional coverage to classify as `equivalence` |
| `THRESHOLD_COVERAGE` | `0.75` | Min unidirectional coverage for directional relation types |

---

## Export formats

Each pipeline run produces the following outputs in `data/outputs/`:

| File | Content |
|---|---|
| `mapping_results.xlsx` | Color-coded Excel for a single framework pair |
| `mapping_summary.xlsx` | Multi-sheet workbook — one sheet per pair + a Global sheet |
| `mapping_all.json` | All relations across all pairs, consolidated in a single JSON array |
| `mapping_all.html` | Interactive HTML review file with checkboxes and localStorage persistence |

In the Streamlit UI, the **Export** tab provides download buttons for all three formats directly in the browser (no file system access required).

---

## Adding a new framework

1. Add your YAML file to `files/surveys/`
2. Use `--adapter-a generic` — the generic adapter handles most YAML structures automatically
3. If needed, add a `FrameworkSchema` entry in `pipeline_standalone.py` to configure the ID prefix

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

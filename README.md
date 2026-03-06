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
├── mapping_pipeline.ipynb          # Notebook — run the pipeline step by step
├── requirements.txt
├── files/surveys/                  # Source YAML files (CIS, NIST, ISO, DORA, ...)
└── referential_mapping/
    ├── pipeline.py                 # CLI entrypoint
    ├── config.py                   # All thresholds & parameters
    ├── models.py                   # Shared data models
    ├── ingestion/
    │   ├── ingester.py             # Step 1 orchestrator (with caching)
    │   └── adapters/
    │       ├── cis_v8.py           # CIS Controls v8 YAML parser
    │       └── nist_csf_v2.py      # NIST CSF v2 YAML parser
    ├── semantic/
    │   └── similarity.py           # Step 2 — embeddings + cosine similarity
    ├── visualization/
    │   └── heatmap.py              # Step 3 — seaborn heatmap
    ├── llm/
    │   └── scorer.py               # Step 4 — async LLM scoring + Excel export
    └── data/
        ├── ref_A_normalized.json   # Step 1 cache
        ├── ref_B_normalized.json
        ├── candidate_pairs.json    # Step 2 cache
        ├── similarity_matrix.npy
        └── outputs/
            ├── correlation_matrix.png
            ├── mapping_results.xlsx
            └── mapping_relations.json
```

---

## Quickstart

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run via notebook
```bash
jupyter notebook mapping_pipeline.ipynb
```

### Run via CLI
```bash
# Steps 1–3 only (no LLM, no API key needed)
python referential_mapping/pipeline.py \
  --ref-a files/surveys/cis-controls-v8-1.yaml --adapter-a cis_v8 \
  --ref-b files/surveys/nistCsfV2.yaml          --adapter-b nist_csf_v2 \
  --skip-step4

# Full pipeline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python referential_mapping/pipeline.py \
  --ref-a files/surveys/cis-controls-v8-1.yaml --adapter-a cis_v8 \
  --ref-b files/surveys/nistCsfV2.yaml          --adapter-b nist_csf_v2
```

### Cache & re-runs
```bash
--force-step1   # re-parse source files
--force-step2   # recompute embeddings
--force-step4   # re-score with LLM
```

---

## Configuration

All thresholds live in `referential_mapping/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `SEMANTIC_THRESHOLD` | `0.40` | Min cosine score to keep a pair (Step 2) |
| `TOP_K` | `5` | Max candidates per requirement (Step 2) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM used for scoring (Step 4) |
| `LLM_CONFIRM_THRESHOLD` | `0.75` | Min LLM score to trigger double verification |
| `LLM_CONCURRENCY` | `5` | Parallel LLM calls (asyncio semaphore) |

---

## Adding a new framework

1. Add your YAML file to `files/surveys/`
2. Create an adapter in `referential_mapping/ingestion/adapters/your_framework.py` — implement a `load(path) -> list[RequirementNormalized]` function
3. Register the adapter alias in `pipeline.py` (`ADAPTER_ALIASES`)
4. Run the pipeline pointing to your new files

---

*by Sensey*

"""
notebook.py — Sandbox interactif RiskHunter (Marimo)

Outil développeur pour tester et explorer la pipeline de mapping référentiels
sans passer par l'interface Streamlit.

Périmètre intentionnellement limité :
- Pas d'export Excel/HTML (Streamlit gère ça)
- Pas de mode multi-framework all-pairs (Streamlit gère ça)
- Pas de Step 5 cleanup (optionnel selon le contexte de test)
- Focalisé sur : exploration des scores, seuils adaptatifs, sample LLM

Usage :
    marimo edit notebook.py     # mode interactif (navigateur)
    marimo run  notebook.py     # mode lecture seule
"""

import marimo as mo

app = mo.App(width="full")


# ── Cellule 1 : Imports & initialisation ──────────────────────────────────────

@app.cell
def _imports():
    import sys
    import os
    from pathlib import Path
    import numpy as np

    # S'assurer que le répertoire racine est dans le path
    ROOT = Path(__file__).parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import pipeline_standalone as pl

    header = mo.md("## 🗺️ RiskHunter — Notebook de test pipeline").center()
    return ROOT, np, os, pl, sys, header


# ── Cellule 2 : Sélection des référentiels ────────────────────────────────────

@app.cell
def _framework_selector(ROOT, mo=mo):
    import glob as _glob
    from pathlib import Path as _Path

    yaml_files = sorted(_glob.glob(str(ROOT / "files" / "surveys" / "*.yaml")))
    yaml_names = [_Path(f).name for f in yaml_files]
    yaml_map   = dict(zip(yaml_names, yaml_files))

    ref_a_sel = mo.ui.dropdown(
        options=yaml_names,
        value=yaml_names[0] if yaml_names else None,
        label="Référentiel A",
    )
    ref_b_sel = mo.ui.dropdown(
        options=yaml_names,
        value=yaml_names[1] if len(yaml_names) > 1 else (yaml_names[0] if yaml_names else None),
        label="Référentiel B",
    )

    selectors = mo.hstack([ref_a_sel, ref_b_sel], gap=2)
    return ref_a_sel, ref_b_sel, yaml_map, selectors


# ── Cellule 3 : Paramètres Step 2 ─────────────────────────────────────────────

@app.cell
def _step2_params(pl, mo=mo):
    sem_thresh = mo.ui.slider(
        start=0.20, stop=0.80, step=0.01,
        value=pl.SEMANTIC_THRESHOLD,
        label=f"Seuil sémantique (défaut : {pl.SEMANTIC_THRESHOLD})",
    )
    top_k_sl = mo.ui.slider(
        start=1, stop=15, step=1,
        value=pl.TOP_K,
        label=f"Top-K (défaut : {pl.TOP_K})",
    )
    params_ui = mo.hstack([sem_thresh, top_k_sl], gap=2)
    return sem_thresh, top_k_sl, params_ui


# ── Cellule 4 : Step 1 — Ingestion ────────────────────────────────────────────

@app.cell
def _step1(ref_a_sel, ref_b_sel, yaml_map, pl, mo=mo):
    import pandas as pd

    mo.stop(
        not ref_a_sel.value or not ref_b_sel.value,
        mo.md("⚠️ Sélectionne deux référentiels ci-dessus."),
    )

    path_a = yaml_map[ref_a_sel.value]
    path_b = yaml_map[ref_b_sel.value]

    reqs_a = pl.run_ingestion(path_a, "generic")
    reqs_b = pl.run_ingestion(path_b, "generic")

    fw_a = reqs_a[0].framework if reqs_a else ref_a_sel.value
    fw_b = reqs_b[0].framework if reqs_b else ref_b_sel.value

    df_a = pd.DataFrame([{"ID": r.id, "Titre": r.title, "Tags": ", ".join(r.tags[:3])} for r in reqs_a])
    df_b = pd.DataFrame([{"ID": r.id, "Titre": r.title, "Tags": ", ".join(r.tags[:3])} for r in reqs_b])

    step1_summary = mo.md(f"""
### Étape 1 — Ingestion
| | Référentiel | Exigences |
|--|--|--|
| **A** | `{fw_a}` | {len(reqs_a)} |
| **B** | `{fw_b}` | {len(reqs_b)} |
""")
    step1_tabs = mo.ui.tabs({
        f"A — {fw_a} ({len(reqs_a)})": mo.ui.table(df_a, pagination=True),
        f"B — {fw_b} ({len(reqs_b)})": mo.ui.table(df_b, pagination=True),
    })
    step1_ui = mo.vstack([step1_summary, step1_tabs])

    return reqs_a, reqs_b, fw_a, fw_b, step1_ui


# ── Cellule 5 : Step 2 — Similarité sémantique ────────────────────────────────

@app.cell
def _step2(reqs_a, reqs_b, fw_a, fw_b, sem_thresh, top_k_sl, pl, np, mo=mo):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # backend non-interactif pour Marimo
    import matplotlib.pyplot as plt

    mo.stop(not reqs_a or not reqs_b, mo.md("⚠️ Attends la fin de l'ingestion (Step 1)."))

    candidates, matrix = pl.run_similarity(
        reqs_a, reqs_b,
        semantic_threshold=sem_thresh.value,
        top_k=top_k_sl.value,
        fw_a_name=fw_a,
        fw_b_name=fw_b,
    )

    scores = [p.semantic_score for p in candidates]

    # Histogramme des scores sémantiques avec mean/median
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(scores, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(scores),   color="#e74c3c", lw=1.5, linestyle="--", label=f"mean={np.mean(scores):.3f}")
    ax.axvline(np.median(scores), color="#2ecc71", lw=1.5, linestyle="--", label=f"median={np.median(scores):.3f}")
    ax.set_xlabel("Score cosinus")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution des scores sémantiques ({len(scores)} paires)")
    ax.legend()
    plt.tight_layout()

    df_pairs = pd.DataFrame([{
        "Score": round(p.semantic_score, 4),
        "ID A": p.id_A, "Titre A": p.title_A,
        "ID B": p.id_B, "Titre B": p.title_B,
    } for p in candidates[:100]])

    total_comb = len(reqs_a) * len(reqs_b)
    pct = 100 * len(candidates) / max(total_comb, 1)
    step2_summary = mo.md(f"""
### Étape 2 — Similarité sémantique
{len(candidates)} paires retenues sur {total_comb} combinaisons (**{pct:.1f}%**).
*(seuil={sem_thresh.value}, top_k={top_k_sl.value})*
""")
    step2_ui = mo.vstack([
        step2_summary,
        mo.as_html(fig),
        mo.md("#### Top 100 paires candidates"),
        mo.ui.table(df_pairs, pagination=True),
    ])

    return candidates, scores, step2_ui


# ── Cellule 6 : Seuils adaptatifs ─────────────────────────────────────────────

@app.cell
def _adaptive_thresholds(candidates, scores, pl, mo=mo):
    mo.stop(not candidates, mo.md("⚠️ Lance d'abord Step 2."))

    thresh_eq, thresh_cov, stats = pl._compute_adaptive_thresholds(scores)

    rows = [
        ("n (paires candidates)",                              str(stats["n"])),
        ("mean",                                               f"{stats['mean']:.4f}"),
        ("médiane",                                            f"{stats['median']:.4f}"),
        ("Q25",                                                f"{stats['q25']:.4f}"),
        ("Q75",                                                f"{stats['q75']:.4f}"),
        ("P90",                                                f"{stats['p90']:.4f}"),
        ("delta appliqué",                                     f"{stats['delta']:+.4f}"),
        ("",                                                   ""),
        (f"THRESHOLD_EQUIVALENCE (défaut={pl.THRESHOLD_EQUIVALENCE})", f"**{thresh_eq}**"),
        (f"THRESHOLD_COVERAGE    (défaut={pl.THRESHOLD_COVERAGE})",    f"**{thresh_cov}**"),
    ]
    md_rows = "\n".join(f"| {k} | {v} |" for k, v in rows)

    adaptive_ui = mo.md(f"""
### Seuils adaptatifs (post-Step 2)

| Indicateur | Valeur |
|---|---|
{md_rows}

> **Principe :** si médiane > **{pl._ADAPTIVE_BASELINE_MEDIAN}** (corpus dense),
> les seuils sont relevés pour éviter les faux positifs "equivalence".
> Ajustement borné à ±{pl._ADAPTIVE_MAX_DELTA}.
""")

    return thresh_eq, thresh_cov, stats, adaptive_ui


# ── Cellule 7 : Paramètres Step 4 ─────────────────────────────────────────────

@app.cell
def _step4_params(mo=mo):
    n_sample = mo.ui.slider(
        start=5, stop=50, step=5,
        value=10,
        label="Nombre de paires à scorer (sample LLM)",
    )
    run_llm_btn = mo.ui.run_button(label="▶ Lancer le scoring LLM (sample)")

    step4_header = mo.md("""
### Étape 4 — Scoring LLM (sample)
⚠️ Nécessite `OPENAI_API_KEY` dans `.env`.
Les seuils adaptatifs calculés à l'étape précédente sont automatiquement utilisés.
""")
    step4_controls = mo.hstack([n_sample, run_llm_btn], gap=2)
    return n_sample, run_llm_btn, step4_header, step4_controls


# ── Cellule 8 : Step 4 — Scoring LLM sur sample ───────────────────────────────

@app.cell
def _step4(run_llm_btn, n_sample, candidates, reqs_a, reqs_b, thresh_eq, thresh_cov, pl, mo=mo):
    import pandas as pd
    from collections import Counter

    mo.stop(not run_llm_btn.value, mo.md("*Clique sur le bouton ci-dessus pour lancer le scoring LLM.*"))
    mo.stop(not candidates, mo.md("⚠️ Lance d'abord Step 2."))

    # Scorer uniquement les N premières paires (déjà triées par score sémantique desc)
    sample = candidates[:n_sample.value]

    relations = pl.run_scorer(
        sample,
        reqs_a,
        reqs_b,
        force=True,           # toujours recalculer en mode notebook (pas de cache)
        thresh_eq=thresh_eq,
        thresh_cov=thresh_cov,
    )

    dist = Counter(r.relation_type for r in relations)
    dist_md = " | ".join(f"**{k}** : {v}" for k, v in sorted(dist.items()))

    df_rel = pd.DataFrame([{
        "relation_type":  r.relation_type,
        "cov A→B":        round(r.coverage_A_to_B, 3),
        "cov B→A":        round(r.coverage_B_to_A, 3),
        "confidence":     round(r.confidence, 3),
        "sem_score":      round(r.semantic_score, 3),
        "ID A":           r.id_A,
        "ID B":           r.id_B,
        "Justification":  (r.justification or "")[:120],
    } for r in relations])

    step4_ui = mo.vstack([
        mo.md(f"**{len(relations)} relations scorées** — {dist_md}"),
        mo.ui.table(df_rel, pagination=True),
    ])

    return relations, step4_ui


if __name__ == "__main__":
    app.run()



# ── Cellule 2 : Sélection des référentiels ────────────────────────────────────

@app.cell
def _framework_selector(ROOT, pl):
    import glob as _glob

    yaml_files = sorted(_glob.glob(str(ROOT / "files" / "surveys" / "*.yaml")))
    yaml_names = [Path(f).name for f in yaml_files]
    yaml_map   = dict(zip(yaml_names, yaml_files))

    ref_a_sel = mo.ui.dropdown(
        options=yaml_names,
        value=yaml_names[0] if yaml_names else None,
        label="Référentiel A",
    )
    ref_b_sel = mo.ui.dropdown(
        options=yaml_names,
        value=yaml_names[1] if len(yaml_names) > 1 else yaml_names[0] if yaml_names else None,
        label="Référentiel B",
    )

    mo.hstack([ref_a_sel, ref_b_sel], gap=2)
    return ref_a_sel, ref_b_sel, yaml_map


# ── Cellule 3 : Paramètres Step 2 ─────────────────────────────────────────────

@app.cell
def _step2_params(pl):
    sem_thresh = mo.ui.slider(
        start=0.20, stop=0.80, step=0.01,
        value=pl.SEMANTIC_THRESHOLD,
        label=f"Seuil sémantique (défaut : {pl.SEMANTIC_THRESHOLD})",
    )
    top_k_sl = mo.ui.slider(
        start=1, stop=15, step=1,
        value=pl.TOP_K,
        label=f"Top-K (défaut : {pl.TOP_K})",
    )
    mo.hstack([sem_thresh, top_k_sl], gap=2)
    return sem_thresh, top_k_sl


# ── Cellule 4 : Step 1 — Ingestion ────────────────────────────────────────────

@app.cell
def _step1(ref_a_sel, ref_b_sel, yaml_map, pl):
    import pandas as pd

    if not ref_a_sel.value or not ref_b_sel.value:
        return mo.md("⚠️ Sélectionne deux référentiels ci-dessus.")

    path_a = yaml_map[ref_a_sel.value]
    path_b = yaml_map[ref_b_sel.value]

    reqs_a = pl.run_ingestion(path_a, "generic")
    reqs_b = pl.run_ingestion(path_b, "generic")

    fw_a = reqs_a[0].framework if reqs_a else ref_a_sel.value
    fw_b = reqs_b[0].framework if reqs_b else ref_b_sel.value

    df_a = pd.DataFrame([{"ID": r.id, "Titre": r.title, "Tags": ", ".join(r.tags[:3])} for r in reqs_a])
    df_b = pd.DataFrame([{"ID": r.id, "Titre": r.title, "Tags": ", ".join(r.tags[:3])} for r in reqs_b])

    summary = mo.md(f"""
### Étape 1 — Ingestion
| | Référentiel | Exigences |
|--|--|--|
| **A** | `{fw_a}` | {len(reqs_a)} |
| **B** | `{fw_b}` | {len(reqs_b)} |
""")

    tabs = mo.ui.tabs({
        f"A — {fw_a} ({len(reqs_a)})": mo.ui.table(df_a, pagination=True),
        f"B — {fw_b} ({len(reqs_b)})": mo.ui.table(df_b, pagination=True),
    })

    return mo.vstack([summary, tabs])
    return reqs_a, reqs_b, fw_a, fw_b


# ── Cellule 5 : Step 2 — Similarité sémantique ────────────────────────────────

@app.cell
def _step2(reqs_a, reqs_b, fw_a, fw_b, sem_thresh, top_k_sl, pl, np):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # backend non-interactif pour Marimo
    import matplotlib.pyplot as plt

    if not reqs_a or not reqs_b:
        return mo.md("⚠️ Attends la fin de l'ingestion (Step 1).")

    candidates, matrix = pl.run_similarity(
        reqs_a, reqs_b,
        semantic_threshold=sem_thresh.value,
        top_k=top_k_sl.value,
        fw_a_name=fw_a,
        fw_b_name=fw_b,
    )

    scores = [p.semantic_score for p in candidates]

    # Histogramme des scores sémantiques
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(scores, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(scores),   color="#e74c3c", lw=1.5, linestyle="--", label=f"mean={np.mean(scores):.3f}")
    ax.axvline(np.median(scores), color="#2ecc71", lw=1.5, linestyle="--", label=f"median={np.median(scores):.3f}")
    ax.set_xlabel("Score cosinus")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Distribution des scores sémantiques ({len(scores)} paires)")
    ax.legend()
    plt.tight_layout()

    # Tableau des top 50 paires
    df_pairs = pd.DataFrame([{
        "Score": round(p.semantic_score, 4),
        "ID A": p.id_A, "Titre A": p.title_A,
        "ID B": p.id_B, "Titre B": p.title_B,
    } for p in candidates[:50]])

    total_comb = len(reqs_a) * len(reqs_b)
    pct = 100 * len(candidates) / max(total_comb, 1)
    summary = mo.md(f"""
### Étape 2 — Similarité sémantique
{len(candidates)} paires retenues sur {total_comb} combinaisons possibles (**{pct:.1f}%**).
*(seuil={sem_thresh.value}, top_k={top_k_sl.value})*
""")

    return mo.vstack([
        summary,
        mo.as_html(fig),
        mo.md("#### Top 50 paires candidates"),
        mo.ui.table(df_pairs, pagination=True),
    ])
    return candidates, scores


# ── Cellule 6 : Seuils adaptatifs ─────────────────────────────────────────────

@app.cell
def _adaptive_thresholds(candidates, scores, pl):
    if not candidates:
        return mo.md("⚠️ Lance d'abord Step 2.")

    thresh_eq, thresh_cov, stats = pl._compute_adaptive_thresholds(scores)

    rows = [
        ("n (paires candidates)", stats["n"]),
        ("mean", stats["mean"]),
        ("médiane", stats["median"]),
        ("Q25", stats["q25"]),
        ("Q75", stats["q75"]),
        ("P90", stats["p90"]),
        ("delta appliqué", f"{stats['delta']:+.4f}"),
        ("", ""),
        (f"THRESHOLD_EQUIVALENCE (défaut={pl.THRESHOLD_EQUIVALENCE})", f"**{thresh_eq}**"),
        (f"THRESHOLD_COVERAGE    (défaut={pl.THRESHOLD_COVERAGE})",    f"**{thresh_cov}**"),
    ]

    md_rows = "\n".join(f"| {k} | {v} |" for k, v in rows)

    return mo.md(f"""
### Seuils adaptatifs (post-Step 2)

| Indicateur | Valeur |
|---|---|
{md_rows}

> **Principe :** si la médiane des scores est **> {pl._ADAPTIVE_BASELINE_MEDIAN}** (corpus dense),
> les seuils sont relevés pour éviter les faux positifs "equivalence".
> Ajustement borné à ±{pl._ADAPTIVE_MAX_DELTA}.
""")
    return thresh_eq, thresh_cov, stats


# ── Cellule 7 : Paramètres Step 4 ─────────────────────────────────────────────

@app.cell
def _step4_params(pl):
    n_sample = mo.ui.slider(
        start=5, stop=50, step=5,
        value=10,
        label="Nombre de paires à scorer (sample LLM)",
    )
    run_llm_btn = mo.ui.run_button(label="▶ Lancer le scoring LLM (sample)")

    mo.md("""
### Étape 4 — Scoring LLM (sample)
⚠️ Nécessite `OPENAI_API_KEY` dans `.env`. Utilise les seuils adaptatifs calculés ci-dessus.
""")
    mo.hstack([n_sample, run_llm_btn], gap=2)
    return n_sample, run_llm_btn


# ── Cellule 8 : Step 4 — Scoring LLM sur sample ───────────────────────────────

@app.cell
def _step4(run_llm_btn, n_sample, candidates, reqs_a, reqs_b, thresh_eq, thresh_cov, pl):
    import pandas as pd

    if not run_llm_btn.value:
        return mo.md("*Clique sur le bouton ci-dessus pour lancer le scoring LLM.*")

    if not candidates:
        return mo.md("⚠️ Lance d'abord Step 2.")

    # Scorer uniquement les N premières paires (tri par score sémantique desc)
    sample = candidates[:n_sample.value]

    relations = pl.run_scorer(
        sample,
        reqs_a,
        reqs_b,
        force=True,   # toujours recalculer en mode notebook
        thresh_eq=thresh_eq,
        thresh_cov=thresh_cov,
    )

    # Distribution des relation_type
    from collections import Counter
    dist = Counter(r.relation_type for r in relations)

    dist_md = " | ".join(f"**{k}** : {v}" for k, v in sorted(dist.items()))

    df_rel = pd.DataFrame([{
        "relation_type":  r.relation_type,
        "cov A→B":        round(r.coverage_A_to_B, 3),
        "cov B→A":        round(r.coverage_B_to_A, 3),
        "confidence":     round(r.confidence, 3),
        "sem_score":      round(r.semantic_score, 3),
        "ID A":           r.id_A,
        "ID B":           r.id_B,
        "Justification":  r.justification or "",
    } for r in relations])

    return mo.vstack([
        mo.md(f"**{len(relations)} relations scorées** — {dist_md}"),
        mo.ui.table(df_rel, pagination=True),
    ])


if __name__ == "__main__":
    app.run()

# Mapping Pipeline — Plan d'implémentation (révisé)

> **Objectif** : pipeline automatisé et générique de mise en correspondance entre deux référentiels de sécurité (ex. CIS v8 ↔ NIST CSF v2 en MVP, mais conçu pour fonctionner avec n'importe quel couple de référentiels).

---

## Principe général

Le pipeline suit une logique d'entonnoir en 4 étapes :

```
Référentiel A + Référentiel B
        │
  [Étape 1] Structuration & normalisation
        │
  [Étape 2] Recherche sémantique sur titres → éliminer ~99% des combinaisons inutiles
        │
  [Étape 3] Matrice de corrélation (visuel) → confirmer les couples pertinents
        │
  [Étape 4] Vérification intelligente LLM (titre + description) → scorer le 1% restant
        │
  Export Excel clair (Ref A ↔ Ref B avec scores)
```

- **Étapes 2 & 3** = filtrage massif (pas de LLM, 0 coût, rapide)
- **Étape 4** = vérification intelligente uniquement sur les paires pertinentes (LLM sur titre + description)

---

## Stack & outils

| Rôle | Outil |
|---|---|
| Langage | Python 3.11+ |
| Parsing Excel/CSV/JSON | `pandas` |
| Embeddings locaux | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Similarité cosinus | `numpy` / `scipy.spatial.distance` |
| Visualisation matrice | `seaborn` + `matplotlib` |
| Appels LLM | `openai` (ou `mistralai`) — **étape 4 uniquement** |
| Validation LLM output | `pydantic` v2 |
| Export Excel | `openpyxl` |

---

## Modèles de données

```python
# Exigence normalisée (générique — fonctionne pour tout référentiel)
RequirementNormalized = {
    "id":          str,   # ex. "CIS-1.1" | "NIST-ID.AM-01"
    "framework":   str,   # ex. "CIS_v8" | "NIST_CSF_v2"
    "title":       str,   # champ principal pour étapes 2 & 3
    "description": str,   # utilisé uniquement à l'étape 4
    "tags":        list[str]  # optionnel — utilisé si dispo pour filtrage préalable
}

# Résultat de mapping final
MappingRelation = {
    "id_A":            str,
    "title_A":         str,
    "id_B":            str,
    "title_B":         str,
    "semantic_score":  float,  # score cosinus (étapes 2 & 3)
    "coverage_A_to_B": float,  # scoring LLM (étape 4)
    "coverage_B_to_A": float,
    "confidence":      float,
    "relation_type":   str,    # "equivalence" | "A_couvre_B" | "B_couvre_A" | "partielle" | "aucun_lien"
}
```

---

## Étape 1 — Structuration & normalisation

> **But** : produire un dataset propre et exploitable. Chaque exigence = une ligne avec `id`, `title`, `description`, `tags`.

- **Input** : fichiers sources des référentiels (Excel, CSV, JSON, YAML)
- **Actions** :
  - Parser les fichiers sources avec `pandas`
  - Joindre les colonnes tags si présentes (ex. `join` sur colonne tags CIS)
  - Garder uniquement les colonnes utiles : `id`, `title`, `description`, `tags`
  - Drop `NaN` sur `id` et `title`
  - Dédupliquer sur `id`
  - Normaliser les chaînes (strip, lowercase pour comparaison)
- **Output** : `data/ref_A_normalized.json` + `data/ref_B_normalized.json`
- **Structure de sortie** : liste de `RequirementNormalized`

> ⚠️ Le join des tags doit être fait **ici**, avant tout traitement. L'étape 1 est le seul endroit où on touche à la structure brute des données.

---

## Étape 2 — Recherche de proximité sémantique sur les titres

> **But** : sur N × M combinaisons possibles, ne conserver que les paires dont les titres sont sémantiquement proches. **Éliminer ~99% des combinaisons inutiles.**

- **Texte encodé** : `title` uniquement (pas de description ici — rapide et sans coût LLM)
- **Modèle** : `all-MiniLM-L6-v2` (local, gratuit)
- **Stratégie** : pour chaque exigence du Ref A, calculer la similarité cosinus avec **toutes** les exigences du Ref B, puis garder le **top-k** (k à calibrer, ex. k=5)
- **Seuil cosinus** : conserver les paires avec score ≥ seuil configurable (ex. 0.50 — intentionnellement bas pour ne rien perdre à ce stade)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

emb_A = model.encode([r["title"] for r in ref_A])
emb_B = model.encode([r["title"] for r in ref_B])

# Normalisation pour produit scalaire = cosinus
emb_A_norm = emb_A / np.linalg.norm(emb_A, axis=1, keepdims=True)
emb_B_norm = emb_B / np.linalg.norm(emb_B, axis=1, keepdims=True)

similarity_matrix = emb_A_norm @ emb_B_norm.T  # shape: (n_A, n_B)
```

- **Output** : liste de paires candidates `(id_A, id_B, semantic_score)` — fichier `data/candidate_pairs.json`

---

## Étape 3 — Matrice de corrélation (visuel / indicatif)

> **But** : visualiser les similarités entre exigences des deux référentiels (à la manière d'une heatmap de features). Sert de diagnostic visuel pour valider la pertinence des paires retenues à l'étape 2.

- **Input** : `similarity_matrix` calculée à l'étape 2
- **Visualisation** : heatmap via `seaborn.heatmap` (exigences Ref A en lignes, Ref B en colonnes)
- **Utilité** : purement indicatif — permet de détecter des clusters de corrélation et des zones vides

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 16))
sns.heatmap(similarity_matrix, xticklabels=[r["id"] for r in ref_B],
            yticklabels=[r["id"] for r in ref_A], cmap="YlOrRd")
plt.title("Matrice de similarité sémantique (titres)")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png", dpi=150)
```

- **Output** : `outputs/correlation_matrix.png`

> Les étapes 2 & 3 ensemble constituent le **filtre massif** : seules les paires avec un bon score sémantique sur les titres passent à l'étape 4.

---

## Étape 4 — Vérification intelligente LLM (titre + description)

> **But** : confirmer ou invalider les corrélations détectées aux étapes 2 & 3, en utilisant cette fois **titre + description** pour plus de précision. **LLM uniquement à partir de cette étape.**

### 4a — Sélection des paires à soumettre au LLM

- Filtrer les paires candidates (étape 2) au-dessus d'un seuil de score sémantique (ex. top-k ou score ≥ 0.60)
- Ce sont les ~1% de combinaisons pertinentes à vérifier intelligemment

### 4b — Scoring LLM (boucle sur les paires retenues)

- **Modèle recommandé** : `gpt-4o-mini`
- **Texte soumis** : `title + description` pour chaque exigence de la paire
- **Double vérification** : pour les paires avec score LLM élevé, effectuer un second appel LLM de confirmation (double vérif.)
- **Forcer JSON** : `response_format={"type": "json_object"}`
- **Retries** : max 2 en cas d'échec parsing
- **Validation** : `pydantic` v2

```python
from pydantic import BaseModel, Field
from typing import Literal

class LLMScoringOutput(BaseModel):
    coverage_A_to_B: float = Field(ge=0.0, le=1.0)
    coverage_B_to_A: float = Field(ge=0.0, le=1.0)
    confidence:      float = Field(ge=0.0, le=1.0)
    relation_type:   Literal["equivalence", "A_couvre_B", "B_couvre_A", "partielle", "aucun_lien"]
```

**Règles relation_type :**

| Condition | relation_type |
|---|---|
| A→B ≥ 0.8 ET B→A ≥ 0.8 | `equivalence` |
| A→B ≥ 0.8 ET B→A < 0.8 | `A_couvre_B` |
| B→A ≥ 0.8 ET A→B < 0.8 | `B_couvre_A` |
| A→B ≥ 0.4 OU B→A ≥ 0.4 | `partielle` |
| Sinon | `aucun_lien` |

### 4c — Diagnostic & export

- **Diagnostic** : à l'issue de la boucle LLM, comparer les scores sémantiques (étape 2) avec les scores LLM (étape 4) pour confirmer ou invalider les corrélations détectées
- **Export Excel** (`openpyxl`) : fichier clair montrant explicitement les corrélations

| Colonne | Description |
|---|---|
| `id_A` | Identifiant exigence Ref A |
| `title_A` | Titre exigence Ref A |
| `id_B` | Identifiant exigence Ref B |
| `title_B` | Titre exigence Ref B |
| `semantic_score` | Score cosinus (étapes 2 & 3) |
| `coverage_A_to_B` | Score LLM A→B |
| `coverage_B_to_A` | Score LLM B→A |
| `confidence` | Confiance LLM |
| `relation_type` | Type de relation |

> Exemple de lecture : *"L'exigence 1 du Ref A est bel et bien très corrélée à l'exigence 6 du Ref B (score: 0.87, equivalence)"*

- **Output** : `outputs/mapping_results.xlsx` + `outputs/mapping_relations.json`

---

## Structure du projet

```
referential_mapping/
├── ingestion/          # étape 1 — parsing & normalisation
├── semantic/           # étape 2 — embeddings + similarité cosinus sur titres
├── visualization/      # étape 3 — matrice de corrélation (heatmap)
├── llm/                # étape 4 — scoring LLM + validation Pydantic + double vérif
├── config.py           # seuils configurables
└── data/
    ├── ref_A_normalized.json
    ├── ref_B_normalized.json
    ├── candidate_pairs.json    # output étape 2
    └── outputs/
        ├── correlation_matrix.png
        ├── mapping_results.xlsx
        └── mapping_relations.json
```

---

## Config (seuils)

```python
# config.py — tous les seuils sont configurables ici
SEMANTIC_THRESHOLD      = 0.50   # score cosinus min pour conserver une paire (étape 2)
TOP_K                   = 5      # nb candidats Ref B par exigence Ref A (étape 2)
LLM_CONFIRM_THRESHOLD   = 0.75   # score LLM min pour déclencher la double vérification
LLM_MAX_RETRIES         = 2
EMBEDDING_MODEL         = "all-MiniLM-L6-v2"
LLM_MODEL               = "gpt-4o-mini"
```

---

## Points d'attention

1. **Généricité** : le pipeline doit fonctionner pour n'importe quel couple de référentiels — ne pas hardcoder de logique propre à CIS ou NIST (ex. le filtrage par tags CIS↔NIST de l'ancienne version est supprimé car non générique)
2. **LLM uniquement à l'étape 4** : les étapes 2 & 3 sont 100% locales, sans appel API
3. **Ne pas relancer l'étape 2** si `candidate_pairs.json` existe déjà (cache)
4. **`coverage_A_to_B ≠ coverage_B_to_A`** — ne jamais moyenner ces scores
5. **Batch async** pour les appels LLM (étape 4) — utiliser `asyncio` + semaphore pour limiter la concurrence
6. **Double vérification LLM** uniquement sur les paires à score élevé — éviter les appels inutiles

---

*RiskHunter — MVP CIS v8 ↔ NIST CSF v2 — plan révisé*

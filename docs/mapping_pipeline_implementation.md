# Mapping Pipeline — Guide d'implémentation

**MVP : CIS v8 ↔ NIST CSF v2**

---

## Stack & outils

| Rôle | Outil |
|---|---|
| Langage | Python 3.11+ |
| Parsing Excel/CSV/JSON | `pandas` |
| Embeddings locaux | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Embeddings API | `openai` — `text-embedding-3-small` |
| Similarité cosinus | `numpy` + `scipy.spatial.distance` |
| Index vectoriel (>3 500 vecteurs) | `faiss-cpu` |
| Validation LLM output | `pydantic` v2 |
| Appels LLM | `openai` (ou `mistralai`) |
| Propagation / matrice adjacence | `scipy.sparse` |
| Export Excel | `openpyxl` |
| Export graphe | `networkx` (GraphML) |

---

## Modèles de données

```python
# Chaque exigence normalisée
RequirementNormalized = {
    "id":          str,       # "CIS-1.1" | "NIST-ID.AM-01"
    "framework":   str,       # "CIS_v8" | "NIST_CSF_v2"
    "chapter":     str,
    "title":       str,
    "description": str,       # enrichi par LLM pour NIST (voir étape 2)
    "tags":        list[str]  # ["Identifier", "IG1", "IG2"]
}

# Relation de mapping entre deux exigences
MappingRelation = {
    "id_A":            str,
    "id_B":            str,
    "coverage_A_to_B": float,   # P(B satisfait | A implémenté)
    "coverage_B_to_A": float,
    "confidence":      float,
    "relation_type":   str,     # voir règles niveau 3
    # "source":          str,     # "llm" | "human_validated"
    "auto":            bool
}
```

---

## Pipeline en 7 étapes

### Étape 1 — Parsing & normalisation

- **Input** : fichiers sources CIS (Excel multi-onglets) + NIST (YAML/JSON/CSV)
- **Librairies** : `pandas`
- **Actions** :
  - Parser chaque feuille Excel avec `pd.read_excel(sheet_name=None)`
  - Drop `NaN` sur les colonnes `id`, `title`
  - Dédupliquer sur `id`
  - Produire une liste de `RequirementNormalized`
- **Output** : `data/cis_normalized.json` + `data/nist_normalized.json`

---

### Étape 2 — Enrichissement NIST (one-shot)

> NIST CSF v2 : 106 sous-catégories avec uniquement un `label` court, pas de `description`.

- **~106 appels LLM**, à faire **une seule fois**, résultat stocké définitivement
- **Prompt** : demander une description de 2–3 phrases pour chaque sous-catégorie NIST
- **Forcer JSON** : `response_format={"type": "json_object"}` (OpenAI)
- **Output** : `data/nist_enriched.json` — ne jamais réexécuter si le fichier existe

```python
if not Path("data/nist_enriched.json").exists():
    enrich_nist_descriptions(nist_normalized)
```

---

### Étape 3 — Filtrage par taxonomie (Niveau 1) (WARNING = ne fonctionnera pas sur toutes les exigences, propre uniquement au NIST et CIS)

> Objectif : passer de 16 218 paires brutes à ~2 500 candidates.

**MVP** : utiliser les tags CIS natifs (déjà alignés sur les fonctions NIST) :

| Tag CIS | Fonction NIST |
|---|---|
| `Gouverner` | GV |
| `Identifier` | ID |
| `Protéger` | PR |
| `Détecter` | DE |
| `Répondre` | RS |
| `Rétablir` | RC |

- Regrouper les exigences CIS par tag NIST
- Ne comparer une exigence CIS qu'avec les sous-catégories NIST de sa fonction
- **Output** : liste de paires `(cis_id, nist_id)` candidate

---

### Étape 4 — Vectorisation & ranking (Niveau 2)

> Objectif : passer de ~2 500 paires à ~450.

- **Texte encodé** : `title + ". " + description`
- **Modèle** : `all-MiniLM-L6-v2` (local, gratuit) ou `text-embedding-3-small` (API, ~$0.02)
- **Stratégie** : pour chaque exigence source, garder le **top-k=10** (revoir la valeur du k = eviter la perte d'info) par score cosinus (pas de seuil fixe — le LLM tranche au niveau 3), à voir si on ne peut pas optimiser (le principe c'est qu'on a pas de valeur fixe connue à la base)
- **Infrastructure** : `numpy` suffit jusqu'à ~3 500 vecteurs ; passer à `faiss` au-delà

**Notes:**
- Recherche / proximité sémantique avec les "titres" d'abord et voir 
- Réduit drastiquement le cout algo (pas besoin d'enrichissement)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([req["title"] + ". " + req["description"] for req in reqs])
# cosine similarity : normalize puis dot product
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_norm = embeddings / norms
similarity_matrix = embeddings_norm @ embeddings_norm.T
```

- **Output** : liste de paires filtrées avec leur score cosinus

---

### Étape 5 — Scoring LLM (Niveau 3)

> Objectif : évaluer la couverture dirigée pour chaque paire candidate.

- **Modèle recommandé** : `gpt-4o-mini` (bon rapport qualité/coût)
- **Forcer JSON** : `response_format={"type": "json_object"}`
- **Retries** : max 2 en cas d'échec de parsing
- **Validation** : `pydantic` v2

**Schema Pydantic :**

```python
from pydantic import BaseModel, Field
from typing import Literal

class LLMScoringOutput(BaseModel):
    coverage_A_to_B: float = Field(ge=0.0, le=1.0)
    coverage_B_to_A: float = Field(ge=0.0, le=1.0)
    confidence:      float = Field(ge=0.0, le=1.0)
    relation_type:   Literal["equivalence", "A_couvre_B", "B_couvre_A", "partielle", "aucun_lien"]
```

**Règles de relation_type :**

| Condition | relation_type |
|---|---|
| A→B ≥ 0.8 ET B→A ≥ 0.8 | `equivalence` |
| A→B ≥ 0.8 ET B→A < 0.8 | `A_couvre_B` |
| B→A ≥ 0.8 ET A→B < 0.8 | `B_couvre_A` |
| A→B ≥ 0.4 OU B→A ≥ 0.4 | `partielle` |
| Sinon | `aucun_lien` |

- **Output** : liste de `MappingRelation` avec `"auto": true`, `"source": "llm"`

---

### Étape 6 — Propagation & ICC 

**Propagation de la conformité** (plusieurs safeguards → une sous-catégorie) :

```python
# P(B) = 1 - prod(1 - cov_i_to_B * c(i))  pour tous les safeguards i → B
p_B = 1 - np.prod([1 - rel.coverage_A_to_B * compliance_score[rel.id_A]
                   for rel in relations if rel.id_B == target])
```

**Multi-hop** (δ = 0.9 par saut) :
```
c(C via B) = cov_B_to_C × c(B) × 0.9^hop
```

**ICC — Indice de Couverture Complète** (CIS entier → chaque sous-catégorie NIST) :
```python
icc_B = 1 - np.prod([1 - rel.coverage_A_to_B for rel in relations if rel.id_B == B])
```

- **Librairie** : `scipy.sparse` pour la matrice d'adjacence
- **Output** : `icc_scores.json` + `compliance_matrix.csv`(ou matrice de corrélation)

**Notes :**
- Matrice de Corrélation des référentiels 

---

### Étape 7 — Export

| Fichier | Format | Librairie | Contenu |
|---|---|---|---|
| `mapping_relations.json` | JSON | built-in | Toutes les `MappingRelation` |
| `compliance_matrix.xlsx` | Excel | `openpyxl` | ICC par sous-catégorie NIST (CIS entier / IG1 / IG1+IG2) |
| `matrice_de_corrélation` | IMG | `openpyxl` | ICC par sous-catégorie NIST (CIS entier / IG1 / IG1+IG2) |
| `graph.graphml` | GraphML | `networkx` | Graphe biparti CIS↔NIST pour Gephi/Neo4j |

---

## Structure du projet

```
referential_mapping/
├── ingestion/          # étape 1 — parsing & normalisation
├── enrichment/         # étape 2 — enrichissement NIST
├── filtering/          # étape 3 — filtrage taxonomie (tags)
├── vectorization/      # étape 4 — embeddings + top-k
├── llm/                # étape 5 — scoring LLM + validation Pydantic
├── propagation/        # étape 6 — propagation + ICC
├── output/             # étape 7 — export JSON/Excel/GraphML
├── config.py           # seuils configurables
└── data/
    ├── cis_normalized.json
    ├── nist_normalized.json
    ├── nist_enriched.json      # généré une fois, ne pas écraser
    └── outputs/
```

---

## Config (seuils)

```python
# config.py
SIMILARITY_THRESHOLD  = 0.65   # cosine min pour garder une paire (niveau 2)
TOP_K                 = 3      # nb candidats NIST par safeguard CIS
PROPAGATION_THRESHOLD = 0.8    # coverage min pour propager la conformité
CONFIDENCE_DECAY      = 0.9    # facteur de décroissance par hop
LLM_MAX_RETRIES       = 2
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"   # ou "text-embedding-3-small"
LLM_MODEL             = "gpt-4o-mini"
```

---

## Points d'attention

1. **Ne jamais écraser une relation `human_validated`** avec une relation auto
2. **`coverage_A_to_B ≠ coverage_B_to_A`** — ne pas moyenner ces scores
3. **`nist_enriched.json`** — vérifier manuellement 10–15% des descriptions avant lancement
4. **Top-k=3** est un compromis rappel/précision — calibrer sur les mappings CIS/ISO existants si disponibles
5. **Batch async** pour les appels LLM (étapes 2 et 5) — utiliser `asyncio` + semaphore pour limiter la concurrence




# Plan réadapté : 
- **1 Structuration:** (étapes join tag à faire avant, garder juste les titres, normaliser = dataset propre et pret à exploiter) / structure de Morgan
- **2 Prendre tous les titres**, faire de la proximité sémantiques sur ces derniers (50 exigences = 50 titres par exemple);
- **3 Faire une matrice de corrélation** (suite à la recherche sématantique) des exigences (àa la manière des features) pour avoir un aperçu des 'similarités' entre exigences (juste à partir des titres toujours) juste à titre visuel / indicatif

- **étape 2 & 3 = VIRER 99% DES COMBNINAISONS INUTILES**
- **étape 4 = VERIFIER DE MANIERE INTELLIGENTE LE 1% RESTANT (avec descriptions pour plus de précision cette fois vu que c'est pertinent)**

- 4 (Confirmation des étapes 2 & 3 LLM = étape 'Intelligente', titre + description cette fois-ci pour checker) automatiser (résultats de l'étape 2 à passer au LLM dans cette étape, pour confirmer la corrélation faites à l'étape 3; étape 3 = visuel exigences similarités, étape 4 = avoir un script qui permet de faire la selection / de boucler sur les exigences et faire le scoring entre les 2 référentiels en entrée. Les couples d'exigences qui ont un bon scoring suite à la matrice de corrélation de l'étape 3 on les garde et on passe à lasuite); puis Faire une boucle factorielle sur les exigences très corrélées (**double vérif. LLM**). A l'issue de cette étape, un diagnostique pour confirmer ou invalider les résultats obtenus lors de notre recherche sémantique (étape 2 & 3); Enfin stocker / exporter les résultats dans un (word, excel, ...) = il faut un support excel clair qui montre clairement les corrélations ("Exigence 1 du ref A est bel et bien très corrélé à l'exigence 6 du ref B")
(- LLM à partir de l'étape 4 uniquement)

- Garder en tete qu'il faut une solution automatisé et générale (fonctionnelle quelque soit alpha)
---

*RiskHunter — MVP CIS v8 ↔ NIST CSF v2*

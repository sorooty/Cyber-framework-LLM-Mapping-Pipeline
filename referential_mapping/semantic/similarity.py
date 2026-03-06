"""
semantic/similarity.py — Étape 2 : recherche de proximité sémantique sur les titres.

- Encode les titres des deux référentiels avec sentence-transformers
- Calcule la matrice de similarité cosinus (n_A × n_B)
- Retourne les paires candidates (top-k par exigence A + seuil min)
- Met en cache candidate_pairs.json et la matrice numpy
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from referential_mapping.models import RequirementNormalized, CandidatePair
from referential_mapping.config import (
    DATA_DIR, EMBEDDING_MODEL, SEMANTIC_THRESHOLD, TOP_K
)

CACHE_PAIRS  = DATA_DIR / "candidate_pairs.json"
CACHE_MATRIX = DATA_DIR / "similarity_matrix.npy"


def run(
    ref_A: list[RequirementNormalized],
    ref_B: list[RequirementNormalized],
    force: bool = False,
) -> tuple[list[CandidatePair], np.ndarray]:
    """
    Retourne (candidate_pairs, similarity_matrix).
    Recharge depuis le cache si disponible (sauf force=True).
    """
    if CACHE_PAIRS.exists() and CACHE_MATRIX.exists() and not force:
        print("[Étape 2] Cache trouvé — rechargement des paires candidates.")
        pairs = _load_pairs()
        matrix = np.load(str(CACHE_MATRIX))
        print(f"  {len(pairs)} paires candidates chargées depuis le cache.")
        return pairs, matrix

    print(f"[Étape 2] Encodage de {len(ref_A)} + {len(ref_B)} titres avec '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    titles_A = [r.title for r in ref_A]
    titles_B = [r.title for r in ref_B]

    emb_A = model.encode(titles_A, show_progress_bar=True, convert_to_numpy=True)
    emb_B = model.encode(titles_B, show_progress_bar=True, convert_to_numpy=True)

    # Normalisation L2 → produit scalaire == cosinus
    emb_A = emb_A / np.linalg.norm(emb_A, axis=1, keepdims=True)
    emb_B = emb_B / np.linalg.norm(emb_B, axis=1, keepdims=True)

    matrix = emb_A @ emb_B.T   # shape (n_A, n_B)

    # Sélection des paires : top-k par ligne ET score >= seuil
    pairs = _select_pairs(ref_A, ref_B, matrix)

    # Cache
    np.save(str(CACHE_MATRIX), matrix)
    _save_pairs(pairs)

    print(f"  {len(pairs)} paires candidates retenues (seuil={SEMANTIC_THRESHOLD}, top_k={TOP_K}).")
    print(f"  Réduction : {len(ref_A) * len(ref_B)} combinaisons → {len(pairs)} paires ({100*len(pairs)/(len(ref_A)*len(ref_B)):.1f}%)")
    return pairs, matrix


def _select_pairs(
    ref_A: list[RequirementNormalized],
    ref_B: list[RequirementNormalized],
    matrix: np.ndarray,
) -> list[CandidatePair]:
    pairs = []
    seen = set()

    for i, req_a in enumerate(ref_A):
        scores = matrix[i]  # shape (n_B,)
        # Top-k indices triés par score décroissant
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        for j in top_indices:
            score = float(scores[j])
            if score < SEMANTIC_THRESHOLD:
                break
            key = (req_a.id, ref_B[j].id)
            if key not in seen:
                seen.add(key)
                pairs.append(CandidatePair(
                    id_A=req_a.id,
                    title_A=req_a.title,
                    id_B=ref_B[j].id,
                    title_B=ref_B[j].title,
                    semantic_score=score,
                ))

    # Trier par score décroissant
    pairs.sort(key=lambda p: p.semantic_score, reverse=True)
    return pairs


def _save_pairs(pairs: list[CandidatePair]) -> None:
    CACHE_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PAIRS, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in pairs], f, ensure_ascii=False, indent=2)


def _load_pairs() -> list[CandidatePair]:
    with open(CACHE_PAIRS, "r", encoding="utf-8") as f:
        return [CandidatePair(**d) for d in json.load(f)]

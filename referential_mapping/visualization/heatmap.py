"""
visualization/heatmap.py — Étape 3 : matrice de corrélation (heatmap visuelle).

Génère une heatmap seaborn de la similarity_matrix (n_A × n_B).
Output : outputs/correlation_matrix.png
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from referential_mapping.models import RequirementNormalized
from referential_mapping.config import OUTPUT_DIR


def run(
    ref_A: list[RequirementNormalized],
    ref_B: list[RequirementNormalized],
    matrix: np.ndarray,
    output_path: Path | None = None,
    show: bool = True,
) -> Path:
    """Génère et sauvegarde la heatmap. Retourne le chemin du fichier.
    show=True : affiche inline (notebook). show=False : ferme la figure après sauvegarde.
    """
    out = output_path or OUTPUT_DIR / "correlation_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    labels_A = [r.id for r in ref_A]
    labels_B = [r.id for r in ref_B]

    # Taille adaptative selon le nombre d'exigences
    fig_w = max(20, len(labels_B) * 0.25)
    fig_h = max(14, len(labels_A) * 0.15)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix,
        xticklabels=labels_B,
        yticklabels=labels_A,
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        ax=ax,
        linewidths=0,
        cbar_kws={"label": "Similarité cosinus"},
    )
    ax.set_title("Matrice de similarité sémantique — titres uniquement", fontsize=14, pad=12)
    ax.set_xlabel("Ref B", fontsize=11)
    ax.set_ylabel("Ref A", fontsize=11)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6)

    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)

    print(f"[Étape 3] Heatmap sauvegardée → {out}")
    return out

"""
pipeline.py — Orchestrateur principal du pipeline de mapping.

Usage :
    python pipeline.py --ref-a files/surveys/cis-controls-v8-1.yaml --adapter-a cis_v8 \
                       --ref-b files/surveys/nistCsfV2.yaml         --adapter-b nist_csf_v2

Options :
    --force-step1   Repaser les fichiers sources même si le cache existe
    --force-step2   Recalculer les embeddings même si le cache existe
    --force-step4   Rescorer avec le LLM même si le cache existe
    --skip-step4    Ne pas lancer le LLM (étapes 1-3 uniquement)
"""
import argparse
import importlib
import sys
from pathlib import Path

# Assure que le répertoire racine est dans le path
sys.path.insert(0, str(Path(__file__).parent.parent))

from referential_mapping.ingestion import ingester
from referential_mapping.semantic import similarity
from referential_mapping.visualization import heatmap
from referential_mapping.llm import scorer

ADAPTER_ALIASES = {
    "cis_v8":      "referential_mapping.ingestion.adapters.cis_v8",
    "nist_csf_v2": "referential_mapping.ingestion.adapters.nist_csf_v2",
}


def _load_adapter(name: str):
    module_path = ADAPTER_ALIASES.get(name, name)
    return importlib.import_module(module_path)


def main():
    parser = argparse.ArgumentParser(description="Pipeline de mapping référentiels")
    parser.add_argument("--ref-a",     required=True, help="Chemin vers le fichier Ref A (YAML)")
    parser.add_argument("--adapter-a", required=True, help="Adaptateur Ref A (ex. cis_v8)")
    parser.add_argument("--ref-b",     required=True, help="Chemin vers le fichier Ref B (YAML)")
    parser.add_argument("--adapter-b", required=True, help="Adaptateur Ref B (ex. nist_csf_v2)")
    parser.add_argument("--force-step1", action="store_true")
    parser.add_argument("--force-step2", action="store_true")
    parser.add_argument("--force-step4", action="store_true")
    parser.add_argument("--skip-step4",  action="store_true", help="Arrêter après l'étape 3")
    args = parser.parse_args()

    adapter_a = _load_adapter(args.adapter_a)
    adapter_b = _load_adapter(args.adapter_b)

    # ── Étape 1 ──────────────────────────────────────────────────────────────
    ref_A, ref_B = ingester.run(
        path_A=args.ref_a, adapter_A=adapter_a,
        path_B=args.ref_b, adapter_B=adapter_b,
        force=args.force_step1,
    )

    # ── Étape 2 ──────────────────────────────────────────────────────────────
    candidates, matrix = similarity.run(ref_A, ref_B, force=args.force_step2)

    # ── Étape 3 ──────────────────────────────────────────────────────────────
    heatmap.run(ref_A, ref_B, matrix, show=False)

    if args.skip_step4:
        print("\n[Pipeline] Étape 4 ignorée (--skip-step4). Pipeline terminé.")
        return

    # ── Étape 4 ──────────────────────────────────────────────────────────────
    relations = scorer.run(candidates, ref_A, ref_B, force=args.force_step4)

    print(f"\n[Pipeline] Terminé — {len(relations)} relations exportées.")


if __name__ == "__main__":
    main()

"""
ingestion/ingester.py — Orchestrateur de l'étape 1.

Usage :
    from referential_mapping.ingestion.ingester import run
    ref_A, ref_B = run(path_A, adapter_A, path_B, adapter_B)

Les résultats sont mis en cache dans data/ pour ne pas re-parser à chaque run.
"""
import json
from pathlib import Path
from referential_mapping.models import RequirementNormalized
from referential_mapping.config import DATA_DIR


def run(
    path_A: str,
    adapter_A,
    path_B: str,
    adapter_B,
    cache_name_A: str = "ref_A_normalized.json",
    cache_name_B: str = "ref_B_normalized.json",
    force: bool = False,
) -> tuple[list[RequirementNormalized], list[RequirementNormalized]]:
    """
    Parse et normalise les deux référentiels.
    Si les fichiers cache existent déjà, les recharge directement (sauf si force=True).
    """
    ref_A = _load_or_parse(path_A, adapter_A, DATA_DIR / cache_name_A, force)
    ref_B = _load_or_parse(path_B, adapter_B, DATA_DIR / cache_name_B, force)

    print(f"[Étape 1] Ref A : {len(ref_A)} exigences  |  Ref B : {len(ref_B)} exigences")
    return ref_A, ref_B


def _load_or_parse(
    source_path: str,
    adapter,
    cache_path: Path,
    force: bool,
) -> list[RequirementNormalized]:
    if cache_path.exists() and not force:
        print(f"  [cache] {cache_path.name}")
        return _load_json(cache_path)

    print(f"  [parse] {Path(source_path).name} ...")
    requirements = adapter.load(source_path)
    _save_json(requirements, cache_path)
    print(f"  [ok]    {len(requirements)} exigences → {cache_path.name}")
    return requirements


def _save_json(requirements: list[RequirementNormalized], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in requirements], f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> list[RequirementNormalized]:
    with open(path, "r", encoding="utf-8") as f:
        return [RequirementNormalized.from_dict(d) for d in json.load(f)]

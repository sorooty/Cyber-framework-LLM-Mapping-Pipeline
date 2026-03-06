"""
ingestion/ingester.py — Orchestrateur de l'étape 1.

Usage :
    from referential_mapping.ingestion.ingester import run
    ref_A, ref_B = run(path_A, adapter_A, path_B, adapter_B)

Les résultats sont mis en cache dans data/ pour ne pas re-parser à chaque run.
Valide automatiquement le parsing contre survey.ts après chaque chargement.
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
    Valide chaque parsing contre survey.ts et affiche un warning si écart.
    """
    ref_A = _load_or_parse(path_A, adapter_A, DATA_DIR / cache_name_A, force)
    ref_B = _load_or_parse(path_B, adapter_B, DATA_DIR / cache_name_B, force)

    _validate(Path(path_A).stem, ref_A)
    _validate(Path(path_B).stem, ref_B)

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


def _validate(source_stem: str, requirements: list[RequirementNormalized]) -> None:
    """Valide le parsing contre survey.ts. Warning silencieux si écart."""
    if not requirements:
        return
    framework_name = requirements[0].framework
    try:
        from referential_mapping.survey_validator import validate_by_yaml_name
        report = validate_by_yaml_name(framework_name, requirements)
        if report.ok:
            print(f"  [✅ survey.ts] {framework_name} : {report.total_parsed}/{report.total_expected}")
        else:
            print(
                f"  [⚠️  survey.ts] {framework_name} : "
                f"{report.total_parsed}/{report.total_expected} "
                f"(écart {report.total_parsed - report.total_expected:+d})"
            )
            for s in report.missing_sections[:3]:
                exp, got = report.section_detail[s]
                print(f"     section {s} manquante : attendu={exp}, parsé={got}")
    except KeyError:
        pass  # framework absent de survey.ts
    except Exception:
        pass  # survey.ts indisponible → on ne bloque pas


def _save_json(requirements: list[RequirementNormalized], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in requirements], f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> list[RequirementNormalized]:
    with open(path, "r", encoding="utf-8") as f:
        return [RequirementNormalized.from_dict(d) for d in json.load(f)]

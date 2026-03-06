"""
ingestion/registry.py — Découverte et chargement des référentiels disponibles.

Scanne le dossier surveys/ et expose la liste des frameworks disponibles.
Valide automatiquement le parsing contre le schéma survey.ts.
"""
from pathlib import Path
from referential_mapping.models import RequirementNormalized
from referential_mapping.ingestion.adapters.generic_yaml import load as generic_load, _name_from_path

# Dossier contenant les YAMLs
SURVEYS_DIR = Path(__file__).parent.parent.parent / "files" / "surveys"


def list_frameworks() -> dict[str, Path]:
    """
    Retourne un dict {display_name: yaml_path} pour tous les YAMLs disponibles.
    Exclut le sous-dossier __MACOSX.
    """
    frameworks = {}
    for yaml_path in sorted(SURVEYS_DIR.rglob("*.yaml")):
        if "__MACOSX" in yaml_path.parts:
            continue
        name = _name_from_path(yaml_path)
        frameworks[name] = yaml_path
    return frameworks


def load_framework(
    yaml_path: str | Path,
    framework_name: str | None = None,
    validate: bool = True,
) -> list[RequirementNormalized]:
    """
    Charge un référentiel depuis son YAML via l'adaptateur générique.
    Si validate=True, vérifie le compte parsé contre survey.ts et loggue un warning si écart.
    Retourne (requirements, validation_info) si validate=True, sinon requirements seul.
    """
    requirements = generic_load(str(yaml_path), framework_name=framework_name)

    if validate and framework_name:
        _validate_and_warn(framework_name, requirements)

    return requirements


def _validate_and_warn(framework_name: str, requirements: list[RequirementNormalized]) -> None:
    """Valide silencieusement le parsing. Loggue un warning en cas d'écart."""
    try:
        from referential_mapping.survey_validator import validate_by_yaml_name
        report = validate_by_yaml_name(framework_name, requirements)
        if not report.ok:
            print(
                f"  ⚠️  [survey.ts] {framework_name} : "
                f"attendu={report.total_expected}, parsé={report.total_parsed} "
                f"(écart {report.total_parsed - report.total_expected:+d})"
            )
            if report.missing_sections:
                for s in report.missing_sections[:5]:
                    exp, got = report.section_detail[s]
                    print(f"     section {s}: attendu={exp}, parsé={got}")
    except KeyError:
        pass  # framework non référencé dans survey.ts → pas de validation possible
    except Exception:
        pass  # survey.ts indisponible → on ne bloque pas le parsing


def get_expected_count(framework_name: str) -> int | None:
    """
    Retourne le nombre d'exigences attendu pour un framework selon survey.ts.
    Retourne None si le framework n'est pas dans survey.ts.
    """
    try:
        from referential_mapping.survey_validator import validate_by_yaml_name, _parse_survey_ts, total_expected, YAML_TO_SURVEY_KEY
        survey_key = YAML_TO_SURVEY_KEY.get(framework_name)
        if not survey_key:
            return None
        survey = _parse_survey_ts()
        return total_expected(survey_key, survey)
    except Exception:
        return None

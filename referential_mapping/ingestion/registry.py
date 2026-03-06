"""
ingestion/registry.py — Découverte et chargement des référentiels disponibles.

Scanne le dossier surveys/ et expose la liste des frameworks disponibles.
"""
from pathlib import Path
import yaml
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
) -> list[RequirementNormalized]:
    """Charge un référentiel depuis son YAML via l'adaptateur générique."""
    return generic_load(str(yaml_path), framework_name=framework_name)

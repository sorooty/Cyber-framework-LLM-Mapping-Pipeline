"""
adapters/cis_v8.py — Parsing & normalisation du référentiel CIS Controls v8.

Source : YAML (ex. cis-controls-v8-1.yaml)
Structure YAML :
  {control_num}: {
    title: str,          # titre du contrôle parent
    desc:  str,
    {n}_label: str,      # titre du safeguard n
    {n}_desc:  str,      # description du safeguard n
    {n}_tags:  str,      # ex. "#Identifier#IG1#IG2"
    ...
  }
Sortie : une exigence par safeguard, id = "CIS-{control}.{n}"
"""
import yaml
from referential_mapping.models import RequirementNormalized

FRAMEWORK = "CIS_v8"


def load(path: str) -> list[RequirementNormalized]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    requirements = []
    for ctrl_num, ctrl in data.items():
        if not isinstance(ctrl, dict):
            continue
        # Trouver tous les safeguards : clés de la forme "{n}_label"
        safeguard_nums = sorted(
            int(k.split("_")[0])
            for k in ctrl
            if k.endswith("_label") and k.split("_")[0].isdigit()
        )
        for n in safeguard_nums:
            title = str(ctrl.get(f"{n}_label", "")).strip()
            if not title:
                continue
            description = str(ctrl.get(f"{n}_desc", "")).strip()
            tags = _parse_tags(str(ctrl.get(f"{n}_tags", "")))
            requirements.append(RequirementNormalized(
                id=f"CIS-{ctrl_num}.{n}",
                framework=FRAMEWORK,
                title=title,
                description=description,
                tags=tags,
            ))

    return requirements


def _parse_tags(raw: str) -> list[str]:
    """Parse '#Identifier#IG1#IG2#IG3' → ['Identifier', 'IG1', 'IG2', 'IG3']"""
    return [t for t in raw.strip("#").split("#") if t]

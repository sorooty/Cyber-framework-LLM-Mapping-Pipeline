"""
ingestion/adapters/generic_yaml.py — Adaptateur universel pour tous les YAMLs du dossier surveys.

Supporte la structure commune à tous les référentiels :
  {section_num}: {
    title: str,
    prefix: str,          (optionnel)
    desc: str,            (optionnel)
    {n}_label: str,       → requirement title
    {n}_desc:  str,       → requirement description
    {n}_prefix: str,      → requirement id (si présent)
    {sub_num}: { ... }    → sous-sections récursives
  }
"""
import re
import yaml
from pathlib import Path
from referential_mapping.models import RequirementNormalized
from referential_mapping.schemas import get_schema, FrameworkSchema


def load(path: str, framework_name: str | None = None) -> list[RequirementNormalized]:
    """
    Parse n'importe quel YAML de référentiel.
    framework_name : si None, déduit du nom de fichier.
    """
    path = Path(path)
    framework = framework_name or _name_from_path(path)
    schema = get_schema(framework)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    requirements: list[RequirementNormalized] = []
    for section_num, section in data.items():
        if not isinstance(section, dict):
            continue
        section_title  = str(section.get("title", "")).strip()
        section_prefix = str(section.get("prefix", "")).strip()

        # Résolution de l'ID de section (overwrite_ids pour ISO 27001, etc.)
        effective_id = str(
            schema.effective_section_id(section_num) if schema else section_num
        )

        _extract(
            node=section,
            framework=framework,
            id_prefix=section_prefix or effective_id,
            parent_tags=[section_title] if section_title else [],
            requirements=requirements,
            schema=schema,
            section_key=section_num,
        )

    return requirements


def _extract(
    node: dict,
    framework: str,
    id_prefix: str,
    parent_tags: list[str],
    requirements: list[RequirementNormalized],
    schema: FrameworkSchema | None = None,
    section_key=None,
) -> None:
    # ── 1. Exigences plates : {n}_label ou {n}_desc (si pas de label) ─────────
    # Collecter tous les indices numériques présents
    all_nums = set()
    for k in node:
        if isinstance(k, str):
            m = k.split("_")
            if len(m) >= 2 and m[0].isdigit() and m[1] in ("label", "desc", "prefix"):
                all_nums.add(int(m[0]))

    for n in sorted(all_nums):
        label  = str(node.get(f"{n}_label", "")).strip()
        desc   = str(node.get(f"{n}_desc",  "")).strip()
        prefix = str(node.get(f"{n}_prefix", "")).strip()

        # Si pas de label, utiliser le début du desc comme titre
        if not label and desc:
            label = desc[:120].rstrip() + ("…" if len(desc) > 120 else "")
            desc  = ""   # desc utilisé comme titre, on évite la redondance

        if not label:
            continue

        req_id = prefix if prefix else f"{id_prefix}.{n}"
        req_id = _clean_id(req_id)

        requirements.append(RequirementNormalized(
            id=req_id,
            framework=framework,
            title=_clean_text(label),
            description=_clean_text(desc),
            tags=parent_tags.copy(),
        ))

    # ── 2. Sous-sections récursives : clés entières ───────────────────────────
    sub_nums = sorted(k for k in node if isinstance(k, int))
    for sub_num in sub_nums:
        sub = node[sub_num]
        if not isinstance(sub, dict):
            continue
        sub_title  = str(sub.get("title",  "")).strip()
        sub_prefix = str(sub.get("prefix", "")).strip()
        sub_id = sub_prefix if sub_prefix else f"{id_prefix}.{sub_num}"

        new_tags = parent_tags + ([sub_title] if sub_title else [])
        _extract(
            node=sub,
            framework=framework,
            id_prefix=sub_id,
            parent_tags=new_tags,
            requirements=requirements,
            schema=schema,
        )


def _name_from_path(path: Path) -> str:
    """cis-controls-v8-1.yaml → CIS_Controls_v8"""
    stem = path.stem
    # Retirer les suffixes de version redondants (_1, _2 …)
    stem = re.sub(r"[-_]\d+$", "", stem)
    return stem.replace("-", "_").replace(" ", "_")


def _clean_id(raw: str) -> str:
    """Supprime les caractères non-ASCII dans les IDs."""
    return re.sub(r"[^\x00-\x7F]", "", raw).strip(". ")


def _clean_text(raw: str) -> str:
    """Nettoie les espaces multiples et retours ligne parasites."""
    return re.sub(r"\s+", " ", raw).strip()

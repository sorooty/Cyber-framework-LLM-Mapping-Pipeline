"""
adapters/nist_csf_v2.py — Parsing & normalisation du référentiel NIST CSF v2.

Source : YAML (ex. nistCsfV2.yaml)
Structure YAML :
  {fn_num}: {
    title: str,         # titre de la fonction (ex. "Gouverner")
    prefix: str,        # ex. "GV"
    desc: str,
    {cat_num}: {        # catégorie
      title: str,
      prefix: str,      # ex. "GV.OC"
      desc: str,
      {n}_prefix: str,  # ex. "GV.OC-01"  ← id de la sous-catégorie
      {n}_label:  str,  # titre de la sous-catégorie
    }
  }
Sortie : une exigence par sous-catégorie, id = prefix (ex. "GV.OC-01")
"""
import yaml
from referential_mapping.models import RequirementNormalized

FRAMEWORK = "NIST_CSF_v2"


def load(path: str) -> list[RequirementNormalized]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    requirements = []
    for fn_num, fn in data.items():
        if not isinstance(fn, dict):
            continue
        fn_title = fn.get("title", "")

        # Catégories : clés entières imbriquées
        cat_nums = [k for k in fn if isinstance(k, int)]
        for cat_num in sorted(cat_nums):
            cat = fn[cat_num]
            if not isinstance(cat, dict):
                continue
            cat_prefix = cat.get("prefix", "")
            cat_desc   = cat.get("desc", "")

            # Sous-catégories : clés de la forme "{n}_prefix" + "{n}_label"
            sc_nums = sorted(
                int(k.split("_")[0])
                for k in cat
                if k.endswith("_prefix") and k.split("_")[0].isdigit()
            )
            for n in sc_nums:
                sc_id    = str(cat.get(f"{n}_prefix", "")).strip()
                sc_label = str(cat.get(f"{n}_label", "")).strip()
                if not sc_id or not sc_label:
                    continue
                requirements.append(RequirementNormalized(
                    id=sc_id,
                    framework=FRAMEWORK,
                    title=sc_label,
                    description=cat_desc,   # description de la catégorie parente
                    tags=[fn_title, cat_prefix],
                ))

    return requirements

"""
survey_validator.py — Valide le parsing YAML contre le schéma officiel survey.ts.

Le fichier survey.ts (fourni par l'équipe dev) contient la structure exacte attendue
pour chaque framework : nombre d'exigences par section/sous-section.

Usage :
    from referential_mapping.survey_validator import validate, expected_counts, SurveySchema
    report = validate("cis_controls_v8", parsed_requirements)
    print(report)
"""
import re
import json
from pathlib import Path
from dataclasses import dataclass, field

# ── Chemin vers survey.ts ─────────────────────────────────────────────────────
SURVEY_TS_PATH = Path(__file__).parent.parent / "files" / "survey.ts"


# ── Parsing du fichier .ts ────────────────────────────────────────────────────

def _parse_survey_ts(path: Path = SURVEY_TS_PATH) -> dict:
    """
    Extrait le contenu JSON-like de survey.ts.
    Stratégie : on isole le bloc JS entre defineSurveyConfig({ ... }) et on le
    convertit en JSON valide pour le parser avec json.loads.
    """
    src = path.read_text(encoding="utf-8")

    # Extraire le contenu entre defineSurveyConfig({ et }) \ntype
    m = re.search(r"defineSurveyConfig\(\{(.+?)\}\)\s*\ntype", src, re.DOTALL)
    if not m:
        raise ValueError("Impossible de localiser defineSurveyConfig({...}) dans survey.ts")

    raw = "{" + m.group(1) + "}"

    # Conversions JS → JSON
    # 1. Commentaires JS (// ...) — en premier pour éviter les faux positifs
    raw = re.sub(r"//[^\n]*", "", raw)
    # 2. Single quotes → double quotes
    raw = raw.replace("'", '"')
    # 3. Trailing commas avant } ou ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    # 4. Clés non-quotées alphabétiques (ex: groups, flatGroups …)
    raw = re.sub(r'(?<!")(\b[a-zA-Z_][a-zA-Z0-9_]*\b)(?!")\s*:', r'"\1":', raw)
    # 5. Clés non-quotées numériques (ex: 1:, 2:, 10: …) — après les alpha pour éviter les doublons
    raw = re.sub(r'(?<!")(\b\d+\b)(?!")\s*:', r'"\1":', raw)
    # 6. Clés mixtes déjà quotées ex: "1a", "2d" → OK après les étapes précédentes

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Aide au debug : afficher le contexte autour de l'erreur
        lines = raw.splitlines()
        err_line = e.lineno - 1
        context = "\n".join(lines[max(0, err_line-3): err_line+3])
        raise ValueError(f"Erreur de parsing JSON survey.ts (ligne {e.lineno}) :\n{context}\n\n{e}")


# ── Calcul du nombre attendu d'exigences ─────────────────────────────────────

def _count_from_groups(groups: dict) -> dict[str, int]:
    """
    Retourne un dict { "section.subsection" ou "section": count_attendu }
    en parcourant la structure groups du survey.ts.

    Logique :
      - groups[sec] = int           → section plate : sec.1 … sec.n
      - groups[sec][sub] = int      → sous-section  : sec.sub.1 … sec.sub.n
    """
    counts: dict[str, int] = {}
    for sec, val in groups.items():
        if isinstance(val, int):
            counts[str(sec)] = val
        elif isinstance(val, dict):
            for sub, n in val.items():
                if isinstance(n, int):
                    key = f"{sec}.{sub}"
                    counts[key] = n
    return counts


def expected_counts(framework_key: str, survey: dict | None = None) -> dict[str, int]:
    """
    Retourne { "section[.subsection]": nb_exigences_attendu } pour un framework.

    framework_key : clé telle qu'elle apparaît dans survey.ts
                    ex. "cis-controls-v8-1", "nistCsfV2", "dora"
    """
    if survey is None:
        survey = _parse_survey_ts()
    if framework_key not in survey:
        raise KeyError(
            f"'{framework_key}' introuvable dans survey.ts.\n"
            f"Clés disponibles : {sorted(survey.keys())}"
        )
    return _count_from_groups(survey[framework_key]["groups"])


def total_expected(framework_key: str, survey: dict | None = None) -> int:
    """Retourne le nombre total d'exigences attendu pour un framework."""
    return sum(expected_counts(framework_key, survey).values())


# ── Rapport de validation ─────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    framework_key: str
    total_expected: int
    total_parsed: int
    missing_sections: list[str] = field(default_factory=list)   # sections sans exigences parsées
    over_sections: list[str]   = field(default_factory=list)    # sections avec trop d'exigences
    section_detail: dict       = field(default_factory=dict)    # { section: (expected, parsed) }

    @property
    def ok(self) -> bool:
        return self.total_expected == self.total_parsed

    def __str__(self) -> str:
        status = "✅ OK" if self.ok else "❌ ÉCART"
        lines = [
            f"\n{'='*60}",
            f" VALIDATION — {self.framework_key}  [{status}]",
            f"{'='*60}",
            f" Attendu  : {self.total_expected} exigences",
            f" Parsé    : {self.total_parsed} exigences",
            f" Écart    : {self.total_parsed - self.total_expected:+d}",
        ]
        if self.missing_sections:
            lines.append(f"\n Sections manquantes ({len(self.missing_sections)}) :")
            for s in self.missing_sections[:20]:
                exp = self.section_detail.get(s, (0, 0))
                lines.append(f"   {s:<15} attendu={exp[0]}  parsé={exp[1]}")
        if self.over_sections:
            lines.append(f"\n Sections en excès ({len(self.over_sections)}) :")
            for s in self.over_sections[:20]:
                exp = self.section_detail.get(s, (0, 0))
                lines.append(f"   {s:<15} attendu={exp[0]}  parsé={exp[1]}")
        lines.append("="*60)
        return "\n".join(lines)


def validate(
    framework_key: str,
    parsed_requirements: list,
    survey: dict | None = None,
) -> ValidationReport:
    """
    Valide une liste de RequirementNormalized contre les counts du survey.ts.

    framework_key : clé survey.ts  (ex. "cis-controls-v8-1")
    parsed_requirements : liste de RequirementNormalized
    """
    if survey is None:
        survey = _parse_survey_ts()

    exp = expected_counts(framework_key, survey)
    total_exp = sum(exp.values())

    # Compter les exigences parsées par section de premier niveau (et sous-section)
    # On extrait le préfixe de l'ID : "1.2.3" → clé "1.2"  ou  "1.2" → clé "1"
    parsed_by_section: dict[str, int] = {}
    for req in parsed_requirements:
        parts = req.id.split(".")
        # Essayer clé "section.subsection" d'abord, puis "section"
        if len(parts) >= 3:
            key2 = f"{parts[0]}.{parts[1]}"
            key1 = parts[0]
            if key2 in exp:
                parsed_by_section[key2] = parsed_by_section.get(key2, 0) + 1
            elif key1 in exp:
                parsed_by_section[key1] = parsed_by_section.get(key1, 0) + 1
            else:
                parsed_by_section[key2] = parsed_by_section.get(key2, 0) + 1
        elif len(parts) == 2:
            key = parts[0]
            parsed_by_section[key] = parsed_by_section.get(key, 0) + 1
        else:
            parsed_by_section[req.id] = parsed_by_section.get(req.id, 0) + 1

    section_detail = {s: (exp.get(s, 0), parsed_by_section.get(s, 0)) for s in
                      set(exp) | set(parsed_by_section)}

    missing = [s for s, (e, p) in section_detail.items() if e > 0 and p < e]
    over    = [s for s, (e, p) in section_detail.items() if p > e]

    return ValidationReport(
        framework_key=framework_key,
        total_expected=total_exp,
        total_parsed=len(parsed_requirements),
        missing_sections=sorted(missing),
        over_sections=sorted(over),
        section_detail=section_detail,
    )


# ── Mapping nom YAML → clé survey.ts ─────────────────────────────────────────

# Dictionnaire de correspondance (à compléter si nécessaire)
YAML_TO_SURVEY_KEY: dict[str, str] = {
    "cis_controls_v8":     "cis-controls-v8-1",
    "nistCsfV2":           "nistCsfV2",
    "nis2":                "nis2",
    "nis2v2":              "nis2v2",
    "dora":                "dora",
    "iso27001":            "iso27001-2022",
    "iso9001":             "iso9001-2015",
    "iso13485":            "iso13485-2016",
    "iso14001":            "iso14001-2015",
    "iso17021":            "iso17021-2015",
    "iso17024":            "iso17024-2012",
    "iso20000":            "iso20000-2018",
    "iso20022_1":          "iso20022-1-2013",
    "iso20022_2":          "iso20022-2-2013",
    "iso26000":            "iso26000-2020",
    "iso27701":            "iso27701-2021",
    "iso42001":            "iso42001-2023",
    "iso50001":            "iso50001-2018",
    "hdsV2":               "hdsV2",
    "pcidssV4":            "pcidssV4",
    "rgpdCnil":            "rgpdCnil",
    "soc2_type2":          "soc2-type2",
    "qualiopiV9":          "qualiopiV9",
    "anssiHygieneGuideV2": "anssiHygieneGuideV2",
    "anssiAISecurity":     "anssiAISecurity",
    "owaspV4_0":           "owaspV4-0-3",
    "owasp_llm_ai_v1":     "owasp-llm-ai-v1",
    "eumdr":               "eumdr-2017",
    "ecc1":                "ecc1-2018",
    "5_20Law":             "5-20Law",
    "afnor_spec":          "afnor-spec-2217",
    "secnumcloud_v3":      "secnumcloud-v3-2",
    "igi11300":            "igi1300-2021",
    "ii901":               "ii901",
    "iec_62443_3_2":       "iec-62443-3-2-2020",
    "ichE6R2":             "ichE6R2",
    "isa6_0_3_tisax":      "isa6-0-3-tisax",
    "ai_act":              "ai-act",
    "fda_cfr_21_part":     "fda-cfr-21-part-111",
    "cosmetovigilance":    "cosmetovigilance-2009",
    "ansm":                "ansm-2022",
}


def validate_by_yaml_name(
    yaml_name: str,
    parsed_requirements: list,
    survey: dict | None = None,
) -> ValidationReport:
    """
    Valide à partir du nom YAML (tel que retourné par list_frameworks()).
    Ex : validate_by_yaml_name("cis_controls_v8", ref_A)
    """
    if survey is None:
        survey = _parse_survey_ts()
    survey_key = YAML_TO_SURVEY_KEY.get(yaml_name)
    if survey_key is None:
        raise KeyError(
            f"Pas de correspondance survey.ts pour '{yaml_name}'.\n"
            f"Ajoute-le dans YAML_TO_SURVEY_KEY dans survey_validator.py."
        )
    return validate(survey_key, parsed_requirements, survey)


# ── Résumé global de tous les frameworks ─────────────────────────────────────

def summary_table(survey: dict | None = None) -> list[dict]:
    """
    Retourne un tableau récapitulatif de tous les frameworks avec leur total attendu.
    Utile pour avoir une vue d'ensemble.
    """
    if survey is None:
        survey = _parse_survey_ts()
    rows = []
    for key, cfg in survey.items():
        try:
            total = total_expected(key, survey)
            flat  = len(cfg.get("flatGroups", []))
            rows.append({
                "survey_key":   key,
                "total_attendu": total,
                "nb_sections":  len(cfg["groups"]),
                "flat_groups":  flat,
            })
        except Exception:
            pass
    return sorted(rows, key=lambda r: r["survey_key"])


if __name__ == "__main__":
    # Test rapide
    survey = _parse_survey_ts()
    print(f"Frameworks dans survey.ts : {len(survey)}\n")
    for row in summary_table(survey):
        print(f"  {row['survey_key']:<30} {row['total_attendu']:>5} req  "
              f"({row['nb_sections']} sections)")

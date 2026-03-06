"""
referential_mapping/schemas.py — Modèle de schéma de framework (pipeline-centric).

Définit FrameworkSchema, un modèle Python enrichi auto-populé depuis survey.ts
(source de vérité structurelle) et complété avec des métadonnées pipeline
(domain, language, display_name).

Usage :
    from referential_mapping.schemas import SCHEMAS, FrameworkSchema
    schema = SCHEMAS["cis_controls_v8"]
    print(schema.total_expected)          # 153
    print(schema.section_count(1))        # nb items section 1
    print(schema.effective_section_id(11))  # 5  (ISO 27001 Annex A)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from referential_mapping.survey_validator import (
    _parse_survey_ts,
    YAML_TO_SURVEY_KEY,
)


# ── Métadonnées manuelles par framework ───────────────────────────────────────
# Ce que survey.ts ne contient pas : domaine sémantique, langue, nom lisible.
# Clé = yaml_name (identique à YAML_TO_SURVEY_KEY)

_METADATA: dict[str, dict] = {
    "cis_controls_v8":     {"display_name": "CIS Controls v8",                  "domain": "cybersecurity",  "language": "en"},
    "nistCsfV2":           {"display_name": "NIST CSF v2",                       "domain": "cybersecurity",  "language": "en"},
    "nis2":                {"display_name": "NIS2",                              "domain": "cybersecurity",  "language": "fr"},
    "nis2v2":              {"display_name": "NIS2 v2",                           "domain": "cybersecurity",  "language": "fr"},
    "dora":                {"display_name": "DORA",                              "domain": "financial",      "language": "fr"},
    "iso27001":            {"display_name": "ISO 27001:2022",                    "domain": "cybersecurity",  "language": "en"},
    "iso27701":            {"display_name": "ISO 27701:2021",                    "domain": "privacy",        "language": "en"},
    "iso9001":             {"display_name": "ISO 9001:2015",                     "domain": "quality",        "language": "en"},
    "iso13485":            {"display_name": "ISO 13485:2016",                    "domain": "medical",        "language": "en"},
    "iso14001":            {"display_name": "ISO 14001:2015",                    "domain": "environment",    "language": "en"},
    "iso17021":            {"display_name": "ISO 17021:2015",                    "domain": "audit",          "language": "en"},
    "iso17024":            {"display_name": "ISO 17024:2012",                    "domain": "certification",  "language": "en"},
    "iso20000":            {"display_name": "ISO 20000:2018",                    "domain": "itsm",           "language": "en"},
    "iso20022_1":          {"display_name": "ISO 20022-1:2013",                  "domain": "financial",      "language": "en"},
    "iso20022_2":          {"display_name": "ISO 20022-2:2013",                  "domain": "financial",      "language": "en"},
    "iso26000":            {"display_name": "ISO 26000:2020",                    "domain": "social",         "language": "en"},
    "iso42001":            {"display_name": "ISO 42001:2023",                    "domain": "ai",             "language": "en"},
    "iso50001":            {"display_name": "ISO 50001:2018",                    "domain": "energy",         "language": "en"},
    "hdsV2":               {"display_name": "HDS v2",                            "domain": "healthcare",     "language": "fr"},
    "pcidssV4":            {"display_name": "PCI DSS v4",                        "domain": "payment",        "language": "en"},
    "rgpdCnil":            {"display_name": "RGPD/CNIL",                         "domain": "privacy",        "language": "fr"},
    "soc2_type2":          {"display_name": "SOC 2 Type II",                     "domain": "cybersecurity",  "language": "en"},
    "qualiopiV9":          {"display_name": "Qualiopi v9",                       "domain": "training",       "language": "fr"},
    "anssiHygieneGuideV2": {"display_name": "ANSSI Guide d'Hygiène v2",          "domain": "cybersecurity",  "language": "fr"},
    "anssiAISecurity":     {"display_name": "ANSSI AI Security",                 "domain": "ai",             "language": "fr"},
    "owaspV4_0":           {"display_name": "OWASP ASVS v4.0",                   "domain": "cybersecurity",  "language": "en"},
    "owasp_llm_ai_v1":     {"display_name": "OWASP LLM AI v1",                   "domain": "ai",             "language": "en"},
    "eumdr":               {"display_name": "EU MDR 2017",                       "domain": "medical",        "language": "en"},
    "ecc1":                {"display_name": "ECC 1.0",                           "domain": "cybersecurity",  "language": "en"},
    "5_20Law":             {"display_name": "Loi 5-20",                          "domain": "law",            "language": "fr"},
    "afnor_spec":          {"display_name": "AFNOR Spec AI 22-17",               "domain": "ai",             "language": "fr"},
    "secnumcloud_v3":      {"display_name": "SecNumCloud v3",                    "domain": "cloud",          "language": "fr"},
    "igi11300":            {"display_name": "IGI 1300:2021",                     "domain": "classified",     "language": "fr"},
    "ii901":               {"display_name": "II 901",                            "domain": "cybersecurity",  "language": "fr"},
    "iec_62443_3_2":       {"display_name": "IEC 62443-3-2:2020",               "domain": "ot_security",    "language": "en"},
    "ichE6R2":             {"display_name": "ICH E6(R2)",                        "domain": "medical",        "language": "en"},
    "isa6_0_3_tisax":      {"display_name": "ISA6 / TISAX",                      "domain": "automotive",     "language": "en"},
    "ai_act":              {"display_name": "EU AI Act",                         "domain": "ai",             "language": "mixed"},
    "fda_cfr_21_part":     {"display_name": "FDA CFR 21 Part 11",                "domain": "medical",        "language": "en"},
    "cosmetovigilance":    {"display_name": "Cosmétovigilance 2009",             "domain": "cosmetics",      "language": "fr"},
    "ansm":                {"display_name": "ANSM 2022",                         "domain": "medical",        "language": "fr"},
}


# ── Modèle ────────────────────────────────────────────────────────────────────

@dataclass
class FrameworkSchema:
    """
    Schéma structurel + métadonnées pipeline d'un framework de conformité.

    Champs auto-populés depuis survey.ts :
        groups         — structure hiérarchique exacte {sec: n} ou {sec: {sub: n}}
        flat_groups    — sections "feuilles" dont on ne doit pas récurser
        overwrite_ids  — remapping d'IDs de sections {clé_yaml: id_affiché}

    Champs métadonnées (renseignés manuellement via _METADATA) :
        display_name, domain, language

    Champs calculés (propriétés) :
        total_expected, all_sections
    """
    key: str                              # yaml_name, ex. "cis_controls_v8"
    survey_key: str                       # clé survey.ts, ex. "cis-controls-v8-1"
    display_name: str                     # nom lisible
    groups: dict                          # structure brute survey.ts
    flat_groups: list[str]                # sous-sections feuilles
    overwrite_ids: dict[int, int]         # {clé_yaml → id_effectif} ex. {11: 5}
    language: str = "en"                  # "en" | "fr" | "mixed"
    domain: str   = "unknown"            # ex. "cybersecurity", "privacy", "ai"

    # ── Accès structurel ─────────────────────────────────────────────────────

    @property
    def total_expected(self) -> int:
        """Nombre total d'exigences attendu (somme de tous les groupes)."""
        return sum(self._iter_counts())

    @property
    def all_sections(self) -> list[str]:
        """Liste de toutes les clés section[.sous-section] déclarées."""
        result = []
        for sec, val in self.groups.items():
            if isinstance(val, int):
                result.append(str(sec))
            elif isinstance(val, dict):
                for sub in val:
                    result.append(f"{sec}.{sub}")
        return result

    def section_count(
        self,
        section: Union[int, str],
        subsection: Union[int, str, None] = None,
    ) -> int:
        """
        Retourne le nombre d'exigences attendu pour une section (et sous-section).
        Accepte des clés int ou str (robuste aux deux formats).
        """
        sec = self.groups.get(section) or self.groups.get(str(section))
        if sec is None:
            return 0
        if isinstance(sec, int):
            return sec if subsection is None else 0
        if subsection is not None:
            return sec.get(subsection, 0) or sec.get(str(subsection), 0)
        # total de la section
        return sum(n for n in sec.values() if isinstance(n, int))

    def effective_section_id(self, section_key: Union[int, str]) -> Union[int, str]:
        """
        Retourne l'ID effectif d'une section après application des overwrite_ids.
        Ex. : schema_iso27001.effective_section_id(11) → 5  (Annex A)
        """
        int_key = int(section_key) if str(section_key).isdigit() else section_key
        return self.overwrite_ids.get(int_key, int_key)

    def is_flat(self, section_key: Union[int, str]) -> bool:
        """True si cette section est déclarée comme 'flat' dans survey.ts."""
        return str(section_key) in self.flat_groups

    # ── Représentation ───────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"FrameworkSchema: {self.display_name}  [{self.domain} / {self.language}]",
            f"  survey_key     : {self.survey_key}",
            f"  total_expected : {self.total_expected}",
            f"  sections       : {len(self.all_sections)}",
        ]
        if self.overwrite_ids:
            lines.append(f"  overwrite_ids  : {self.overwrite_ids}")
        if self.flat_groups:
            lines.append(f"  flat_groups    : {self.flat_groups[:5]}{'...' if len(self.flat_groups) > 5 else ''}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FrameworkSchema(key={self.key!r}, "
            f"total={self.total_expected}, "
            f"domain={self.domain!r})"
        )

    # ── Interne ──────────────────────────────────────────────────────────────

    def _iter_counts(self):
        for val in self.groups.values():
            if isinstance(val, int):
                yield val
            elif isinstance(val, dict):
                for n in val.values():
                    if isinstance(n, int):
                        yield n


# ── Construction du registre SCHEMAS ─────────────────────────────────────────

def _build_schemas(survey: dict | None = None) -> dict[str, "FrameworkSchema"]:
    """
    Construit le dict SCHEMAS[yaml_name → FrameworkSchema] en fusionnant :
      - Les données structurelles issues de survey.ts (groups, flatGroups, overwriteGroupsId)
      - Les métadonnées manuelles de _METADATA (display_name, domain, language)
    """
    if survey is None:
        survey = _parse_survey_ts()

    schemas: dict[str, FrameworkSchema] = {}

    for yaml_key, survey_key in YAML_TO_SURVEY_KEY.items():
        if survey_key not in survey:
            continue

        cfg = survey[survey_key]
        meta = _METADATA.get(yaml_key, {})

        # overwriteGroupsId : clés peuvent être strings (ex. "11") → int
        raw_overwrite = cfg.get("overwriteGroupsId", {})
        overwrite_ids = {int(k): int(v) for k, v in raw_overwrite.items()}

        schemas[yaml_key] = FrameworkSchema(
            key=yaml_key,
            survey_key=survey_key,
            display_name=meta.get("display_name", yaml_key),
            groups=cfg.get("groups", {}),
            flat_groups=cfg.get("flatGroups", []),
            overwrite_ids=overwrite_ids,
            language=meta.get("language", "en"),
            domain=meta.get("domain", "unknown"),
        )

    return schemas


# Registre global — chargé une seule fois au démarrage du module
SCHEMAS: dict[str, FrameworkSchema] = _build_schemas()


def get_schema(yaml_name: str) -> FrameworkSchema | None:
    """Retourne le schéma d'un framework, ou None s'il n'est pas enregistré."""
    return SCHEMAS.get(yaml_name)


def list_schemas() -> list[str]:
    """Retourne la liste triée des yaml_names disponibles."""
    return sorted(SCHEMAS.keys())

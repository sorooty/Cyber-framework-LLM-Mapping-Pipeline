"""
models.py — Modèles de données partagés dans tout le pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequirementNormalized:
    id: str           # ex. "CIS-1.1" | "ISO-A.5.9"
    framework: str    # ex. "CIS_v8"  | "ISO_27001_2022"
    title: str        # champ principal — étapes 2 & 3
    description: str  # utilisé à l'étape 4 uniquement
    tags: list[str] = field(default_factory=list)  # optionnel

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "framework":   self.framework,
            "title":       self.title,
            "description": self.description,
            "tags":        self.tags,
        }

    @staticmethod
    def from_dict(d: dict) -> "RequirementNormalized":
        return RequirementNormalized(
            id=d["id"],
            framework=d["framework"],
            title=d["title"],
            description=d.get("description", ""),
            tags=d.get("tags", []),
        )


@dataclass
class CandidatePair:
    id_A: str
    title_A: str
    id_B: str
    title_B: str
    semantic_score: float

    def to_dict(self) -> dict:
        return {
            "id_A":           self.id_A,
            "title_A":        self.title_A,
            "id_B":           self.id_B,
            "title_B":        self.title_B,
            "semantic_score": round(self.semantic_score, 4),
        }


@dataclass
class MappingRelation:
    id_A: str
    title_A: str
    id_B: str
    title_B: str
    semantic_score: float
    coverage_A_to_B: float
    coverage_B_to_A: float
    confidence: float
    relation_type: str  # "equivalence"|"A_couvre_B"|"B_couvre_A"|"partielle"|"aucun_lien"

    def to_dict(self) -> dict:
        return {
            "id_A":            self.id_A,
            "title_A":         self.title_A,
            "id_B":            self.id_B,
            "title_B":         self.title_B,
            "semantic_score":  round(self.semantic_score, 4),
            "coverage_A_to_B": round(self.coverage_A_to_B, 4),
            "coverage_B_to_A": round(self.coverage_B_to_A, 4),
            "confidence":      round(self.confidence, 4),
            "relation_type":   self.relation_type,
        }

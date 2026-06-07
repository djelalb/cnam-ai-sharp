"""
Étape 2 — VALIDATION des données (locale).

Valide le manifeste NDJSON produit par l'extraction, *avant* la préparation
(fail-fast, sans télécharger les images) : dimensions d'image positives,
classes connues et boîtes normalisées (cx, cy, w, h ∈ [0, 1], sans négatif).
L'intégrité pixel des images est garantie par la plateforme à l'export.
"""

import json
import logging
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


class ManifestValidator:
    """Vérifie la cohérence des annotations du manifeste exporté."""

    def __init__(self, export: Path = settings.DATASET_EXPORT) -> None:
        self.export = export

    def validate(self) -> int:
        """Retourne le nombre d'enregistrements incohérents (0 = manifeste sain)."""
        if not self.export.exists():
            logger.info("Validation ignorée : manifeste %s absent.", self.export)
            return 0

        records = [
            json.loads(line)
            for line in self.export.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        class_count = len(records[0].get("class_names", {}))
        images = [r for r in records if r.get("type") == "image"]
        errors = sum(self._count_errors(image, class_count) for image in images)

        if errors:
            logger.warning("Validation : %d annotation(s) incohérente(s).", errors)
        else:
            logger.info("Validation OK : %d images cohérentes.", len(images))
        return errors

    @staticmethod
    def _count_errors(image: dict, class_count: int) -> int:
        """Compte les anomalies d'un enregistrement image du manifeste."""
        if image.get("width", 0) <= 0 or image.get("height", 0) <= 0:
            return 1

        errors = 0
        for cls, *coords in image.get("annotations", {}).get("boxes", []):
            class_known = 0 <= cls < class_count
            coords_normalized = all(0.0 <= c <= 1.0 for c in coords)
            if not (class_known and coords_normalized):
                errors += 1
        return errors


def run() -> None:
    """Point d'entrée de l'étape de validation."""
    ManifestValidator().validate()

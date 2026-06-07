"""
Étape 1 — EXTRACTION (locale).

Le dataset est hébergé sur Ultralytics HUB. Cette étape résout le *slug*
lisible (``username/dataset``) en identifiant technique, puis le persiste
pour les étapes suivantes (l'entraînement en a besoin).
"""

import logging
import sys

import requests

from src.config import settings

logger = logging.getLogger(__name__)


class DatasetResolver:
    """Résout et persiste l'identifiant du dataset Ultralytics HUB."""

    def __init__(self) -> None:
        self.headers = {"Authorization": f"Bearer {settings.ULTRALYTICS_API_KEY}"}
        self.api_url = "https://platform.ultralytics.com/api/datasets"

    def resolve(self) -> str:
        """Retourne l'ID du dataset et l'écrit dans ``DATASET_ID_FILE``."""
        target = f"{settings.ULTRALYTICS_USERNAME}/{settings.ULTRALYTICS_DATASET}"
        logger.info("Résolution de %s...", target)

        response = requests.get(self.api_url, headers=self.headers)
        response.raise_for_status()

        dataset = self._find_dataset(response.json().get("datasets", []))
        if not dataset:
            raise ValueError("Dataset introuvable sur le compte Ultralytics.")

        dataset_id = dataset.get("_id") or dataset.get("id")
        self._persist(dataset_id)
        logger.info("ID résolu et sauvegardé : %s", dataset_id)
        return dataset_id

    @staticmethod
    def _find_dataset(datasets: list[dict]) -> dict | None:
        """Sélectionne le dataset correspondant au compte courant."""
        return next(
            (
                d
                for d in datasets
                if d.get("slug") == settings.ULTRALYTICS_DATASET
                and d.get("username") == settings.ULTRALYTICS_USERNAME
            ),
            None,
        )

    @staticmethod
    def _persist(dataset_id: str) -> None:
        """Écrit l'ID résolu sur disque pour les étapes suivantes."""
        settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.DATASET_ID_FILE.write_text(dataset_id)


def run() -> None:
    """Point d'entrée de l'étape d'extraction."""
    try:
        DatasetResolver().resolve()
    except Exception as exc:
        logger.error("Échec de l'extraction : %s", exc)
        sys.exit(1)

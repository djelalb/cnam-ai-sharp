"""
Module d'extraction de données pour le projet SHARP.
Résout le slug Ultralytics HUB en datasetID.
"""

import logging
import sys

import requests

from src.config import settings

# Configuration du logging centralisé
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetResolver:
    """Résolveur de slugs Ultralytics."""

    def __init__(self):
        self.headers = {"Authorization": f"Bearer {settings.ULTRALYTICS_API_KEY}"}
        self.api_url = "https://platform.ultralytics.com/api/datasets"

    def resolve(self) -> str:
        """Résout l'ID du dataset depuis la plateforme."""
        try:
            target = f"{settings.ULTRALYTICS_USERNAME}/{settings.ULTRALYTICS_DATASET}"
            logger.info(f"Résolution de {target}...")
            response = requests.get(self.api_url, headers=self.headers)
            response.raise_for_status()

            datasets = response.json().get("datasets", [])
            dataset = next(
                (
                    d
                    for d in datasets
                    if d.get("slug") == settings.ULTRALYTICS_DATASET
                    and d.get("username") == settings.ULTRALYTICS_USERNAME
                ),
                None,
            )

            if not dataset:
                raise ValueError("Dataset introuvable sur votre compte.")

            dataset_id = dataset.get("_id") or dataset.get("id")

            # Sauvegarde persistante
            settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(settings.DATASET_ID_FILE, "w") as f:
                f.write(dataset_id)

            logger.info(f"ID résolu et sauvegardé : {dataset_id}")
            return dataset_id

        except Exception as e:
            logger.error(f"Échec de l'extraction : {e}")
            sys.exit(1)


def run():
    """Lance l'étape d'extraction."""
    DatasetResolver().resolve()


if __name__ == "__main__":
    run()

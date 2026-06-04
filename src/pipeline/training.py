"""
Module d'entraînement Cloud pour le projet SHARP.
Utilise l'API REST d'Ultralytics pour piloter les GPU distants.
"""

import logging
import sys

import requests

from src.config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CloudTrainer:
    """Orchestrateur d'entraînement Cloud."""

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {settings.ULTRALYTICS_API_KEY}",
            "Content-Type": "application/json",
        }
        self.api_url = "https://platform.ultralytics.com/api"

    def _get_project_id(self) -> str:
        """Résout l'ID du projet."""
        url = f"{self.api_url}/projects"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        projects = response.json().get("projects", [])
        project = next(
            (
                p
                for p in projects
                if p.get("slug") == settings.ULTRALYTICS_PROJECT
                and p.get("username") == settings.ULTRALYTICS_USERNAME
            ),
            None,
        )
        if not project:
            raise ValueError(f"Projet {settings.ULTRALYTICS_PROJECT} introuvable.")

        return project.get("_id") or project.get("id")

    def train(self):
        """Lance le job d'entraînement Cloud."""
        try:
            # 1. Lecture de l'ID dataset
            if not settings.DATASET_ID_FILE.exists():
                raise FileNotFoundError("ID dataset manquant. Lancez l'extraction.")

            with open(settings.DATASET_ID_FILE, "r") as f:
                dataset_id = f.read().strip()

            # 2. Résolution projet
            project_id = self._get_project_id()

            # 3. Payload
            payload = {
                "name": settings.ULTRALYTICS_EXP_NAME,
                "datasetId": dataset_id,
                "projectId": project_id,
                "cfg": {
                    "model": settings.MODEL_VARIANT,
                    "epochs": settings.EPOCHS,
                    "patience": settings.PATIENCE,
                    "imgsz": 640,
                    "degrees": 15.0,
                    "hsv_v": 0.4,
                    "mosaic": 0.5,
                    "fliplr": 0.5,
                    "task": "detect",
                },
            }

            # 4. Envoi
            logger.info("Déclenchement du job Cloud...")
            response = requests.post(
                f"{self.api_url}/models", headers=self.headers, json=payload
            )
            response.raise_for_status()

            data = response.json()
            model_id = data.get("id") or data.get("data", {}).get("id")

            logger.info("✅ Entraînement Cloud lancé !")
            logger.info(
                f"🔗 Suivi : https://platform.ultralytics.com/models/{model_id}"
            )

        except Exception as e:
            logger.error(f"Échec de l'entraînement : {e}")
            sys.exit(1)


def run():
    """Lance l'étape d'entraînement."""
    CloudTrainer().train()


if __name__ == "__main__":
    run()

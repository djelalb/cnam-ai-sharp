"""
Étape 1 — EXTRACTION (locale).

Récupère le dataset annoté depuis Ultralytics HUB via l'API REST :
1. résout le *slug* lisible (``username/dataset``) en identifiant technique ;
2. demande un export et obtient une URL de téléchargement signée ;
3. télécharge le manifeste **NDJSON** localement (consommé par la préparation).

L'identifiant est également persisté car l'étape de training en a besoin.
"""

import logging
import sys
from pathlib import Path

import requests

from src.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 60


class DatasetExtractor:
    """Résout, exporte et télécharge le dataset depuis Ultralytics HUB."""

    def __init__(self) -> None:
        self.headers = {"Authorization": f"Bearer {settings.ULTRALYTICS_API_KEY}"}
        self.api_url = "https://platform.ultralytics.com/api/datasets"

    def extract(self) -> Path:
        """Télécharge le manifeste NDJSON et retourne son chemin local."""
        dataset_id = self._resolve_id()
        self._persist_id(dataset_id)

        export_url = self._request_export_url(dataset_id)
        self._download(export_url, settings.DATASET_EXPORT)

        logger.info("Manifeste téléchargé : %s", settings.DATASET_EXPORT)
        return settings.DATASET_EXPORT

    def _resolve_id(self) -> str:
        """Résout l'ID du dataset à partir de son slug."""
        target = f"{settings.ULTRALYTICS_USERNAME}/{settings.ULTRALYTICS_DATASET}"
        logger.info("Résolution de %s...", target)

        response = requests.get(self.api_url, headers=self.headers, timeout=_TIMEOUT)
        response.raise_for_status()

        dataset = next(
            (
                d
                for d in response.json().get("datasets", [])
                if d.get("slug") == settings.ULTRALYTICS_DATASET
                and d.get("username") == settings.ULTRALYTICS_USERNAME
            ),
            None,
        )
        if not dataset:
            raise ValueError("Dataset introuvable sur le compte Ultralytics.")

        dataset_id = dataset.get("_id") or dataset.get("id")
        logger.info("ID résolu : %s", dataset_id)
        return dataset_id

    def _request_export_url(self, dataset_id: str) -> str:
        """Demande un export et renvoie l'URL de téléchargement signée."""
        response = requests.get(
            f"{self.api_url}/{dataset_id}/export",
            headers=self.headers,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["downloadUrl"]

    @staticmethod
    def _persist_id(dataset_id: str) -> None:
        """Écrit l'ID sur disque pour l'étape de training."""
        settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.DATASET_ID_FILE.write_text(dataset_id)

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        """Télécharge le NDJSON depuis l'URL signée (sans en-tête d'auth)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=_TIMEOUT)
        response.raise_for_status()
        dest.write_bytes(response.content)


def run() -> None:
    """Point d'entrée de l'étape d'extraction."""
    try:
        DatasetExtractor().extract()
    except Exception as exc:
        logger.error("Échec de l'extraction : %s", exc)
        sys.exit(1)

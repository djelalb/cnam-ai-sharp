"""
Module d'extraction de données pour le projet SHARP.
Résout le slug Ultralytics HUB en datasetID et prépare
l'environnement pour l'entraînement.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis .env.local
load_dotenv(".env.local")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UltralyticsHubResolver:
    """Classe pour résoudre les identifiants Ultralytics HUB."""

    API_BASE_URL = "https://platform.ultralytics.com/api"

    def __init__(self, api_key: Optional[str] = None):
        """Initialise le résolveur avec la clé API."""
        self.api_key = api_key or os.getenv("ULTRALYTICS_API_KEY")
        if not self.api_key:
            logger.error("Clé API manquante dans .env.local ou arguments.")
            sys.exit(1)

        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def resolve_id(self, username: str, slug: str) -> str:
        """
        Récupère le datasetId réel via l'API REST.
        """
        url = f"{self.API_BASE_URL}/datasets"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            datasets = data.get("datasets", [])
            dataset = next(
                (
                    d
                    for d in datasets
                    if d.get("slug") == slug and d.get("username") == username
                ),
                None,
            )

            if not dataset:
                raise ValueError(
                    f"Dataset '{username}/{slug}' introuvable sur votre compte."
                )

            dataset_id = dataset.get("_id") or dataset.get("id")
            logger.info(f"ID résolu pour {username}/{slug} : {dataset_id}")
            return dataset_id

        except Exception as e:
            logger.error(f"Erreur de résolution HUB : {e}")
            sys.exit(1)

    def save_id(self, dataset_id: str, output_dir: str = "data/raw") -> None:
        """Sauvegarde l'ID pour les étapes suivantes."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        id_file = out_path / "dataset_hub_id.txt"
        with open(id_file, "w") as f:
            f.write(dataset_id)
        logger.info(
            f"Dataset ID sauvegardé dans {id_file}. Prêt pour l'entraînement HUB."
        )


def main(argv: Optional[list[str]] = None) -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description="Résolution Dataset SHARP")
    parser.add_argument(
        "--username", default=os.getenv("ULTRALYTICS_USERNAME", "djelal-boudji")
    )
    parser.add_argument("--slug", default=os.getenv("ULTRALYTICS_DATASET", "hands"))
    parser.add_argument("--output", default="data/raw")

    args = parser.parse_args(argv)

    resolver = UltralyticsHubResolver()
    dataset_id = resolver.resolve_id(args.username, args.slug)
    resolver.save_id(dataset_id, args.output)


if __name__ == "__main__":
    main()

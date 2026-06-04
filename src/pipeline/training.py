"""
Module d'entraînement pour le projet SHARP.
Lance l'entraînement YOLO11 nano sur Ultralytics HUB avec
des paramètres optimisés pour la détection de mains.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from ultralytics import YOLO, hub

# Chargement des variables d'environnement
load_dotenv(".env.local")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SHARPTrainer:
    """Classe orchestrant l'entraînement du modèle sur Ultralytics HUB."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le trainer et s'authentifie sur le HUB.
        """
        self.api_key = api_key or os.getenv("ULTRALYTICS_API_KEY")
        if not self.api_key:
            logger.error(
                "Clé API manquante. Utilisez --api-key ou ULTRALYTICS_API_KEY."
            )
            sys.exit(1)

        try:
            hub.login(self.api_key)
            logger.info("Authentification HUB réussie.")
        except Exception as e:
            logger.error(f"Erreur d'authentification HUB : {e}")
            sys.exit(1)

    def _get_dataset_id(self, id_file_path: str = "data/raw/dataset_hub_id.txt") -> str:
        """
        Lit l'ID du dataset depuis le fichier local généré par l'étape d'extraction.
        """
        path = Path(id_file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ID introuvable : {id_file_path}. Lancez l'extraction d'abord."
            )

        with open(path, "r") as f:
            dataset_id = f.read().strip()

        if not dataset_id:
            raise ValueError("Le fichier d'ID est vide.")

        logger.info(f"ID du dataset récupéré : {dataset_id}")
        return dataset_id

    def train(self, dataset_id_file: str = "data/raw/dataset_hub_id.txt") -> None:
        """
        Lance l'entraînement avec les hyperparamètres cibles.
        """
        try:
            # 1. Récupération de l'identifiant
            dataset_id = self._get_dataset_id(dataset_id_file)

            # 2. Instanciation du modèle YOLO11 version Nano (optimisé pour les FPS)
            logger.info("Chargement du modèle YOLO11n...")
            model = YOLO("yolo11n.pt")

            # 3. Lancement de l'entraînement
            logger.info("Démarrage de l'entraînement sur Ultralytics HUB...")
            model.train(
                data=dataset_id,
                epochs=150,
                patience=30,
                imgsz=640,
                # Augmentation & Robustesse
                degrees=15.0,  # Rotation pour l'inclinaison des mains
                hsv_v=0.4,  # Résilience aux variations d'éclairage
                mosaic=0.5,  # Distinction de plusieurs mains
                fliplr=0.5,  # Symétrie horizontale
                # Environnement
                device="cpu",  # Forcer le CPU
                exist_ok=True,
            )

            logger.info("Entraînement terminé avec succès.")

        except Exception as e:
            logger.error(f"Échec critique lors de l'entraînement : {e}")
            sys.exit(1)


def main(argv: Optional[list[str]] = None) -> None:
    """Point d'entrée CLI pour l'entraînement."""
    parser = argparse.ArgumentParser(description="Entraînement du modèle SHARP")
    parser.add_argument("--api-key", type=str, help="Clé API Ultralytics")
    parser.add_argument(
        "--id-file",
        type=str,
        default="data/raw/dataset_hub_id.txt",
        help="Chemin vers le fichier contenant l'ID du dataset",
    )

    args = parser.parse_args(argv)

    trainer = SHARPTrainer(api_key=args.api_key)
    trainer.train(dataset_id_file=args.id_file)


if __name__ == "__main__":
    main()

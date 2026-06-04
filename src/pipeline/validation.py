"""
Module de validation physique des données pour le projet SHARP.
"""

import logging
from pathlib import Path

from PIL import Image

from src.config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, data_dir: Path = settings.RAW_DATA_DIR):
        self.data_dir = data_dir

    def validate(self):
        logger.info("--- Étape : VALIDATION ---")
        if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            logger.info("Statut : Délégué au Cloud (Aucune donnée physique locale).")
            return

        logger.info(f"Démarrage de la validation physique dans {self.data_dir}")
        img_exts = (".jpg", ".jpeg", ".png")
        images = [p for p in self.data_dir.rglob("*") if p.suffix.lower() in img_exts]

        valid_count = 0
        for img_p in images:
            try:
                with Image.open(img_p) as img:
                    img.verify()
                label_p = img_p.with_suffix(".txt")
                if label_p.exists():
                    valid_count += 1
            except Exception:
                logger.warning(f"Fichier corrompu supprimé : {img_p}")
                img_p.unlink()

        logger.info(f"Validation terminée. {valid_count} paires valides trouvées.")


def run():
    """Lance l'étape de validation."""
    DataValidator().validate()


if __name__ == "__main__":
    run()

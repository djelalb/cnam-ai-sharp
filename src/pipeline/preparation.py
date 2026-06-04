"""
Module de préparation et split physique des données pour le projet SHARP.
"""

import logging
import random
import shutil
from pathlib import Path

import yaml

from src.config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreparator:
    def __init__(
        self,
        input_dir: Path = settings.RAW_DATA_DIR,
        output_dir: Path = settings.PROCESSED_DATA_DIR,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        random.seed(42)

    def prepare(self):
        logger.info("--- Étape : PRÉPARATION ---")
        if not self.input_dir.exists() or not any(self.input_dir.rglob("*.jpg")):
            logger.info("Statut : Délégué au Cloud (Split plateforme actif).")
            return

        logger.info("Préparation du split physique (60/20/20)...")
        img_exts = (".jpg", ".jpeg", ".png")
        pairs = []
        for img_p in self.input_dir.rglob("*"):
            if img_p.suffix.lower() in img_exts:
                lbl_p = img_p.with_suffix(".txt")
                if lbl_p.exists():
                    pairs.append((img_p, lbl_p))

        if not pairs:
            logger.error("Aucune donnée à splitter !")
            return

        random.shuffle(pairs)
        n = len(pairs)
        splits = {
            "train": pairs[: int(n * 0.6)],
            "val": pairs[int(n * 0.6) : int(n * 0.8)],
            "test": pairs[int(n * 0.8) :],
        }

        for name, data in splits.items():
            (self.output_dir / name / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / name / "labels").mkdir(parents=True, exist_ok=True)
            for img, lbl in data:
                shutil.copy2(img, self.output_dir / name / "images" / img.name)
                shutil.copy2(lbl, self.output_dir / name / "labels" / lbl.name)

        # Génération du config.yaml local
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {
                0: "0_doigt",
                1: "1_doigt",
                2: "2_doigts",
                3: "3_doigts",
                4: "4_doigts",
                5: "5_doigts",
            },
        }
        config_file = self.output_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        logger.info(f"Split terminé. Configuration prête dans {config_file}")


def run():
    """Lance l'étape de préparation."""
    DataPreparator().prepare()


if __name__ == "__main__":
    run()

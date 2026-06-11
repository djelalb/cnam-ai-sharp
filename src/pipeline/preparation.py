"""
Étape 3 — PRÉPARATION (locale).

L'export Ultralytics est un manifeste **NDJSON** : une ligne d'en-tête
(``type=dataset`` avec ``class_names``) puis une ligne par image
(``url``, ``split`` et ``annotations.boxes`` au format YOLO normalisé).
Les images ne sont pas incluses : seule leur URL CDN l'est.

Cette étape matérialise donc l'arborescence YOLO attendue par Ultralytics :
téléchargement des images, écriture des labels ``.txt`` et génération
dynamique du ``config.yaml`` consommé par ``model.train()`` / ``model.val()``.
"""

import json
import logging
import sys
from pathlib import Path

import requests
import yaml

from src.config import settings

logger = logging.getLogger(__name__)

_IMG_TIMEOUT = 30


def build_data_config(names: dict[int, str], splits: set[str], root: Path) -> dict:
    """Construit le dict ``config.yaml`` attendu par Ultralytics.

    Les clés ``train`` et ``val`` sont obligatoires : un split absent est
    replié sur un split disponible. La clé ``test`` n'est ajoutée que si le
    split correspondant est effectivement présent dans l'export.
    """
    available = {s: f"{s}/images" for s in ("train", "val", "test") if s in splits}
    config = {
        "path": str(root),
        "train": available.get("train", available.get("val", "val/images")),
        "val": available.get("val", available.get("train", "train/images")),
        "names": names,
    }
    if "test" in available:
        config["test"] = available["test"]
    return config


class DatasetPreparator:
    """Convertit le manifeste NDJSON Ultralytics en dataset YOLO sur disque."""

    def __init__(
        self,
        export: Path = settings.DATASET_EXPORT,
        output: Path = settings.PROCESSED_DATA_DIR,
    ) -> None:
        self.export = export
        self.output = output

    def prepare(self, limit: int | None = None) -> None:
        """Matérialise le dataset YOLO et écrit le ``config.yaml``."""
        if not self.export.exists():
            logger.info("Préparation ignorée : export %s absent.", self.export)
            return

        records = self._read_manifest()
        names = {int(k): v for k, v in records[0]["class_names"].items()}
        images = [r for r in records if r.get("type") == "image"]
        if limit is not None:
            images = images[:limit]

        for index, record in enumerate(images, start=1):
            self._materialize(record)
            if index % 50 == 0:
                logger.info("... %d/%d images préparées", index, len(images))

        self._write_config(names, {r["split"] for r in images})
        logger.info(
            "Préparation terminée : %d images écrites dans %s",
            len(images),
            self.output,
        )

    def _read_manifest(self) -> list[dict]:
        """Charge les enregistrements NDJSON (en-tête + images)."""
        with self.export.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _materialize(self, record: dict) -> None:
        """Télécharge une image et écrit son label YOLO associé."""
        split = record["split"]
        image_path = self.output / split / "images" / record["file"]
        label_path = (self.output / split / "labels" / record["file"]).with_suffix(
            ".txt"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        if not image_path.exists():
            self._download(record["url"], image_path)

        boxes = record.get("annotations", {}).get("boxes", [])
        label_path.write_text(
            "".join(
                f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes
            )
        )

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        """Récupère une image depuis son URL CDN signée."""
        response = requests.get(url, timeout=_IMG_TIMEOUT)
        response.raise_for_status()
        dest.write_bytes(response.content)

    def _write_config(self, names: dict[int, str], splits: set[str]) -> None:
        """Écrit le ``config.yaml`` Ultralytics dans le dossier de sortie."""
        config = build_data_config(names, splits, self.output.resolve())
        (self.output / "config.yaml").write_text(yaml.dump(config, sort_keys=False))


def run() -> None:
    """Point d'entrée de l'étape de préparation."""
    try:
        DatasetPreparator().prepare()
    except Exception as exc:
        logger.error("Échec de la préparation : %s", exc)
        sys.exit(1)

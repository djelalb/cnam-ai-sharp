"""
Service d'inférence découplé du transport (API/WS).
"""

import logging
import re
from typing import Any, Dict

import numpy as np
from ultralytics import YOLO

from src.config import settings

logger = logging.getLogger(__name__)


def _finger_count(class_name: str) -> int:
    """Dérive le nombre de doigts depuis le nom de classe.

    Robuste aux deux conventions de nommage rencontrées
    (``"3"`` comme ``"3_doigts"``/``"3_fingers"``) : on extrait
    le premier entier présent dans le nom plutôt que de se reposer
    sur l'ordre des indices de classes.
    """
    match = re.search(r"\d+", class_name)
    return int(match.group()) if match else 0


class InferenceService:
    """Gère le cycle de vie et l'exécution du modèle YOLO."""

    def __init__(self):
        self.model: YOLO | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Charge le modèle en mémoire."""
        try:
            if not settings.MODEL_PATH.exists():
                logger.warning(f"Modèle non trouvé à {settings.MODEL_PATH}")
                return
            self.model = YOLO(settings.MODEL_PATH)
            logger.info(f"Modèle YOLO chargé depuis {settings.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Exécute l'inférence avec les paramètres de haute précision."""
        if self.model is None:
            return {"boxes": [], "total_fingers": 0}

        results = self.model(
            frame,
            imgsz=settings.IMG_SIZE,
            conf=settings.CONFIDENCE,
            iou=settings.IOU_THRESHOLD,
            max_det=settings.MAX_DET,
            verbose=False,
        )
        detections = []
        total_fingers = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = r.names[cls]
                fingers = _finger_count(class_name)
                total_fingers += fingers
                detections.append(
                    {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": round(conf, 2),
                        "class": cls,
                        "fingers": fingers,
                        "label": f"{fingers} fingers",
                    }
                )

        return {"boxes": detections, "total_fingers": total_fingers}

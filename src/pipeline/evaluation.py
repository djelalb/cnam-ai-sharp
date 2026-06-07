"""
Étape 5 — ÉVALUATION sur le jeu de test (déléguée, repliable en local).

L'entraînement cloud évalue automatiquement le meilleur modèle sur le split
*test* ; les métriques (mAP50, mAP50-95, précision, rappel) sont conservées
dans l'experiment tracking Ultralytics.

Si un dataset est disponible localement (``DATA_CONFIG`` présent), l'étape
rejoue l'évaluation officielle via ``model.val(split="test")`` — la méthode
exigée par le sujet — afin de pouvoir reproduire les métriques hors ligne.
"""

import logging

from src.config import settings

logger = logging.getLogger(__name__)


def run() -> None:
    """Évalue localement si possible, sinon trace la délégation au cloud."""
    if not (settings.MODEL_PATH.exists() and settings.DATA_CONFIG.exists()):
        logger.info(
            "Évaluation déléguée : model.val() exécutée sur le split test lors "
            "de l'entraînement cloud ; métriques dans l'experiment tracking."
        )
        return

    from ultralytics import YOLO

    logger.info("Évaluation locale sur le split test (%s)...", settings.DATA_CONFIG)
    metrics = YOLO(settings.MODEL_PATH).val(
        data=str(settings.DATA_CONFIG), split="test"
    )
    logger.info(
        "mAP50=%.3f | mAP50-95=%.3f | précision=%.3f | rappel=%.3f",
        metrics.box.map50,
        metrics.box.map,
        metrics.box.mp,
        metrics.box.mr,
    )

"""
Étape 5 — ÉVALUATION (locale si dataset présent, sinon déléguée).

Rejoue l'évaluation officielle via ``model.val()`` — la méthode exigée par le
sujet — sur le split défini par ``settings.EVAL_SPLIT``. La plateforme
n'exporte que train/val : l'évaluation se fait donc sur le split de validation.

Si aucun dataset n'a été préparé localement, l'étape rappelle que
l'entraînement cloud évalue automatiquement le modèle et conserve les
métriques dans l'experiment tracking.
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

    logger.info(
        "Évaluation locale sur le split '%s' (%s)...",
        settings.EVAL_SPLIT,
        settings.DATA_CONFIG,
    )
    metrics = YOLO(settings.MODEL_PATH).val(
        data=str(settings.DATA_CONFIG), split=settings.EVAL_SPLIT
    )
    logger.info(
        "mAP50=%.3f | mAP50-95=%.3f | précision=%.3f | rappel=%.3f",
        metrics.box.map50,
        metrics.box.map,
        metrics.box.mp,
        metrics.box.mr,
    )

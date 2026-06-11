"""
Étape 5 — ÉVALUATION (locale, sur le run entraîné).

Télécharge les poids du run à évaluer (nom passé en argument, ex: ``exp-14``,
sinon demandé de façon interactive) puis rejoue l'évaluation officielle via
``model.val()`` — la méthode exigée par le sujet — sur le split de test.
"""

import logging
import sys

from src.config import settings
from src.pipeline import hub

logger = logging.getLogger(__name__)


def run(model_name: str | None = None) -> None:
    """Évalue le run ``model_name`` (demandé si absent) sur le split de test."""
    if not settings.DATA_CONFIG.exists():
        logger.info("Évaluation ignorée : dataset non préparé (lancez 'preparation').")
        return

    name = model_name or _prompt_model_name()
    if not name:
        logger.info("Aucun run fourni : évaluation annulée.")
        return

    try:
        _evaluate(name)
    except Exception as exc:
        logger.error("Échec de l'évaluation : %s", exc)
        sys.exit(1)


def _evaluate(name: str) -> None:
    """Télécharge les poids du run et journalise les métriques de ``test``."""
    weights = hub.download_model(name, settings.MODELS_DIR)

    from ultralytics import YOLO

    logger.info("Évaluation de %s sur le split '%s'...", name, settings.EVAL_SPLIT)
    metrics = YOLO(weights).val(
        data=str(settings.DATA_CONFIG), split=settings.EVAL_SPLIT
    )
    logger.info(
        "mAP50=%.3f | mAP50-95=%.3f | précision=%.3f | rappel=%.3f",
        metrics.box.map50,
        metrics.box.map,
        metrics.box.mp,
        metrics.box.mr,
    )


def _prompt_model_name() -> str:
    """Demande interactivement le nom du run (ex: ``exp-14``)."""
    try:
        return input("Nom du run à évaluer (ex: exp-14) : ").strip()
    except EOFError:
        return ""

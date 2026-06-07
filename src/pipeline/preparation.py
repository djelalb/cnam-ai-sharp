"""
Étape 3 — PRÉPARATION (déléguée).

Le dataset est déjà au format YOLO et le découpage train/val/test
(60/20/20, seed 42) est réalisé directement sur la plateforme Ultralytics,
qui génère également le ``config.yaml`` consommé par ``model.train()``.
Aucune préparation locale n'est donc nécessaire.
"""

import logging

logger = logging.getLogger(__name__)


def run() -> None:
    """Trace la délégation de la préparation/split à la plateforme."""
    logger.info(
        "Préparation déléguée : format YOLO, split train/val/test "
        "(60/20/20, seed 42) et config.yaml gérés sur la plateforme."
    )

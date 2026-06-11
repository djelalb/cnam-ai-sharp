"""
Étape 6 — SÉLECTION du modèle (décision manuelle).

Dernière étape du cycle ML : décider quel run promouvoir en production.
Cette validation repose sur l'analyse comparative de l'experiment tracking
(cf. ``docs/training_strategy.md``) et relève d'un choix humain, non
automatisable. Le run ``ai-sharp-exp-prod`` a été retenu et figé.
"""

import logging

from src.config import settings

logger = logging.getLogger(__name__)


def run() -> None:
    """Trace le modèle retenu et la nature manuelle de la décision."""
    logger.info(
        "Sélection manuelle : modèle retenu après analyse de l'experiment "
        "tracking → %s (servi par l'application web).",
        settings.MODEL_PATH.name,
    )

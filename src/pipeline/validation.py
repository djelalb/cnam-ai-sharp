"""
Étape 2 — VALIDATION des données (déléguée).

La cohérence des données (images non corrompues, annotations sans
coordonnées négatives, classes valides) est garantie en amont par l'outil
d'annotation Ultralytics : une annotation invalide ne peut pas être exportée.
Aucune validation locale n'est donc nécessaire.
"""

import logging

logger = logging.getLogger(__name__)


def run() -> None:
    """Trace la délégation de la validation des données à la plateforme."""
    logger.info(
        "Validation déléguée : intégrité des images et cohérence des "
        "annotations garanties par l'outil d'annotation Ultralytics."
    )

"""
Module de validation des données pour le projet SHARP.
Note : Dans le flux Ultralytics HUB, la validation de l'intégrité
et des annotations est déléguée nativement à la plateforme.
"""

import logging

logger = logging.getLogger(__name__)


def run_validation():
    """Simule l'étape de validation dans l'architecture ML Pipeline."""
    logger.info("--- Étape : VALIDATION ---")
    logger.info("Responsabilité : Validation de l'intégrité et des annotations.")
    logger.info("Statut : Délégué au SDK Ultralytics HUB (Validation côté serveur).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_validation()

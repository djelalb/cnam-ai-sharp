"""
Module de préparation des données pour le projet SHARP.
Note : Dans le flux Ultralytics HUB, le split (train/val/test)
est géré nativement par la plateforme lors du versionnement du dataset.
"""

import logging

logger = logging.getLogger(__name__)


def run_preparation():
    """Simule l'étape de préparation dans l'architecture ML Pipeline."""
    logger.info("--- Étape : PRÉPARATION ---")
    logger.info("Responsabilité : Split déterministe et génération de config.yaml.")
    logger.info("Statut : Délégué au SDK Ultralytics HUB (Split plateforme actif).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_preparation()

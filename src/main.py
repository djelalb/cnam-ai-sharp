"""
Orchestrateur principal du projet SHARP.
Permet de piloter les étapes de la pipeline ML.
"""

import argparse
import logging
import sys

from src.pipeline import extraction, preparation, training, validation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SHARP Pipeline Orchestrator")
    parser.add_argument(
        "step",
        choices=["extraction", "validation", "preparation", "training", "all"],
        help="Étape de la pipeline à exécuter",
    )

    args = parser.parse_args()

    try:
        if args.step == "extraction" or args.step == "all":
            extraction.run()

        if args.step == "validation" or args.step == "all":
            validation.run()

        if args.step == "preparation" or args.step == "all":
            preparation.run()

        if args.step == "training" or args.step == "all":
            training.run()

    except KeyboardInterrupt:
        logger.info("\nArrêt par l'utilisateur.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

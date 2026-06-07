"""
Orchestrateur de la pipeline ML du projet SHARP.

Déroule les 6 étapes du cycle ML (cf. ``src.pipeline``). Certaines étapes sont déléguées
à la plateforme Ultralytics etsont tracées plutôt qu'exécutées localement, afin
de rendre le pipeline lisible de bout en bout.
"""

import argparse
import logging
import sys

from src.pipeline import PIPELINE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHARP Pipeline Orchestrator")
    parser.add_argument(
        "step",
        choices=[stage.key for stage in PIPELINE] + ["all"],
        help="Étape de la pipeline à exécuter ('all' pour tout dérouler)",
    )
    args = parser.parse_args()

    try:
        for index, stage in enumerate(PIPELINE, start=1):
            if args.step not in ("all", stage.key):
                continue
            logger.info(
                "[%d/%d] %s (%s)",
                index,
                len(PIPELINE),
                stage.name.upper(),
                stage.kind.value,
            )
            stage.run()
    except KeyboardInterrupt:
        logger.info("Arrêt par l'utilisateur.")
        sys.exit(0)


if __name__ == "__main__":
    main()

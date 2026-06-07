"""
Pipeline ML du projet SHARP.

Le cycle de vie ML comporte **6 étapes**, une par module (SRP). Le sujet
autorise l'usage de la plateforme Ultralytics : le dataset y est annoté et
l'entraînement GPU y est exécuté. Le reste est traité localement — notamment
la préparation, car l'export de la plateforme est un manifeste NDJSON qu'il
faut convertir en arborescence YOLO. Le ``Training`` reste piloté localement
mais s'exécute sur le cloud ; la ``Sélection`` finale est une décision humaine.

Statut de chaque étape :

==  ===========  ==========  ==========================================
#   Étape        Statut      Réalisation
==  ===========  ==========  ==========================================
1   Extraction   local       API HUB → télécharge le manifeste NDJSON
2   Validation   local       cohérence des annotations du manifeste
3   Préparation  local       manifeste NDJSON → arborescence YOLO
4   Training     cloud       déclenche le job d'entraînement GPU cloud
5   Évaluation   local       télécharge le run choisi → model.val()
6   Sélection    manuel      run retenu après analyse du tracking
==  ===========  ==========  ==========================================
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from src.pipeline import (
    evaluation,
    extraction,
    preparation,
    selection,
    training,
    validation,
)


class StageKind(str, Enum):
    """Où l'étape est réellement exécutée."""

    LOCAL = "local"  # par ce dépôt
    CLOUD = "cloud"  # sur l'infrastructure GPU Ultralytics
    MANUAL = "manuel"  # décision humaine


@dataclass(frozen=True)
class Stage:
    """Une étape de la pipeline ML : son identité et la fonction qui la lance."""

    key: str
    name: str
    kind: StageKind
    run: Callable[..., None]


#: Définition ordonnée des 6 étapes (source de vérité de l'orchestrateur).
PIPELINE: tuple[Stage, ...] = (
    Stage("extraction", "Extraction", StageKind.LOCAL, extraction.run),
    Stage("validation", "Validation", StageKind.LOCAL, validation.run),
    Stage("preparation", "Préparation", StageKind.LOCAL, preparation.run),
    Stage("training", "Training", StageKind.CLOUD, training.run),
    Stage("evaluation", "Évaluation", StageKind.LOCAL, evaluation.run),
    Stage("selection", "Sélection", StageKind.MANUAL, selection.run),
)

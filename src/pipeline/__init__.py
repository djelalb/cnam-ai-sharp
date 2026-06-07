"""
Pipeline ML du projet SHARP.

Le cycle de vie ML comporte **6 étapes**. Le sujet autorise explicitement
l'usage de la plateforme Ultralytics : en conséquence, la gestion du dataset
et l'entraînement GPU y sont réalisés. La majorité des étapes sont donc
*déléguées* à la plateforme et ne contiennent pas de code local — ce choix
est assumé et documenté, chaque étape restant matérialisée par un module
dédié (un fichier par responsabilité).

Statut de chaque étape :

==  ===========  ==========  ==========================================
#   Étape        Statut      Réalisation
==  ===========  ==========  ==========================================
1   Extraction   local       résout l'ID du dataset hébergé
2   Validation   délégué     cohérence des annotations à l'annotation
3   Préparation  délégué     format YOLO + split 60/20/20 (seed 42)
4   Training     local       déclenche le job d'entraînement GPU cloud
5   Évaluation   délégué     model.val() sur test → experiment tracking
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
    """Qui exécute réellement l'étape."""

    LOCAL = "local"  # exécutée par ce dépôt
    DELEGATED = "délégué"  # réalisée par la plateforme Ultralytics
    MANUAL = "manuel"  # décision humaine via l'experiment tracking


@dataclass(frozen=True)
class Stage:
    """Une étape de la pipeline ML, ainsi que la façon dont elle est honorée."""

    key: str
    name: str
    kind: StageKind
    run: Callable[[], None]


#: Définition ordonnée des 6 étapes (source de vérité de l'orchestrateur).
PIPELINE: tuple[Stage, ...] = (
    Stage("extraction", "Extraction", StageKind.LOCAL, extraction.run),
    Stage("validation", "Validation", StageKind.DELEGATED, validation.run),
    Stage("preparation", "Préparation", StageKind.DELEGATED, preparation.run),
    Stage("training", "Training", StageKind.LOCAL, training.run),
    Stage("evaluation", "Évaluation", StageKind.DELEGATED, evaluation.run),
    Stage("selection", "Sélection", StageKind.MANUAL, selection.run),
)

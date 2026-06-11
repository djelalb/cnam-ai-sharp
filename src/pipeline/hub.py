"""Client REST minimal pour la plateforme Ultralytics (HUB).

Centralise les accès partagés par plusieurs étapes : résolution du projet
et téléchargement des poids d'un run entraîné.
"""

import logging
from pathlib import Path

import requests

from src.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://platform.ultralytics.com/api"
_TIMEOUT = 60


def headers() -> dict[str, str]:
    """En-tête d'authentification de l'API."""
    return {"Authorization": f"Bearer {settings.ULTRALYTICS_API_KEY}"}


def resolve_project_id() -> str:
    """Résout l'ID du projet courant à partir de son slug."""
    response = requests.get(f"{BASE_URL}/projects", headers=headers(), timeout=_TIMEOUT)
    response.raise_for_status()

    project = next(
        (
            p
            for p in response.json().get("projects", [])
            if p.get("slug") == settings.ULTRALYTICS_PROJECT
            and p.get("username") == settings.ULTRALYTICS_USERNAME
        ),
        None,
    )
    if not project:
        raise ValueError(f"Projet {settings.ULTRALYTICS_PROJECT} introuvable.")
    return project.get("_id") or project.get("id")


def download_model(name: str, dest_dir: Path) -> Path:
    """Télécharge les poids ``.pt`` du run ``name`` et renvoie le chemin local."""
    model_id = _resolve_model_id(name)
    weights = _find_weights_file(model_id)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / weights["name"]
    logger.info("Téléchargement des poids %s...", weights["name"])
    dest.write_bytes(requests.get(weights["downloadUrl"], timeout=_TIMEOUT).content)
    return dest


def _resolve_model_id(name: str) -> str:
    """Résout l'ID d'un run à partir de son slug (ex: ``ai-sharp-exp-prod``)."""
    response = requests.get(
        f"{BASE_URL}/models",
        headers=headers(),
        params={"projectId": resolve_project_id()},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()

    model = next(
        (m for m in response.json().get("models", []) if m.get("slug") == name), None
    )
    if not model:
        raise ValueError(f"Run '{name}' introuvable dans le projet.")
    return model.get("_id") or model.get("id")


def _find_weights_file(model_id: str) -> dict:
    """Récupère le fichier de poids ``.pt`` téléchargeable d'un run."""
    response = requests.get(
        f"{BASE_URL}/models/{model_id}/files", headers=headers(), timeout=_TIMEOUT
    )
    response.raise_for_status()

    weights = next(
        (
            f
            for f in response.json().get("files", [])
            if f["name"].endswith(".pt") and f.get("downloadUrl")
        ),
        None,
    )
    if not weights:
        raise ValueError(f"Aucun poids .pt téléchargeable pour le run {model_id}.")
    return weights

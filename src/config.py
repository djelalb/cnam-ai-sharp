"""
Configuration centrale du projet SHARP utilisant Pydantic.
Gère la validation des variables d'environnement et les chemins de base.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres globaux du projet."""

    # Authentification
    ULTRALYTICS_API_KEY: str = Field(..., min_length=10)
    ULTRALYTICS_USERNAME: str = "djelal-boudji"

    # Dataset & Projet
    ULTRALYTICS_DATASET: str = "hands"
    ULTRALYTICS_PROJECT: str = "ai-sharp"
    ULTRALYTICS_EXP_NAME: str = "SHARP_Training"

    # --- Hyperparamètres d'IA (V2 Optimisée) ---
    MODEL_VARIANT: str = "yolo11n.pt"
    EPOCHS: int = 200
    PATIENCE: int = 50
    IMG_SIZE: int = 640
    BATCH_SIZE: int = 16

    # --- Augmentation de données (Recall & Robustness Boost) ---
    AUG_DEGREES: float = 25.0  # Augmenté pour la flexibilité angulaire
    AUG_HSV_V: float = 0.6  # Augmenté pour la résilience lumineuse
    AUG_MOSAIC: float = 1.0  # Maximisé pour la détection multi-mains
    AUG_FLIPLR: float = 0.5  # Conservé pour la symétrie

    # Chemins locaux

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # Fichiers techniques
    DATASET_ID_FILE: Path = RAW_DATA_DIR / "dataset_hub_id.txt"

    # Configuration du chargement
    model_config = SettingsConfigDict(
        env_file=".env.local", env_file_encoding="utf-8", extra="ignore"
    )


# Instance globale pour accès facile
settings = Settings()

"""
Configuration centralisée et typée via Pydantic.
Seule source de vérité pour le projet.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres globaux validés à l'exécution."""

    # Auth & Platform
    ULTRALYTICS_API_KEY: str | None = Field(None)
    ULTRALYTICS_USERNAME: str = "djelal-boudji"
    ULTRALYTICS_PROJECT: str = "ai-sharp"
    ULTRALYTICS_DATASET: str = "hands"
    ULTRALYTICS_EXP_NAME: str = "ai-sharp-exp-2"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"  # tous les poids .pt vivent ici
    DATASET_ID_FILE: Path = RAW_DATA_DIR / "dataset_hub_id.txt"
    DATASET_EXPORT: Path = RAW_DATA_DIR / "hands.ndjson"  # manifeste exporté
    DATA_CONFIG: Path = PROCESSED_DATA_DIR / "config.yaml"

    # AI Model (modèle de production : serving + sélection)
    MODEL_PATH: Path = MODELS_DIR / "ai-sharp-exp-2.pt"
    MODEL_VARIANT: str = "yolo11s.pt"
    IMG_SIZE: int = 640  # aligné sur la frame capturée (640x480)
    CONFIDENCE: float = 0.4
    IOU_THRESHOLD: float = 0.45
    MAX_DET: int = 4  # au plus 4 mains par frame
    EVAL_SPLIT: str = "val"  # split d'évaluation (la plateforme exporte train/val)

    # Hyperparamètres d'entraînement (augmentations incluses)
    EPOCHS: int = 200
    PATIENCE: int = 50
    AUG_DEGREES: float = 15.0
    AUG_HSV_V: float = 0.6
    AUG_MOSAIC: float = 0.7
    AUG_MIXUP: float = 0.1
    AUG_FLIPLR: float = 0.5

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

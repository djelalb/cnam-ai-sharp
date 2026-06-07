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
    ULTRALYTICS_EXP_NAME: str = "ai-sharp-exp"

    # AI Model
    MODEL_PATH: Path = Path("exp-14.pt")
    MODEL_VARIANT: str = "yolo11n.pt"
    IMG_SIZE: int = 640  # Aligné sur la frame capturée (640x480)
    CONFIDENCE: float = 0.4  # Un peu plus sensible
    IOU_THRESHOLD: float = 0.45  # Supprime les boîtes en double (NMS)
    MAX_DET: int = 4  # Au plus 4 mains détectées par frame

    # Hyperparameters V3 (Balanced)
    EPOCHS: int = Field(200, validation_alias="ULTRALYTICS_EPOCHS")
    PATIENCE: int = Field(50, validation_alias="ULTRALYTICS_PATIENCE")
    AUG_DEGREES: float = 15.0
    AUG_HSV_V: float = 0.6
    AUG_MOSAIC: float = 0.7
    AUG_MIXUP: float = 0.1
    AUG_FLIPLR: float = 0.5

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    DATASET_ID_FILE: Path = RAW_DATA_DIR / "dataset_hub_id.txt"
    DATA_CONFIG: Path = PROCESSED_DATA_DIR / "config.yaml"

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

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
    ULTRALYTICS_API_KEY: str = Field(..., min_length=10)
    ULTRALYTICS_USERNAME: str = "djelal-boudji"
    ULTRALYTICS_PROJECT: str = "ai-sharp"
    ULTRALYTICS_DATASET: str = "hands"

    # AI Model
    MODEL_PATH: Path = Path("exp-14.pt")
    IMG_SIZE: int = 640
    CONFIDENCE: float = 0.5

    # Hyperparameters V3 (Balanced)
    EPOCHS: int = 200
    PATIENCE: int = 50
    AUG_DEGREES: float = 15.0
    AUG_HSV_V: float = 0.6
    AUG_MOSAIC: float = 0.7
    AUG_MIXUP: float = 0.1

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    DATASET_ID_FILE: Path = RAW_DATA_DIR / "dataset_hub_id.txt"

    model_config = SettingsConfigDict(
        env_file=".env.local", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

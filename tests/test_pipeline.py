"""
Tests unitaires de base pour valider l'environnement CI du projet SHARP.
"""

import os
import sys


def test_python_version():
    """Vérifie que la version de Python est bien la 3.12."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12


def test_config_loading():
    """
    Vérifie que la configuration peut être importée.
    Note: On utilise une variable d'env factice pour passer la validation Pydantic.
    """
    os.environ["ULTRALYTICS_API_KEY"] = "dummy_key_for_testing"
    try:
        from src.config import settings

        assert settings.ULTRALYTICS_PROJECT == "ai-sharp"
    except Exception as e:
        raise AssertionError(f"Échec de l'import de la config : {e}") from e


def test_project_structure():
    """Vérifie que l'arborescence src existe."""
    assert os.path.exists("src/pipeline")
    assert os.path.exists("src/main.py")

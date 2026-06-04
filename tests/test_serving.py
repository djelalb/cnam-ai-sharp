"""
Tests unitaires respectant les principes FIRST.
Fast, Independent, Repeatable, Self-Validating, Timely.
"""

import numpy as np
import pytest

from src.serving.api.inference import InferenceService


@pytest.fixture
def service():
    return InferenceService()


def test_inference_with_empty_frame(service):
    """Vérifie le comportement avec une frame noire (pas de mains)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = service.predict(frame)

    assert "total_fingers" in result
    assert "boxes" in result
    assert result["total_fingers"] == 0
    assert len(result["boxes"]) == 0


def test_config_immutability():
    """Vérifie que la config est bien chargée."""
    from src.config import settings

    assert settings.ULTRALYTICS_PROJECT == "ai-sharp"

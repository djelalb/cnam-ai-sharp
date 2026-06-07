"""Tests du service d'inférence (comportement, sans charger de modèle réel)."""

import numpy as np
import pytest

from src.serving.api.inference import InferenceService, _finger_count


class _FakeArray:
    """Imite un tenseur YOLO indexable exposant ``tolist()``."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, _index):
        return self

    def tolist(self):
        return self.value


class _FakeBox:
    def __init__(self, cls: int, conf: float, xyxy: list[float]):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = _FakeArray(xyxy)


class _FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeModel:
    """Modèle factice renvoyant des résultats prédéfinis."""

    def __init__(self, results):
        self._results = results

    def __call__(self, _frame, **_kwargs):
        return self._results


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "class_name, expected",
    [("0", 0), ("3", 3), ("5", 5), ("2_doigts", 2), ("4_fingers", 4)],
)
def test_finger_count_parses_class_name(class_name, expected):
    assert _finger_count(class_name) == expected


def test_predict_sums_fingers_across_all_hands(frame):
    names = {i: str(i) for i in range(6)}
    boxes = [_FakeBox(2, 0.9, [0, 0, 10, 10]), _FakeBox(3, 0.8, [5, 5, 20, 20])]
    service = InferenceService(model=_FakeModel([_FakeResult(boxes, names)]))

    result = service.predict(frame)

    assert result["total_fingers"] == 5
    assert [box["fingers"] for box in result["boxes"]] == [2, 3]
    assert result["boxes"][0]["label"] == "2 fingers"


def test_predict_returns_empty_when_no_hand_detected(frame):
    names = {i: str(i) for i in range(6)}
    service = InferenceService(model=_FakeModel([_FakeResult([], names)]))

    assert service.predict(frame) == {"boxes": [], "total_fingers": 0}


def test_predict_returns_empty_when_model_unavailable(frame):
    service = InferenceService(model=_FakeModel([]))
    service.model = None

    assert service.predict(frame) == {"boxes": [], "total_fingers": 0}

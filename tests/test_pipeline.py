"""Tests de la pipeline ML (préparation et validation des données)."""

import json

from src.pipeline.preparation import build_data_config
from src.pipeline.validation import ManifestValidator


def test_data_config_always_declares_train_and_val():
    """Ultralytics exige train ET val, même si seul 'val' est disponible."""
    config = build_data_config({0: "0", 1: "1"}, {"val"}, root="/data")

    assert {"train", "val", "test"} <= config.keys()
    assert config["names"] == {0: "0", 1: "1"}
    assert config["test"] == config["val"]


def test_data_config_maps_each_existing_split():
    config = build_data_config({0: "0"}, {"train", "val"}, root="/data")

    assert config["train"] == "train/images"
    assert config["val"] == "val/images"


def _write_manifest(tmp_path, boxes):
    """Écrit un manifeste NDJSON minimal (en-tête + une image)."""
    lines = [
        {"type": "dataset", "class_names": {"0": "0", "1": "1"}},
        {"type": "image", "width": 100, "height": 100, "annotations": {"boxes": boxes}},
    ]
    export = tmp_path / "manifest.ndjson"
    export.write_text("\n".join(json.dumps(line) for line in lines))
    return export


def test_validation_accepts_coherent_annotations(tmp_path):
    export = _write_manifest(tmp_path, [[0, 0.5, 0.5, 0.2, 0.2]])

    assert ManifestValidator(export).validate() == 0


def test_validation_flags_out_of_range_coordinates(tmp_path):
    export = _write_manifest(tmp_path, [[0, 0.5, 0.5, 1.5, 0.2]])  # largeur > 1

    assert ManifestValidator(export).validate() == 1


def test_validation_flags_unknown_class(tmp_path):
    export = _write_manifest(tmp_path, [[9, 0.5, 0.5, 0.2, 0.2]])  # classe inconnue

    assert ManifestValidator(export).validate() == 1

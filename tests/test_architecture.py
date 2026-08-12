"""Guard dependency direction and compatibility paths introduced by cleanup."""

from pathlib import Path


def test_video_generator_facade_preserves_public_types():
    from sgdjscc_lab.video.generation.contracts import GenerationRequest as NewRequest
    from sgdjscc_lab.video.generation.factory import build_generator as new_factory
    from sgdjscc_lab.video.video_generator import GenerationRequest, build_generator

    assert GenerationRequest is NewRequest
    assert build_generator is new_factory


def test_lower_layers_do_not_import_training_or_evaluators():
    root = Path(__file__).parents[1] / "src" / "sgdjscc_lab"
    checks = {
        root / "data" / "datasets.py": "from sgdjscc_lab.training",
        root / "guidance" / "object_extractor.py": (
            "from sgdjscc_lab.evaluators.object_preservation import _COCO_CLASSES"
        ),
        root / "acceleration" / "water_filling.py": (
            "from sgdjscc_lab.training.noise_schedule"
        ),
    }
    for path, forbidden in checks.items():
        assert forbidden not in path.read_text(encoding="utf-8"), path

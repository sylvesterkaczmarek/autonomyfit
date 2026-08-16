import pytest

from autonomyfit.tasks import normalize_task, task_ids, task_spec


def test_stage4_task_system_covers_priority_categories():
    assert set(task_ids()) == {
        "detection",
        "classification",
        "segmentation",
        "pose",
        "depth",
        "ocr",
        "vlm",
        "anomaly",
        "asr",
        "embedding",
    }


def test_task_aliases_are_normalized():
    assert normalize_task("object-detection") == "detection"
    assert normalize_task("audio") == "asr"
    assert normalize_task("pose_estimation") == "pose"
    assert normalize_task("automatic-speech-recognition") == "asr"
    assert task_spec("embeddings").id == "embedding"


def test_unknown_task_is_explicit():
    with pytest.raises(ValueError, match="unknown task"):
        normalize_task("robot-vla")

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    id: str
    label: str
    aliases: tuple[str, ...]
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    default_accuracy_direction: str = "higher"


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec("detection", "Object detection", ("object-detection", "detect"), ("image",), ("object-detections",)),
    TaskSpec("classification", "Image classification", ("image-classification", "classify"), ("image",), ("class-label",)),
    TaskSpec("segmentation", "Segmentation", ("segment", "instance-segmentation", "semantic-segmentation"), ("image",), ("mask",)),
    TaskSpec("pose", "Pose estimation", ("pose-estimation", "keypoints", "keypoint-detection"), ("image",), ("keypoints",)),
    TaskSpec("depth", "Depth estimation", ("depth-estimation", "monocular-depth"), ("image",), ("depth-map",)),
    TaskSpec("ocr", "Optical character recognition", ("optical-character-recognition", "text-recognition"), ("image",), ("text",)),
    TaskSpec("vlm", "Compact vision-language / multimodal", ("multimodal", "vision-language", "image-text"), ("image", "text"), ("text",)),
    TaskSpec("anomaly", "Anomaly detection", ("anomaly-detection", "visual-anomaly"), ("image",), ("anomaly-score",)),
    TaskSpec("asr", "Automatic speech recognition", ("speech-recognition", "automatic-speech-recognition", "audio"), ("audio",), ("text",), "lower"),
    TaskSpec("embedding", "Embeddings", ("embeddings", "representation", "feature-extraction"), ("image",), ("embedding",)),
)

_TASK_BY_ID = {item.id: item for item in TASK_SPECS}
_TASK_ALIASES = {
    alias.casefold(): item.id
    for item in TASK_SPECS
    for alias in (item.id, item.label, *item.aliases)
}


def task_ids() -> tuple[str, ...]:
    return tuple(item.id for item in TASK_SPECS)


def normalize_task(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    task_id = _TASK_ALIASES.get(normalized)
    if task_id is None:
        valid = ", ".join(task_ids())
        raise ValueError(f"unknown task {value!r}; available: {valid}")
    return task_id


def task_spec(value: str) -> TaskSpec:
    return _TASK_BY_ID[normalize_task(value)]

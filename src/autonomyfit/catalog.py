from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .models import AccuracyMetric, BenchmarkRecord, ModelProfile


def _read_json(name: str) -> Any:
    resource = files("autonomyfit.data").joinpath(name)
    return json.loads(resource.read_text(encoding="utf-8"))


def _load_document(path: Path | None, bundled_name: str) -> Any:
    if path is None:
        return _read_json(bundled_name)
    return json.loads(path.read_text(encoding="utf-8"))


def load_models(path: Path | None = None) -> list[ModelProfile]:
    raw = _load_document(path, "models.json")
    models: list[ModelProfile] = []
    for item in raw["models"]:
        accuracy = item.get("accuracy")
        models.append(
            ModelProfile(
                id=item["id"],
                display_name=item["display_name"],
                family=item["family"],
                task=item["task"],
                params_m=float(item["params_m"]),
                source_id=item["source_id"],
                source_url=item["source_url"],
                runtimes=tuple(item["runtimes"]),
                accuracy=AccuracyMetric(**accuracy) if accuracy else None,
                flops_b=float(item["flops_b"]) if item.get("flops_b") is not None else None,
                input_size=int(item["input_size"]) if item.get("input_size") else None,
                published_memory_gb=(
                    float(item["published_memory_gb"])
                    if item.get("published_memory_gb") is not None
                    else None
                ),
                memory_scope=item.get("memory_scope"),
                notes=item.get("notes"),
            )
        )
    validate_models(models)
    return models


def load_benchmarks() -> list[BenchmarkRecord]:
    raw = _read_json("benchmarks.json")
    return [BenchmarkRecord(**item) for item in raw["benchmarks"]]


def load_hardware_profiles() -> dict[str, dict[str, Any]]:
    raw = _read_json("hardware_profiles.json")
    profiles = raw["profiles"]
    return {item["id"]: item for item in profiles}


def validate_models(models: list[ModelProfile]) -> None:
    ids = [model.id for model in models]
    duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate model ids: {', '.join(duplicates)}")
    for model in models:
        if model.params_m <= 0:
            raise ValueError(f"{model.id}: params_m must be positive")
        if not model.runtimes:
            raise ValueError(f"{model.id}: at least one runtime is required")
        if model.accuracy and model.accuracy.value < 0:
            raise ValueError(f"{model.id}: accuracy value must be non-negative")

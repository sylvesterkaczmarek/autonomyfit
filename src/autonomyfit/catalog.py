from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .models import AccuracyMetric, BenchmarkRecord, ModelProfile, RegistryProvenance
from .registry import RegistryClient, models_from_registry, validate_registry_document


@dataclass(frozen=True)
class LoadedCatalog:
    models: tuple[ModelProfile, ...]
    provenance: RegistryProvenance


def _read_json(name: str) -> Any:
    resource = files("autonomyfit.data").joinpath(name)
    return json.loads(resource.read_text(encoding="utf-8"))


def _load_document(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_model(item: dict[str, Any]) -> ModelProfile:
    accuracy = item.get("accuracy")
    return ModelProfile(
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
        verification_status="legacy-custom",
    )


def _load_custom(path: Path) -> LoadedCatalog:
    raw = _load_document(path)
    if not isinstance(raw, dict):
        raise TypeError("custom catalog must be a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version == 2:
        validate_registry_document(raw)
        models = models_from_registry(raw)
        metadata = raw["registry"]
        provenance = RegistryProvenance(
            source="custom",
            registry_version=int(metadata["registry_version"]),
            generated_at=metadata["generated_at"],
            expires_at=metadata["expires_at"],
            signature_verified=False,
            registry_url=str(path),
        )
    elif schema_version == 1:
        items = raw.get("models")
        if not isinstance(items, list):
            raise ValueError("legacy custom catalog must contain a models list")
        models = [_legacy_model(item) for item in items]
        validate_models(models)
        provenance = RegistryProvenance(
            source="custom",
            signature_verified=False,
            registry_url=str(path),
            warning="legacy schema v1 custom catalog",
        )
    else:
        raise ValueError(f"unsupported custom catalog schema {schema_version!r}")
    return LoadedCatalog(tuple(models), provenance)


def load_model_catalog(
    path: Path | None = None,
    *,
    offline: bool = False,
    force_refresh: bool = False,
    client: RegistryClient | None = None,
) -> LoadedCatalog:
    if path is not None:
        return _load_custom(path)
    registry_client = client or RegistryClient()
    snapshot = registry_client.load(offline=offline, force=force_refresh)
    return LoadedCatalog(snapshot.models, snapshot.provenance)


def load_models(
    path: Path | None = None,
    *,
    offline: bool = False,
    client: RegistryClient | None = None,
) -> list[ModelProfile]:
    return list(load_model_catalog(path, offline=offline, client=client).models)


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

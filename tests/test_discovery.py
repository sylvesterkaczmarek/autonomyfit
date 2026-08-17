from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autonomyfit.cli import app
from autonomyfit.discovery import (
    HF_TAG_TASKS,
    HF_TASKS,
    DiscoveryCandidate,
    DiscoveryError,
    HuggingFaceAdapter,
    NvidiaAdapter,
    _license_status,
    apply_discovery,
    candidate_to_registry_model,
    deduplicate_candidates,
)

NOW = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)


class FakeTransport:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get_json(self, url, *, timeout=15.0):
        del timeout
        self.urls.append(url)
        for marker, value in self.responses:
            if marker in url:
                if isinstance(value, Exception):
                    raise value
                return json.loads(json.dumps(value))
        raise AssertionError(f"unexpected URL: {url}")


def _hf_detail(
    model_id="HuggingFaceTB/SmolVLM-500M-Instruct",
    *,
    sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    license_id="apache-2.0",
    params=500_000_000,
    pipeline_tag="image-text-to-text",
):
    card = {
        "license": license_id,
        "model_name": model_id.rsplit("/", 1)[-1],
        "base_model": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "base_model_relation": "finetune",
        "new_version": "HuggingFaceTB/SmolVLM2-500M-Instruct",
    }
    return {
        "id": model_id,
        "sha": sha,
        "createdAt": "2026-08-15T08:00:00Z",
        "lastModified": "2026-08-16T08:00:00Z",
        "downloads": 12345,
        "likes": 321,
        "pipeline_tag": pipeline_tag,
        "library_name": "transformers",
        "tags": ["transformers", "safetensors"],
        "safetensors": {"total": params} if params is not None else {},
        "cardData": card,
    }


def _hf_transport(detail, selected_pipeline="image-text-to-text"):
    summary = [{"id": detail["id"]}]
    responses = [
        (f"pipeline_tag={pipeline}", summary if pipeline == selected_pipeline else [])
        for pipeline in HF_TASKS
    ]
    responses.extend((f"filter={tag}", []) for tag in HF_TAG_TASKS)
    responses.append((f"/api/models/{detail['id']}", detail))
    return FakeTransport(responses)


def _ocr_transport(detail):
    summary = [{"id": detail["id"]}]
    responses = [(f"pipeline_tag={pipeline}", []) for pipeline in HF_TASKS]
    responses.append(("filter=text_recognition", summary))
    responses.append((f"/api/models/{detail['id']}", detail))
    return FakeTransport(responses)


def _candidate(**overrides):
    values = {
        "provider": "huggingface",
        "upstream_id": "Vendor/TinyVision-100M",
        "source_url": "https://huggingface.co/Vendor/TinyVision-100M",
        "publisher": "Vendor",
        "display_name": "TinyVision-100M",
        "family": "TinyVision",
        "variant": "100M",
        "task": "detection",
        "revision": "dddddddddddddddddddddddddddddddddddddddd",
        "release_date": "2026-08-15",
        "last_modified": "2026-08-16T09:00:00Z",
        "params_m": 100.0,
        "license_id": "apache-2.0",
        "library": "transformers",
        "runtimes": ("pytorch", "transformers"),
        "aliases": ("Vendor/TinyVision-100M",),
        "official_publisher": True,
    }
    values.update(overrides)
    return DiscoveryCandidate(**values)


def _registry():
    return json.loads(
        Path("registry/source/registry-v2.json").read_text(encoding="utf-8")
    )


def test_huggingface_discovers_new_compact_vlm():
    detail = _hf_detail()
    adapter = HuggingFaceAdapter(
        transport=_hf_transport(detail),
        trusted_publishers={"HuggingFaceTB"},
        limit_per_task=5,
    )
    records = adapter.discover(NOW)
    assert len(records) == 1
    model = records[0]
    assert model.task == "vlm"
    assert model.params_m == 500.0
    assert model.revision == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert model.lifecycle() == "SOURCE_VERIFIED"


def test_revision_identity_changes_with_revision():
    first = _candidate(revision="a")
    second = _candidate(revision="b")
    assert first.identity_key == second.identity_key
    assert first.revision_identity != second.revision_identity


def test_deduplication_merges_aliases_and_prefers_vendor_adapter():
    generic = _candidate(
        provider="huggingface",
        upstream_id="nvidia/TinyVision-100M",
        publisher="nvidia",
        aliases=("tinyvision",),
    )
    vendor = _candidate(
        provider="nvidia",
        upstream_id="nvidia/TinyVision-100M",
        publisher="nvidia",
        aliases=("nvidia/TinyVision-100M",),
    )
    result = deduplicate_candidates([generic, vendor])
    assert len(result) == 1
    assert result[0].provider == "nvidia"
    assert set(result[0].aliases) == {
        "tinyvision",
        "nvidia/TinyVision-100M",
    }


def test_malformed_huggingface_list_record_is_ignored():
    responses = [
        (f"pipeline_tag={pipeline}", [{"not_id": "broken"}] if pipeline == "object-detection" else [])
        for pipeline in HF_TASKS
    ]
    responses.extend((f"filter={tag}", []) for tag in HF_TAG_TASKS)
    transport = FakeTransport(responses)
    assert HuggingFaceAdapter(transport=transport).discover(NOW) == []


def test_missing_license_is_flagged_and_not_source_verified():
    detail = _hf_detail(license_id=None)
    adapter = HuggingFaceAdapter(
        transport=_hf_transport(detail),
        trusted_publishers={"HuggingFaceTB"},
    )
    model = adapter.discover(NOW)[0]
    assert model.license_id is None
    assert model.lifecycle() == "NORMALIZED"
    converted = candidate_to_registry_model(
        model,
        model_id="smolvlm-500m-instruct",
        checked_at="2026-08-16T10:00:00Z",
    )
    assert converted is None


def test_missing_parameter_metadata_stays_discovery_only():
    detail = _hf_detail(params=None)
    adapter = HuggingFaceAdapter(
        transport=_hf_transport(detail),
        trusted_publishers={"HuggingFaceTB"},
    )
    model = adapter.discover(NOW)[0]
    assert model.metadata_complete is False
    assert model.lifecycle() == "DISCOVERED"
    assert (
        candidate_to_registry_model(
            model,
            model_id="missing-params",
            checked_at="2026-08-16T10:00:00Z",
        )
        is None
    )


def test_unsupported_task_is_not_promoted():
    model = _candidate(task="audio")
    assert (
        candidate_to_registry_model(
            model,
            model_id="unsupported",
            checked_at="2026-08-16T10:00:00Z",
        )
        is None
    )


def test_nvidia_adapter_uses_official_huggingface_publisher_scope():
    detail = _hf_detail(
        model_id="nvidia/TinyVLM-1B",
        params=1_000_000_000,
    )
    adapter = NvidiaAdapter(
        transport=_hf_transport(detail),
        limit_per_task=2,
    )
    records = adapter.discover(NOW)
    assert records[0].provider == "nvidia"
    assert records[0].official_publisher is True
    assert any("author=nvidia" in url for url in adapter.transport.urls)


def test_stale_source_is_exposed_in_manifest():
    old = _candidate(last_modified="2024-01-01T00:00:00Z")
    result = apply_discovery(_registry(), [old], now=NOW)
    item = next(
        entry
        for entry in result.manifest["candidates"]
        if entry["identity_key"] == old.identity_key
    )
    assert item["source_stale"] is True


def test_missing_candidate_becomes_deprecated_tombstone():
    prior = _candidate().to_manifest_dict()
    previous = {
        "schema_version": 1,
        "generated_at": "2026-08-15T10:00:00Z",
        "candidates": [prior],
    }
    result = apply_discovery(
        _registry(),
        [],
        now=NOW,
        previous_manifest=previous,
    )
    tombstone = result.manifest["candidates"][0]
    assert tombstone["lifecycle"] == "DEPRECATED"
    assert tombstone["deprecated"] is True
    assert "not observed in latest discovery" in tombstone["warnings"]


def test_new_version_and_base_model_relationship_are_preserved():
    detail = _hf_detail()
    adapter = HuggingFaceAdapter(
        transport=_hf_transport(detail),
        trusted_publishers={"HuggingFaceTB"},
    )
    model = adapter.discover(NOW)[0]
    assert model.base_model == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert model.base_model_relation == "finetune"
    assert model.new_version == "HuggingFaceTB/SmolVLM2-500M-Instruct"


def test_registry_generation_is_deterministic():
    model = _candidate()
    first = apply_discovery(_registry(), [model], now=NOW)
    second = apply_discovery(_registry(), [model], now=NOW)
    assert first.registry == second.registry
    assert first.manifest == second.manifest


def test_no_change_path_for_incomplete_discovery_record():
    result = apply_discovery(
        _registry(),
        [_candidate(params_m=None, metadata_complete=False)],
        now=NOW,
    )
    assert result.changed is False
    assert result.freshness_refresh is False


def test_changed_path_increments_registry_version():
    registry = _registry()
    result = apply_discovery(registry, [_candidate()], now=NOW)
    assert result.changed is True
    assert result.registry["registry"]["registry_version"] == (
        registry["registry"]["registry_version"] + 1
    )
    assert any(item["display_name"] == "TinyVision-100M" for item in result.registry["models"])



def test_mirror_does_not_duplicate_or_replace_curated_canonical_source():
    registry = _registry()
    original = next(item for item in registry["models"] if item["id"] == "yolo26n")
    mirror = _candidate(
        upstream_id="ultralytics/yolo26n",
        source_url="https://huggingface.co/ultralytics/yolo26n",
        publisher="ultralytics",
        display_name="YOLO26n",
        family="YOLO26",
        variant="n",
        params_m=2.4,
        runtimes=("transformers",),
    )
    result = apply_discovery(registry, [mirror], now=NOW)
    matches = [item for item in result.registry["models"] if item["id"] == "yolo26n"]
    assert len(matches) == 1
    assert matches[0]["upstream"]["source_url"] == original["upstream"]["source_url"]
    assert matches[0]["compatibility"]["runtimes"] == original["compatibility"]["runtimes"]


def test_untrusted_complete_record_stays_out_of_signed_registry():
    model = _candidate(official_publisher=False)
    result = apply_discovery(_registry(), [model], now=NOW)
    assert not any(
        item["display_name"] == "TinyVision-100M"
        for item in result.registry["models"]
    )
    manifest = next(
        item
        for item in result.manifest["candidates"]
        if item["identity_key"] == model.identity_key
    )
    assert manifest["lifecycle"] == "NORMALIZED"


def test_restrictive_license_is_flagged():
    status, warning = _license_status("cc-by-nc-4.0")
    assert status == "restricted"
    assert warning


runner = CliRunner()


def test_models_command_filters_source_and_emits_json():
    result = runner.invoke(
        app,
        ["models", "--offline", "--source", "huggingface", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["models"]
    assert all(
        "huggingface" in item["source_id"].casefold()
        for item in payload["models"]
    )


def test_search_and_info_commands_work_offline():
    search = runner.invoke(
        app,
        ["search", "smolvlm", "--offline", "--json"],
    )
    assert search.exit_code == 0, search.output
    matches = json.loads(search.stdout)["models"]
    assert matches
    model_id = matches[0]["id"]

    info = runner.invoke(
        app,
        ["info", model_id, "--offline", "--json"],
    )
    assert info.exit_code == 0, info.output
    payload = json.loads(info.stdout)
    assert payload["model"]["id"] == model_id
    assert payload["model"]["license_status"]

def test_huggingface_short_revision_is_not_source_verified():
    detail = _hf_detail(sha="abc123")
    adapter = HuggingFaceAdapter(
        transport=_hf_transport(detail),
        trusted_publishers={"HuggingFaceTB"},
    )
    model = adapter.discover(NOW)[0]
    assert model.revision is None
    assert model.lifecycle() == "NORMALIZED"
    assert candidate_to_registry_model(
        model, model_id="short-revision", checked_at="2026-08-16T10:00:00Z"
    ) is None


@pytest.mark.parametrize(
    ("pipeline_tag", "expected_task", "expected_output"),
    [
        ("image-classification", "classification", "class-label"),
        ("image-segmentation", "segmentation", "mask"),
        ("keypoint-detection", "pose", "keypoints"),
        ("depth-estimation", "depth", "depth-map"),
        ("automatic-speech-recognition", "asr", "text"),
        ("image-feature-extraction", "embedding", "embedding"),
    ],
)
def test_huggingface_discovers_additional_canonical_tasks(
    pipeline_tag, expected_task, expected_output
):
    detail = _hf_detail(
        model_id="google/TinyEdge-100M",
        params=100_000_000,
        pipeline_tag=pipeline_tag,
    )
    adapter = HuggingFaceAdapter(
        transport=_hf_transport(detail, selected_pipeline=pipeline_tag),
        trusted_publishers={"google"},
    )
    model = adapter.discover(NOW)[0]
    assert model.task == expected_task
    assert model.lifecycle() == "SOURCE_VERIFIED"
    converted = candidate_to_registry_model(
        model, model_id=f"tiny-{expected_task}", checked_at="2026-08-16T10:00:00Z"
    )
    assert converted is not None
    assert expected_output in converted["modalities"]["output"]
    if expected_task == "asr":
        assert converted["modalities"]["input"] == ["audio"]


def test_ocr_requires_explicit_text_recognition_tag_and_compatible_pipeline():
    detail = _hf_detail(
        model_id="PaddlePaddle/TinyOCR-10M",
        params=10_000_000,
        pipeline_tag="image-to-text",
    )
    detail["tags"] = ["transformers", "safetensors", "text_recognition"]
    adapter = HuggingFaceAdapter(
        transport=_ocr_transport(detail),
        trusted_publishers={"PaddlePaddle"},
    )
    model = adapter.discover(NOW)[0]
    assert model.task == "ocr"
    converted = candidate_to_registry_model(
        model, model_id="tiny-ocr", checked_at="2026-08-16T10:00:00Z"
    )
    assert converted is not None
    assert converted["modalities"] == {"input": ["image"], "output": ["text"]}


def test_ocr_free_form_image_to_text_without_task_tag_is_not_discovered_as_ocr():
    detail = _hf_detail(
        model_id="PaddlePaddle/Captioner-10M",
        params=10_000_000,
        pipeline_tag="image-to-text",
    )
    detail["tags"] = ["transformers", "safetensors"]
    adapter = HuggingFaceAdapter(
        transport=_ocr_transport(detail),
        trusted_publishers={"PaddlePaddle"},
    )
    assert adapter.discover(NOW) == []


def test_partial_huggingface_query_failure_does_not_drop_other_tasks():
    detail = _hf_detail(
        model_id="google/TinyClassifier-100M",
        params=100_000_000,
        pipeline_tag="image-classification",
    )
    responses = []
    for pipeline in HF_TASKS:
        if pipeline == "depth-estimation":
            responses.append((f"pipeline_tag={pipeline}", DiscoveryError("temporary outage")))
        elif pipeline == "image-classification":
            responses.append((f"pipeline_tag={pipeline}", [{"id": detail["id"]}]))
        else:
            responses.append((f"pipeline_tag={pipeline}", []))
    responses.extend((f"filter={tag}", []) for tag in HF_TAG_TASKS)
    responses.append((f"/api/models/{detail['id']}", detail))
    records = HuggingFaceAdapter(transport=FakeTransport(responses)).discover(NOW)
    assert len(records) == 1
    assert records[0].task == "classification"


def test_malformed_huggingface_repo_id_is_ignored():
    detail = _hf_detail(model_id="../escape", pipeline_tag="image-classification")
    adapter = HuggingFaceAdapter(
        transport=_hf_transport(detail, selected_pipeline="image-classification")
    )
    assert adapter.discover(NOW) == []


def test_existing_huggingface_registry_source_is_revisited_even_when_not_trending():
    detail = _hf_detail(
        model_id="HuggingFaceTB/SeedModel-100M",
        params=100_000_000,
        pipeline_tag="image-text-to-text",
    )
    registry = {
        "models": [
            {
                "id": "seed-model",
                "task": "vlm",
                "upstream": {
                    "source_url": "https://huggingface.co/HuggingFaceTB/SeedModel-100M"
                },
            }
        ]
    }
    adapter = HuggingFaceAdapter(
        transport=FakeTransport([(f"/api/models/{detail['id']}", detail)]),
        trusted_publishers={"HuggingFaceTB"},
    )
    records = adapter.discover_registry(registry)
    assert len(records) == 1
    assert records[0].revision == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_short_direct_revision_cannot_be_promoted():
    model = _candidate(revision="mutable-main")
    assert model.lifecycle() == "NORMALIZED"
    assert candidate_to_registry_model(
        model, model_id="mutable", checked_at="2026-08-16T10:00:00Z"
    ) is None


def test_upstream_revision_change_invalidates_old_benchmark_status():
    registry = _registry()
    target = next(item for item in registry["models"] if item["id"] == "detr-resnet-50")
    target["verification"]["status"] = "benchmarked"
    target["evidence"]["benchmark_refs"] = ["old-exact-benchmark"]
    candidate = _candidate(
        upstream_id="facebook/detr-resnet-50",
        source_url="https://huggingface.co/facebook/detr-resnet-50",
        publisher="facebook",
        display_name="detr-resnet-50",
        family="detr-resnet",
        variant="50",
        revision="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        params_m=41.631008,
    )
    result = apply_discovery(registry, [candidate], now=NOW)
    updated = next(item for item in result.registry["models"] if item["id"] == "detr-resnet-50")
    assert updated["upstream"]["revision"] == candidate.revision
    assert updated["verification"]["status"] == "source_verified"
    assert updated["evidence"]["benchmark_refs"] == []

def test_slug_collision_from_unrelated_source_gets_unique_id():
    registry = _registry()
    existing = next(item for item in registry["models"] if item["id"] == "yolo26n")
    original_runtimes = list(existing["compatibility"]["runtimes"])
    collision = _candidate(
        upstream_id="Vendor/Unrelated-Model",
        source_url="https://huggingface.co/Vendor/Unrelated-Model",
        display_name="YOLO26n",
        family="Unrelated",
        variant="other",
        runtimes=("transformers",),
    )
    result = apply_discovery(registry, [collision], now=NOW)
    assert any(
        item["id"].startswith("yolo26n-")
        and item["upstream"]["source_url"] == collision.source_url
        for item in result.registry["models"]
    )
    preserved = next(item for item in result.registry["models"] if item["id"] == "yolo26n")
    assert preserved["compatibility"]["runtimes"] == original_runtimes

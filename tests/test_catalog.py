import json

from autonomyfit.catalog import (
    load_benchmarks,
    load_hardware_profiles,
    load_model_catalog,
    load_models,
)


def test_bundled_fallback_catalog_is_valid_and_unique():
    models = load_models(offline=True)
    assert len(models) >= 7
    assert len({model.id for model in models}) == len(models)
    assert {model.task for model in models} == {"detection", "vlm"}


def test_benchmarks_reference_known_models_and_hardware():
    model_ids = {model.id for model in load_models(offline=True)}
    hardware_ids = set(load_hardware_profiles())
    for benchmark in load_benchmarks():
        assert benchmark.model_id in model_ids
        assert benchmark.hardware_id in hardware_ids
        assert benchmark.latency_ms > 0
        assert benchmark.measured is True


def test_legacy_custom_catalog_remains_supported(tmp_path):
    document = {
        "schema_version": 1,
        "models": [
            {
                "id": "custom-detector",
                "display_name": "Custom Detector",
                "family": "custom",
                "task": "detection",
                "params_m": 5.0,
                "source_id": "test",
                "source_url": "https://example.com/model",
                "runtimes": ["onnx"],
            }
        ],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document))
    loaded = load_model_catalog(path)
    assert loaded.provenance.source == "custom"
    assert loaded.models[0].verification_status == "legacy-custom"


def test_schema_v2_custom_catalog_is_supported(tmp_path):
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "src/autonomyfit/data/fallback_registry.json"
    )
    path = tmp_path / "registry.json"
    path.write_text(source.read_text())
    loaded = load_model_catalog(path)
    assert loaded.provenance.source == "custom"
    assert loaded.provenance.registry_version == 1

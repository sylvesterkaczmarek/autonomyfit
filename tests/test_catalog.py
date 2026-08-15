from autonomyfit.catalog import load_benchmarks, load_hardware_profiles, load_models


def test_bundled_catalog_is_valid_and_unique():
    models = load_models()
    assert len(models) >= 7
    assert len({model.id for model in models}) == len(models)
    assert {model.task for model in models} == {"detection", "vlm"}


def test_benchmarks_reference_known_models_and_hardware():
    model_ids = {model.id for model in load_models()}
    hardware_ids = set(load_hardware_profiles())
    for benchmark in load_benchmarks():
        assert benchmark.model_id in model_ids
        assert benchmark.hardware_id in hardware_ids
        assert benchmark.latency_ms > 0
        assert benchmark.measured is True

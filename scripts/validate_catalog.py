from autonomyfit.catalog import load_benchmarks, load_hardware_profiles, load_models

models = load_models(offline=True)
benchmarks = load_benchmarks()
hardware = load_hardware_profiles()

assert models
assert benchmarks
assert hardware
assert len({model.id for model in models}) == len(models)
print(
    f"catalog valid: {len(models)} models, {len(benchmarks)} benchmarks, "
    f"{len(hardware)} hardware profiles"
)

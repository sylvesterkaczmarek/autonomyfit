# AutonomyFit

![AutonomyFit](assets/social/github-social-card-autonomyfit.png)

[![CI](https://github.com/sylvesterkaczmarek/autonomyfit/actions/workflows/ci.yml/badge.svg)](https://github.com/sylvesterkaczmarek/autonomyfit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

AutonomyFit detects local edge-AI hardware, tracks a continuously updated signed model registry, and ranks models against memory, latency, throughput, accuracy, power and runtime constraints. Performance claims are tied to explicit evidence rather than model names alone.

## Install

```bash
pip install autonomyfit
```

## Quick start

```bash
autonomyfit scan
autonomyfit registry update
autonomyfit models
autonomyfit recommend --task detection --fps 30 --latency-ms 40
```

A known target can be screened without the device attached:

```bash
autonomyfit recommend --hardware-profile jetson-orin-nx-16gb --fps 200 --latency-ms 5
```

The published YOLO26n / Jetson Orin NX / TensorRT FP16 reference is 4.13 ms, but AutonomyFit 0.4 deliberately treats it as **vendor reference evidence**, not automatic proof for every YOLO26n artifact. Because that table does not pin an exact model revision and artifact SHA-256, a performance constraint returns `BENCHMARK_REQUIRED` until exact evidence is available.

## Evidence-aware decisions

| Outcome | Meaning |
|---|---|
| `VERIFIED_FIT` | Exact identity-matched local or standardized evidence satisfies the requested constraints. |
| `FEASIBLE` | Compatibility and memory screening pass with no unresolved performance constraint. |
| `BENCHMARK_REQUIRED` | A performance/power constraint needs exact target-device evidence. |
| `CONSTRAINT_FAIL` | Exact applicable evidence or catalogued constraints show a violation. |
| `NO_FIT` | A hard compatibility or memory screen fails. |

Evidence quality is explicit:

```text
local-measured
standardized
vendor-published
third-party-reproducible
metadata-estimate
```

A result can be useful without being proof. JSON and human output state whether benchmark identity is exact or context-only, along with model revision, artifact hash, hardware, runtime, precision, source date and evidence quality.

See [docs/evidence.md](docs/evidence.md).

## Continuously updated registry

The PyPI package is the decision engine. Model intelligence is a separately versioned, signed registry that can update without a new package release.

```bash
autonomyfit registry status
autonomyfit registry update
autonomyfit search detr-resnet-50
autonomyfit info eagle2-1b
autonomyfit models --source nvidia
```

The registry client verifies Sigstore identity, schema/freshness and local rollback state, caches the last accepted registry and supports offline fallback:

```bash
autonomyfit recommend --offline --task detection
```

See [docs/registry.md](docs/registry.md) and [docs/discovery.md](docs/discovery.md).

## Exact artifact matching

For a deployable artifact, include its identity in the recommendation:

```bash
autonomyfit recommend \
  --model-id my-model \
  --model-revision FULL_REVISION \
  --artifact model.onnx \
  --runtime onnx \
  --precision fp16 \
  --latency-ms 10
```

AutonomyFit hashes the artifact with SHA-256. Known revision, artifact, hardware, runtime or precision mismatches do not silently share benchmark evidence.

## Local benchmarking

Check backend availability:

```bash
autonomyfit benchmark-backends
```

For ONNX Runtime:

```bash
pip install 'autonomyfit[benchmark]'
autonomyfit benchmark model.onnx \
  --model-id my-model \
  --model-revision FULL_REVISION \
  --iterations 100 \
  --warmup 20 \
  -o result.json
```

Then validate and import the report:

```bash
autonomyfit benchmark-inspect result.json
autonomyfit benchmark-import result.json
```

A local report records the exact artifact SHA-256, model revision, hardware identity, runtime/provider versions, deterministic input seed, input shapes, load time, mean/median/p50/p90/p95/p99 latency, throughput, process-RSS peak, scoped NVIDIA/Jetson power where available, thermal readings where exposed and a reproducibility fingerprint.

Full local filesystem paths and raw hostnames are not written into the report.

### Native backends

- `OnnxRuntimeBackend`: in-process ONNX Runtime execution-provider benchmark.
- `TensorRTBackend`: native NVIDIA `trtexec` summary parsing for ONNX or TensorRT engine/plan artifacts.
- `OpenVINOBackend`: native Intel `benchmark_app` latency-oriented benchmark.
- `CoreMLBackend`: macOS Core ML benchmark for generic numeric `MLMultiArray` inputs.

Unsupported backends or input semantics fail explicitly. See [docs/benchmarking.md](docs/benchmarking.md).

## Power and memory scope

AutonomyFit does not equate all watts or memory numbers.

- Jetson power is labelled by the observed rail, such as VDD_IN.
- NVIDIA `power.draw` is labelled as GPU board power.
- These are not represented as whole-system wall power.
- Generic local peak memory is labelled `process RSS`, not accelerator memory.

## MLPerf

AutonomyFit includes strict parsing/import primitives for VALID MLPerf results. Standardized evidence is only accepted as exact proof when the benchmark workload is the correct one and the model revision plus artifact identity are explicitly supplied. AutonomyFit does not map a different MLPerf detector onto YOLO26 by task similarity alone.

## Model discovery

The discovery pipeline uses provider adapters and separates discovery from approval. New candidates can remain `DISCOVERED` or `NORMALIZED` without being promoted into high-confidence recommendations. Official upstream provenance is preferred and licences are retained rather than normalized away.

The scheduled registry refresh runs daily, validates deterministic output and signs only meaningful registry changes. Registry updates do not publish PyPI packages.

## Development

```bash
git clone https://github.com/sylvesterkaczmarek/autonomyfit.git
cd autonomyfit
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
make evidence-validate
make smoke
```

CI covers Python 3.10 through 3.13, registry/evidence validation, Ruff, tests, wheel/sdist build, `twine check`, installed-wheel smoke tests and signed-registry refresh behavior.

## Limitations

- Vendor benchmarks remain contextual when exact artifact/revision identity is absent.
- Synthetic local inputs benchmark execution performance, not task accuracy or end-to-end autonomy safety.
- Process RSS is not GPU/accelerator memory.
- Power depends on measurement scope, power mode, thermals, clocks and peripherals.
- Native backend availability depends on vendor tooling installed on the target machine.
- Core ML generic benchmarking does not replace Xcode/Instruments profiling; Apple also introduced Core AI for new custom neural-network deployment workflows.
- Standardized MLPerf ingestion is intentionally empty unless an actually applicable model/system/artifact tuple exists.

See [docs/reproducibility.md](docs/reproducibility.md).

## Cite

> Kaczmarek, S. (2026). *AutonomyFit*. GitHub.

## License

MIT. See [LICENSE](LICENSE).

© **Sylvester Kaczmarek** · [https://www.sylvesterkaczmarek.com](https://www.sylvesterkaczmarek.com)

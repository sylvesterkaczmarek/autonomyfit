# AutonomyFit

![AutonomyFit](assets/social/github-social-card-autonomyfit.png)

[![CI](https://github.com/sylvesterkaczmarek/autonomyfit/actions/workflows/ci.yml/badge.svg)](https://github.com/sylvesterkaczmarek/autonomyfit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

AutonomyFit detects local edge-AI hardware and ranks models against memory, latency,
throughput, accuracy, power and runtime constraints. The PyPI package is the decision engine.
Model intelligence lives in a separately versioned, signed registry that can discover and
publish new model metadata without a new package release.

## Install

```bash
pip install autonomyfit
```

For development:

```bash
git clone https://github.com/sylvesterkaczmarek/autonomyfit.git
cd autonomyfit
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Quick start

```bash
autonomyfit scan
autonomyfit recommend --task detection --fps 30 --latency-ms 40
```

A known Jetson target can be evaluated without having the device attached:

```bash
autonomyfit recommend \
  --hardware-profile jetson-orin-nx-16gb \
  --fps 200 \
  --latency-ms 5
```

The current exact YOLO26n / Jetson Orin NX / TensorRT FP16 evidence produces a
`VERIFIED_FIT` at 4.13 ms. Unknown performance remains unknown.

## Continuous model intelligence

AutonomyFit 0.3 separates three concerns:

```text
PyPI engine                  Discovery pipeline                Signed registry
------------------------     -----------------------------     -------------------------
hardware detection           provider adapters                 normalized model identity
constraint/ranking engine    metadata-only discovery           upstream revision/source
local benchmarking           deduplication/aliases             licence and compatibility
registry verification        lifecycle and source quality      evidence and freshness
CLI search/info         <--- approval boundary           ---> independently versioned data
```

The scheduled discovery pipeline runs every day and currently evaluates:

- Hugging Face Hub model APIs for supported task families
- Hugging Face model-card metadata and safetensors parameter metadata
- NVIDIA's official Hugging Face publisher feed, with NGC treated as a higher-trust
  vendor ecosystem for future direct enrichment
- Ultralytics' official GitHub release and model-configuration signals
- configured vendor GitHub release feeds

Discovery never imports or executes model repository code and never downloads model weights.
It uses machine-readable metadata endpoints only.

The pipeline distinguishes:

| State | Meaning |
|---|---|
| `DISCOVERED` | Seen upstream, but metadata is incomplete. |
| `NORMALIZED` | Canonical identity/revision is known, but approval evidence is incomplete. |
| `SOURCE_VERIFIED` | Trusted publisher, exact revision, licence and required metadata are present. |
| `COMPATIBILITY_VERIFIED` | Runtime/format support has additional compatibility evidence. |
| `BENCHMARKED` | Exact benchmark evidence exists for at least one supported tuple. |
| `DEPRECATED` | Previously observed discovery record is no longer seen upstream. |

Only records that reach `SOURCE_VERIFIED` can be automatically promoted into the signed
model registry. A newly discovered community model with missing licence, missing parameter
count, unknown publisher or incomplete runtime metadata remains in the discovery audit data
and is not silently promoted into recommendations.

See [docs/discovery.md](docs/discovery.md).

## Registry updates do not require PyPI updates

A normal recommendation checks the verified cache first and conditionally refreshes the
official registry when needed. Registry data is accepted only after:

- Sigstore verification against the exact registry-publishing workflow identity
- Registry Schema v2 validation
- signed generation/expiry checks
- monotonic registry-version checks
- rollback and same-version/content-replacement checks

Inspect or update the registry:

```bash
autonomyfit registry status
autonomyfit registry update
```

New model records can therefore appear after `autonomyfit registry update` without
`pip install --upgrade autonomyfit`.

A package upgrade is required only when the engine, CLI, supported schema behavior or trust
client changes.

## Search and inspect models

List registry models with discovery-oriented filters:

```bash
autonomyfit models
autonomyfit models --task vlm
autonomyfit models --source nvidia
autonomyfit models --new-since 2026-08-01
autonomyfit models --status source_verified
```

Search normalized identity and provenance:

```bash
autonomyfit search "smolvlm"
autonomyfit search "object detection" --task detection
```

Inspect one model:

```bash
autonomyfit info yolo26n
autonomyfit info yolo26n --json
```

The older `autonomyfit catalog` command remains as a backward-compatible registry listing
alias.

## Offline operation

```bash
autonomyfit recommend --offline --task detection
autonomyfit models --offline
```

Offline mode uses the last verified cache. If no verified cache exists, it uses the registry
snapshot bundled with the package. Stale data is labelled rather than silently presented as
current.

Clear downloaded registry data with:

```bash
autonomyfit registry clear-cache
```

The highest trusted registry version is retained to preserve rollback protection.

## Registry provenance

Human and JSON output expose registry source (`remote`, `cache`, `bundled-fallback` or
`custom`), registry version/freshness, upstream source URL, revision where known, model
verification state and licence metadata.

```bash
autonomyfit recommend --task detection --json
autonomyfit models --json
```

## Fit decisions

| Outcome | Meaning |
|---|---|
| `VERIFIED_FIT` | Exact available evidence satisfies the requested constraints. |
| `FEASIBLE` | Compatibility and memory screening pass with no unresolved performance constraint. |
| `BENCHMARK_REQUIRED` | Compatibility passes, but latency, FPS or power requires target evidence. |
| `CONSTRAINT_FAIL` | A measured or catalogued constraint is violated. |
| `NO_FIT` | A hard compatibility or memory screen fails. |

Measurements from one hardware/runtime/precision tuple are never transferred to another.

## Custom catalogues

Registry Schema v2 separates model identity, upstream provenance, modalities, parameters,
input information, runtime/precision compatibility, licence, evidence references,
verification state and freshness metadata.

Legacy schema-v1 custom catalogues remain supported:

```bash
autonomyfit recommend \
  --catalog examples/custom-models.json \
  --task detection
```

User-provided catalogues are not treated as official signed registry data.

## Local benchmark

Install optional ONNX benchmarking dependencies:

```bash
pip install 'autonomyfit[benchmark]'
```

Benchmark a local ONNX model:

```bash
autonomyfit benchmark model.onnx --iterations 100 --warmup 20 -o result.json
```

For a dynamic single input:

```bash
autonomyfit benchmark model.onnx --shape 1,3,640,640
```

The benchmark reports mean, p50, p95 and p99 latency, derived FPS, provider, input shape and
mean sampled power where a supported NVIDIA/Jetson power reader is available.

## Supply-chain and privacy model

The discovery job fetches metadata only. It does not:

- execute arbitrary Python from model repositories
- trust remote model cards as executable input
- download multi-gigabyte weights for discovery
- automatically promote records with missing licence or identity evidence
- overwrite the signed registry when generated data is invalid

Provider failures are recorded or fail closed. Deduplication prefers authoritative vendor
sources over mirrors. Registry publication still uses the Stage 1 Sigstore trust chain.

NVIDIA NGC was evaluated as a curated, signed model ecosystem. Its public documentation
exposes signed-model and repository/version APIs, but Stage 2 does not depend on an
undocumented broad NGC search endpoint. NVIDIA discovery therefore uses the official NVIDIA
publisher on Hugging Face until a stable public catalog-list API is appropriate.

## Scheduled refresh behavior

`.github/workflows/registry-refresh.yml` runs daily at 05:17 UTC.

It:

1. queries providers
2. normalizes and deduplicates records
3. applies source-quality and licence gates
4. validates deterministic Registry v2 output
5. runs discovery tests and lint
6. commits only meaningful model/discovery changes
7. renews registry freshness only when needed
8. invokes the existing Sigstore registry publisher when registry bytes changed

Ordinary discovery refreshes never publish to PyPI.

## Development validation

```bash
make lint
make test
make smoke
make registry-validate
pytest -q tests/test_discovery.py
```

CI tests Python 3.10 through 3.13, validates the registry, builds wheel and sdist
distributions, runs `twine check`, installs the built wheel and smoke-tests the installed CLI.

## Documentation

- [Registry architecture and trust](docs/registry.md)
- [Continuous discovery](docs/discovery.md)
- [Catalog and evidence model](docs/catalog.md)
- [Evidence semantics](docs/evidence.md)
- [Reproducibility](docs/reproducibility.md)

## Cite

> Kaczmarek, S. (2026). *AutonomyFit*. GitHub.

```bibtex
@software{Kaczmarek_2026_AutonomyFit,
  author = {Sylvester Kaczmarek},
  title  = {{AutonomyFit}},
  year   = {2026},
  url    = {https://github.com/sylvesterkaczmarek/autonomyfit}
}
```

## License

MIT. See [LICENSE](LICENSE).

© **Sylvester Kaczmarek** · [https://www.sylvesterkaczmarek.com](https://www.sylvesterkaczmarek.com)

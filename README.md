# AutonomyFit

![AutonomyFit](assets/social/github-social-card-autonomyfit.png)

[![CI](https://github.com/sylvesterkaczmarek/autonomyfit/actions/workflows/ci.yml/badge.svg)](https://github.com/sylvesterkaczmarek/autonomyfit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

AutonomyFit detects local edge-AI hardware and ranks models against memory, latency,
throughput, accuracy, power and runtime constraints. The Python package is the decision
engine. Model intelligence lives in a separately versioned, signed registry that can update
without publishing a new PyPI release.

## Install

```bash
pip install autonomyfit
```

Or clone for development:

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
autonomyfit recommend   --hardware-profile jetson-orin-nx-16gb   --fps 200   --latency-ms 5
```

The current exact YOLO26n / Jetson Orin NX / TensorRT FP16 evidence produces a
`VERIFIED_FIT` at 4.13 ms. Candidates without matching performance evidence remain
`BENCHMARK_REQUIRED` when a performance constraint is requested.

## Continuously updated registry

AutonomyFit 0.2 separates software releases from model-data releases:

```text
PyPI package                         Official model registry
------------------------------       ---------------------------------
hardware detection                   model identity and family
constraint and ranking engine        upstream provenance and revision
benchmarking                         licence and compatibility metadata
registry verification client   <---  accuracy/memory evidence
                                     freshness and verification state
```

A normal recommendation checks the cached registry first. When the cache becomes old,
AutonomyFit performs a conditional refresh. A changed official registry is accepted only
after Sigstore verification against the repository's registry publishing workflow identity,
schema validation, signed freshness checks and local rollback checks.

Inspect or refresh it explicitly:

```bash
autonomyfit registry status
autonomyfit registry update
```

Model additions therefore do **not** require `pip install --upgrade autonomyfit`. A PyPI
upgrade is needed only when the engine, schema support or client behavior changes.

### Offline operation

```bash
autonomyfit recommend --offline --task detection
```

Offline mode uses the last verified cache. If no verified cache exists, it uses the small
registry snapshot bundled with the package. Stale data is labelled explicitly rather than
silently presented as current.

Clear only downloaded registry data with:

```bash
autonomyfit registry clear-cache
```

The highest trusted registry version is deliberately retained to preserve rollback protection.

See [docs/registry.md](docs/registry.md) for the update and trust model.

## Registry provenance

Human output identifies the registry source (`remote`, `cache`, `bundled-fallback` or
`custom`) and warns when freshness is degraded. JSON output additionally exposes registry
version, generation/expiry timestamps, signature status, model source URL, upstream revision
when known, last-checked/last-verified timestamps and licence metadata.

```bash
autonomyfit recommend --task detection --json
```

## Fit decisions

| Outcome | Meaning |
|---|---|
| `VERIFIED_FIT` | Exact available evidence satisfies the requested constraints. |
| `FEASIBLE` | Compatibility and memory screening pass with no unverified performance constraint. |
| `BENCHMARK_REQUIRED` | Compatibility passes, but latency, FPS or power requires target-device evidence. |
| `CONSTRAINT_FAIL` | A measured or catalogued constraint is violated. |
| `NO_FIT` | A hard compatibility or memory screen fails. |

Unknown performance remains unknown. Measurements from one hardware/runtime/precision tuple
are never silently transferred to another.

## Model registry

List currently loaded models:

```bash
autonomyfit catalog
autonomyfit catalog --task vlm
```

Registry Schema v2 separates model identity, upstream provenance, modalities, parameters,
input information, runtime/precision compatibility, licence, evidence references,
verification state and freshness metadata. See [docs/catalog.md](docs/catalog.md).

Legacy schema-v1 custom catalogues remain supported:

```bash
autonomyfit recommend --catalog examples/custom-models.json --task detection
```

Custom catalogues are explicitly user-provided and are not treated as official signed registry
data.

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

## Security model

The official registry is signed keylessly in GitHub Actions with Sigstore. The client verifies
both the artifact signature and the expected GitHub workflow identity. It additionally rejects
lower registry versions, rejects changed content that reuses an already trusted version, and
checks signed `generated_at` / `expires_at` timestamps.

The cache is written atomically. Invalid remote data never replaces a previously valid cache.
Network or trust failures fall back to previously verified data or the bundled snapshot with a
visible warning.

Stage 1 uses Sigstore rather than a full TUF repository because automated GitHub OIDC signing
does not require maintaining long-lived private signing keys. If registry distribution later
moves to multiple mirrors or gains managed offline/online signing infrastructure, adopting a
full TUF role hierarchy remains a reasonable hardening step.

## Evidence and limitations

Current exact performance evidence remains intentionally narrow. The registry can now update
independently, but Stage 1 does not yet automatically discover every newly released model.
Registry population is still curated. Automated upstream discovery, normalization and
approval are the next stage.

Also:

- a registry fit is not proof that an end-to-end robotic workload meets its deadline
- published latency does not include every preprocessing, communication or control-loop cost
- parameter-derived memory values are screening estimates unless explicitly marked published
- synthetic ONNX execution does not validate task accuracy or safety
- power depends on clocks, thermal state, power mode, peripherals and workload
- model weights and runtimes retain their upstream licences and terms

See [docs/evidence.md](docs/evidence.md) and
[docs/reproducibility.md](docs/reproducibility.md).

## Development validation

```bash
make test
make smoke
python scripts/validate_registry.py registry/source/registry-v2.json
```

CI tests Python 3.10 through 3.13, builds wheel and sdist distributions, checks package
metadata, installs the built wheel and smoke-tests the installed CLI.

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

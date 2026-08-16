# AutonomyFit

![AutonomyFit](assets/social/github-social-card-autonomyfit.png)

[![CI](https://github.com/sylvesterkaczmarek/autonomyfit/actions/workflows/ci.yml/badge.svg)](https://github.com/sylvesterkaczmarek/autonomyfit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

AutonomyFit is an evidence-aware model-selection and benchmarking CLI for edge AI and autonomous systems. It detects the target hardware, filters models by deployment constraints, builds conservative Pareto frontiers, ranks for an explicit engineering objective, and tells you what is measured, estimated, or still unknown.

## Install

```bash
pip install autonomyfit
```

## Start here

```bash
autonomyfit scan
autonomyfit tasks
autonomyfit models --offline
autonomyfit recommend --task detection --hardware-profile jetson-orin-nx-16gb --objective balanced
autonomyfit compare yolo26n rfdetr-nano --hardware-profile nvidia-t4-16gb --objective accuracy
```

AutonomyFit 0.5 supports ten extensible task categories:

```text
detection  classification  segmentation  pose  depth
ocr        vlm             anomaly       asr   embedding
```

The signed continuous registry contains a curated set of practical model families rather than hundreds of weakly sourced entries. Stage 4 adds multi-task YOLO26, RF-DETR, RepViT, MobileSAM, Depth Anything V2, PP-OCRv6, SmolVLM2, EfficientAD, Whisper, MobileCLIP and DINOv2 families alongside the existing evidence model.

## Objective-aware ranking

Hard constraints are applied before ranking. Candidates that remain feasible are assigned conservative Pareto layers, then ordered for one explicit objective:

```bash
autonomyfit recommend --task classification --objective latency
autonomyfit recommend --task classification --objective throughput
autonomyfit recommend --task classification --objective accuracy
autonomyfit recommend --task detection --objective power
autonomyfit recommend --task segmentation --objective memory
autonomyfit recommend --task depth --objective balanced
```

`balanced` is not a hidden hand-tuned score. It normalizes the known latency, throughput, task metric, power and memory quantities within the feasible candidate set and penalizes missing objective coverage. Pareto dominance is only asserted between candidates with the same known objective set, so missing evidence cannot create a dominance claim.

See [docs/ranking.md](docs/ranking.md).

## Recommendation confidence

Each recommendation carries a numeric `0-100` confidence score plus a compatibility label. The score is the equal-weight mean of six disclosed components:

- exactness of the hardware match
- runtime/precision match
- evidence quality
- evidence freshness
- model revision/artifact identity
- requested-quantity coverage

An unresolved requested constraint caps confidence at 55. JSON output exposes every component, so the number can be audited instead of treated as a black-box label.

## Real engineering filters

```bash
autonomyfit recommend \
  --task segmentation \
  --hardware-profile apple-m4-pro-24gb \
  --objective memory \
  --max-params-m 30 \
  --license Apache-2.0 \
  --min-confidence 40 \
  --top 5
```

Other useful controls include `--runtime`, `--precision`, `--family`, `--max-memory-gb`, `--verified-only`, and `--include-experimental`.

## Hardware and runtime coverage

First-class profiles now cover practical NVIDIA Jetson and discrete GPUs, Apple Silicon, Intel CPU/GPU/NPU systems, AMD Ryzen AI systems, Qualcomm Snapdragon X Elite and Arm CPU targets such as Raspberry Pi 5.

AutonomyFit distinguishes native runtimes from ONNX Runtime execution-provider capability. TensorRT, OpenVINO and Core ML are native benchmark paths. QNN, XNNPACK, OpenVINO EP, CoreML EP, TensorRT EP, CUDA EP and Vitis AI EP can be detected as provider capabilities, but provider presence is never represented as proof that a specific graph is fully supported.

See [docs/hardware.md](docs/hardware.md).

## Evidence-aware decisions

| Outcome | Meaning |
|---|---|
| `VERIFIED_FIT` | Exact identity-matched local or standardized evidence satisfies the requested constraints. |
| `FEASIBLE` | Hard compatibility and memory screening pass with no unresolved requested performance constraint. |
| `BENCHMARK_REQUIRED` | A requested quantity cannot be defended with exact applicable evidence. |
| `CONSTRAINT_FAIL` | Exact applicable evidence shows a requested threshold is violated. |
| `NO_FIT` | A hard model/runtime/precision/memory/size feasibility condition fails. |

Evidence quality remains explicit: `local-measured`, `standardized`, `vendor-published`, `third-party-reproducible`, or `metadata-estimate`. Vendor reference tables remain useful context but do not silently become proof for a different artifact.

See [docs/evidence.md](docs/evidence.md).

## Compare models

```bash
autonomyfit compare \
  yolo26n rfdetr-nano rfdetr-small \
  --hardware-profile nvidia-t4-16gb \
  --objective balanced \
  --json
```

Comparison uses the same feasibility, Pareto and confidence engine as `recommend`. JSON includes objective rank, Pareto layer, dominance relationships, hard-constraint results, measured data, estimates, unknown quantities and the benchmark that would most reduce uncertainty.

## Local benchmarking

```bash
pip install 'autonomyfit[benchmark]'
autonomyfit benchmark-backends
autonomyfit benchmark model.onnx --model-id my-model --model-revision COMMIT -o result.json
autonomyfit benchmark-inspect result.json
autonomyfit benchmark-import result.json
```

ONNX Runtime can target an installed provider explicitly, for example:

```bash
autonomyfit benchmark model.onnx --backend onnxruntime --provider QNNExecutionProvider --model-id my-model
autonomyfit benchmark model.onnx --backend onnxruntime --provider XNNPACKExecutionProvider --model-id my-model
```

A provider being installed does not guarantee full operator coverage. The benchmark is the check.

See [docs/benchmarking.md](docs/benchmarking.md).

## Registry design and package size

The PyPI package remains the decision engine. Model coverage lives in a separately versioned signed registry, with a compact bundled fallback for offline use. Normal CLI use respects the existing verified cache/refresh policy rather than downloading the registry on every invocation.

```bash
autonomyfit registry status
autonomyfit registry update
autonomyfit search mobileclip
autonomyfit info ppocrv6-tiny
```

## Limitations

- A hardware/runtime capability does not establish model operator coverage.
- Most newly added families currently have source-verified metadata rather than exact target-machine benchmarks.
- Cross-family accuracy ranking is only meaningful when the task metric is comparable; entries without a comparable metric remain unknown for that objective.
- Memory screens derived from parameters are estimates and are labelled as such.
- Generic process RSS is not accelerator memory.
- Power scope remains platform-specific unless measured by an applicable standardized method.
- Robotic/VLA policies are intentionally not yet a canonical task; the task registry is structured so they can be added without redesigning the engine.

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

CI covers Python 3.10 through 3.13, registry/evidence validation, Ruff, the full test suite, wheel/sdist build, `twine check`, installed-wheel smoke tests and signed-registry behavior.

## Cite

> Kaczmarek, S. (2026). *AutonomyFit*. GitHub.

## License

MIT. See [LICENSE](LICENSE).

© **Sylvester Kaczmarek** · [https://www.sylvesterkaczmarek.com](https://www.sylvesterkaczmarek.com)

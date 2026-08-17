# Benchmarking

AutonomyFit benchmarks exact artifacts on detected hardware and records the execution context as evidence. A bundled hardware profile is a screening target only and cannot create `local-measured` evidence.

## Evidence matrix identity

`autonomyfit benchmark-matrix` exposes the dimensions that make a benchmark applicable:

- model ID and immutable model revision
- artifact SHA-256
- exact detected machine identity
- runtime and runtime version
- provider/device and provider version
- precision and quantisation
- batch size and input shapes
- power mode when relevant
- material software-stack fingerprint

Local measured evidence is eligible for `VERIFIED_FIT` only when this execution context is complete and the caller requests a matching context. A benchmark from another machine, batch, shape, provider or material software stack remains contextual rather than silently overriding other evidence.

```bash
autonomyfit benchmark-matrix --local-only
autonomyfit benchmark-matrix --model-id MODEL --json
```

## ONNX Runtime

`OnnxRuntimeBackend` runs deterministic synthetic inputs in-process, separates session load from timed inference, records min/mean/median/p50/p90/p95/p99/max/stdev latency, throughput, process RSS and scoped NVIDIA/Jetson power where available. Accelerator-provider requests disable CPU execution-provider fallback and the report records requested/active providers, provider options and ONNX Runtime build information. Synthetic inputs measure execution, not task accuracy.

## TensorRT

`TensorRTBackend` uses `trtexec`. ONNX inputs are inspected to record concrete input names/shapes before the command runs. Dynamic non-batch shapes must be supplied explicitly. Serialized engine/plan artifacts require explicit trust. TensorRT engine evidence is tied to the recorded GPU/software stack and is not assumed portable. `trtexec --warmUp` is a millisecond setting and is labelled as such in backend options.

## OpenVINO

`OpenVINOBackend` uses `benchmark_app`, defaults to an explicit `CPU` device rather than `AUTO`, records exposed OpenVINO devices where available, and fails if an explicitly requested detected device is unavailable. Native benchmark-app warmup is owned by the tool; AutonomyFit does not misreport the generic CLI warmup count as executed OpenVINO warmup iterations.

## Core ML

`CoreMLBackend` supports numeric `MLMultiArray` inputs on macOS. Compute units are explicit (`ALL`, `CPU_ONLY`, `CPU_AND_GPU`, or `CPU_AND_NE`) and recorded in the evidence. Image-typed or otherwise unsupported generic inputs fail explicitly. Detailed Neural Engine/GPU attribution still requires Apple profiling tools.

## Reproducible local run

```bash
autonomyfit benchmark model.onnx \
  --model-id my-model \
  --model-revision COMMIT \
  --backend onnxruntime \
  --provider CPUExecutionProvider \
  --precision fp32 \
  --batch-size 1 \
  -o result.json

autonomyfit benchmark-inspect result.json
autonomyfit benchmark-import result.json
autonomyfit benchmark-matrix --local-only
```

Reports include an artifact SHA-256, exact machine ID, runtime/provider versions, input shapes, batch, deterministic seed, latency distribution, throughput, process RSS, scoped power/energy when available, environment fingerprint and material software-stack fingerprint. The artifact digest is checked before and after native execution.

## Validation status

The regular CI executes a real ONNX Runtime `CPUExecutionProvider` benchmark against a generated ONNX graph and validates/imports its report. That is native runtime validation on the GitHub-hosted CI machine, not evidence for Jetson, NVIDIA GPU, Apple Silicon or Intel GPU/NPU hardware. Platform-specific support must be described as physically validated only after a report was measured on that detected target.


## Self-hosted target validation

`.github/workflows/hardware-validation.yml` is the permanent manual harness for Jetson, discrete NVIDIA, Intel and physical Apple target runs. Attach the target as a self-hosted GitHub runner, give it a dedicated label such as `autonomyfit-hardware`, preinstall the vendor runtime, and dispatch the workflow with an exact model path, immutable model revision, runtime/provider/device, precision, batch and shape context.

The workflow captures `scan`, backend availability, the schema-v2 benchmark report, an inspected summary and a separate machine-class attestation. `operator_machine_class=physical` is an operator attestation and is recorded as such; AutonomyFit does not independently infer that a self-hosted runner is bare-metal. VM/container runs remain useful native execution validation but must not be described as physical target evidence.

Multi-input ONNX graphs use repeatable `--input-shape NAME=...` values (semicolon-separated in the workflow input). Named shapes are checked against graph input names, ranks and fixed dimensions before execution.

# Deployment validation

Deployment assessment records identity, compatibility, conversion, measurement and recommendation separately.

## Flow

1. Resolve a model from the signed registry.
2. Resolve or provide an artifact with immutable revision and content identity where possible.
3. Apply the artifact trust policy before loading anything into a runtime.
4. Check the model/runtime relationship and local runtime availability.
5. Run format-specific structural checks.
6. Optionally convert using an installed vendor/runtime toolchain.
7. Optionally run a generic numerical post-conversion check where the contract permits it.
8. Benchmark the exact artifact on the detected machine.
9. Import the benchmark as exact local evidence if requested.
10. Re-run the recommendation engine using the current measurement, whether or not it was imported.
11. Compare the local measurement with applicable registry evidence.
12. Emit a schema-validated JSON or Markdown deployment report.

## Artifact discovery

`autonomyfit artifacts MODEL` inspects supported upstream artifact metadata without executing model repository code.

For Hugging Face sources, a requested branch/tag/ref is resolved to a full immutable commit SHA before download. Artifact candidates are classified as static or execution-sensitive. The managed cache stores the computed SHA-256 and verifies it every time the cache entry is reopened. If upstream LFS SHA-256 metadata is exposed, it must match the downloaded bytes.

Offline acquisition never falls back to network access. Hub acquisition requires exactly one verified cache record matching the requested model/revision/filename. Direct URL acquisition is refused with `--offline`; supply a local artifact instead.

## Trust boundary

Automatic handling is deliberately narrow.

Safe static candidates include ONNX, safetensors and metadata files. Repository Python, pickle-style PyTorch serialization, native libraries and serialized TensorRT engines are not automatically trusted.

A local `.pt`/`.pth` artifact can be identified and hashed without loading it. Conversion requires `--trust-artifact`, which is an explicit statement that the user controls or otherwise trusts that serialization.

A serialized TensorRT engine is treated as executable state. AutonomyFit refuses to deserialize an engine that was neither built locally in the current validation workflow nor explicitly trusted by the user.

## Artifact identity

Single-file artifacts use the ordinary byte SHA-256.

ONNX graphs with external tensors include every referenced companion file in their bundle identity, including references inside nested graphs and constants. Use the complete bundle digest with `--sha256`. The ONNX parser is required even for identity checks; install `autonomyfit[deployment]` or `autonomyfit[benchmark]`. Missing files, traversal paths and symbolic links are rejected before the runtime opens external data. Automatic ONNX acquisition currently accepts self-contained graphs; provide complete external-data bundles locally.

OpenVINO IR is a multi-file identity when a sibling `.bin` exists. Core ML `.mlpackage` artifacts are directory identities. These use a deterministic manifest digest over relative member names and each member's byte SHA-256. Symbolic links are rejected so a bundle cannot silently pull bytes from outside its identity boundary. Modifying any member changes the deployment artifact identity.

Before conversion or benchmarking, the admitted digest is rechecked. Conversion and benchmarking also compare the source digest before and after execution and fail rather than recording evidence if the input identity changes.

## Conversion

The generic conversion matrix is intentionally small:

| Source | Target | Tool | Trust requirement |
|---|---|---|---|
| ONNX | TensorRT | `trtexec` | static ONNX input |
| ONNX | OpenVINO IR | `ovc` or `openvino.convert_model` | static ONNX input |
| trusted TorchScript | ONNX | `torch.onnx.export` | explicit `--trust-artifact` + shape |
| trusted TorchScript | Core ML ML Program | `coremltools` | explicit `--trust-artifact` + shape |

A conversion record contains source and target identities, tool/version, command, duration, companion artifacts, warnings and equivalence status.

OpenVINO explicitly enables FP16 weight compression only for `fp16`; `fp32` disables it in both supported tool paths. TensorRT FP32 conversion disables TF32, while automatic INT8 conversion is refused because the generic path cannot establish calibration scales. Use a separately built, trusted engine for that workflow. TorchScript export to ONNX accepts `fp32` or `artifact` and performs no precision conversion; Core ML conversion accepts `fp16`, `fp32` or `artifact`. Unsupported settings fail before deserialisation or conversion.

Conversion success means the target tool produced an artifact. It does not mean application accuracy is unchanged.

## Generic correctness check

For ONNX -> OpenVINO, AutonomyFit can compare deterministic synthetic numeric outputs when all of these conditions hold:

- compatible numeric tensor input contract
- resolvable shapes
- preserved input mapping
- comparable numeric output shapes
- ONNX Runtime and OpenVINO are installed

Tolerances must be finite and non-negative. NaN or infinite outputs fail, including matching NaNs. The reported tolerance describes only the sampled numeric comparison; it does not establish task-level accuracy or dataset performance.

## Benchmarking and local evidence

`--benchmark` always measures the current machine. A profile can be used for screening, but AutonomyFit refuses to write local benchmark evidence for a profile that does not match the detected target.

Successful reports preserve latency distribution, throughput, process RSS, power/energy where available, hardware identity, runtime/provider versions, precision, input shapes, warmup/iteration counts, deterministic seed and reproduction command.

`--no-import-local` prevents storage for future runs; the current benchmark still drives this assessment. If a requested limit lacks exact applicable evidence, status remains `benchmark-required`, even after successful execution. A metadata memory screen and a declared precision label do not become measured accelerator-memory or per-operator precision guarantees.

Exact local evidence can override generic registry evidence only when model revision, artifact digest, hardware, runtime/provider and precision match. Local evidence is ignored after its freshness window or after material stack identity changes.

## Registry comparison

When comparable non-local registry evidence exists, the report contains:

- registry evidence ID and quality
- expected representative latency
- expected p50/median-to-p95/p99/max range when available
- local latency
- local/expected ratio
- classification as materially slower, materially faster, within 20%, or not comparable
- batch/input shape mismatches
- power-mode mismatch
- software-stack mismatch
- explicit thermal comparability warning

The 20% band is an engineering flag, not a statistical significance claim.

## Reports

```bash
autonomyfit validate MODEL --artifact model.onnx --runtime onnx --benchmark --report report.json
autonomyfit report report.json -o report.md
```

The JSON report is validated against `deployment-report-v1.schema.json`, including rejection of non-finite numbers. It records identity, compatibility and measurement scope, including reproduction commands.

## Candidate loop

```bash
autonomyfit recommend --task detection --top 3

autonomyfit assess MODEL1 MODEL2 \
  --artifact MODEL1=./model1.onnx \
  --artifact MODEL2=./model2.onnx \
  --runtime onnx \
  --json
```

`assess` benchmarks each exact artifact, imports valid local evidence and then asks the normal ranking engine to reorder the selected models. It does not create a separate hidden ranking system.

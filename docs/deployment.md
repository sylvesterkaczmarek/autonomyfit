# Deployment validation

AutonomyFit 0.6 adds an end-to-end deployment-assessment layer. The system is intentionally split into identity, compatibility, conversion, measurement and recommendation stages so a success in one stage cannot be misreported as proof for another.

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
10. Re-run the recommendation engine using that local evidence.
11. Compare the local measurement with applicable registry evidence.
12. Emit a schema-validated JSON or Markdown deployment report.

## Artifact discovery

`autonomyfit artifacts MODEL` inspects supported upstream artifact metadata without executing model repository code.

For Hugging Face sources, a requested branch/tag/ref is resolved to a full immutable commit SHA before download. Artifact candidates are classified as static or execution-sensitive. The managed cache stores the computed SHA-256 and verifies it every time the cache entry is reopened. If upstream LFS SHA-256 metadata is exposed, it must match the downloaded bytes.

Offline acquisition never falls back to network access. It requires exactly one verified cache record matching the requested model/revision/filename.

## Trust boundary

Automatic handling is deliberately narrow.

Safe static candidates include ONNX, safetensors and metadata files. Repository Python, pickle-style PyTorch serialization, native libraries and serialized TensorRT engines are not automatically trusted.

A local `.pt`/`.pth` artifact can be identified and hashed without loading it. Conversion requires `--trust-artifact`, which is an explicit statement that the user controls or otherwise trusts that serialization.

A serialized TensorRT engine is treated as executable state. AutonomyFit refuses to deserialize an engine that was neither built locally in the current validation workflow nor explicitly trusted by the user.

## Artifact identity

Single-file artifacts use the ordinary byte SHA-256.

OpenVINO IR is a multi-file identity when a sibling `.bin` exists. Core ML `.mlpackage` artifacts are directory identities. These use a deterministic manifest digest over relative member names and each member's byte SHA-256. Modifying any member changes the deployment artifact identity.

## Conversion

The generic conversion matrix is intentionally small:

| Source | Target | Tool | Trust requirement |
|---|---|---|---|
| ONNX | TensorRT | `trtexec` | static ONNX input |
| ONNX | OpenVINO IR | `ovc` or `openvino.convert_model` | static ONNX input |
| trusted TorchScript | ONNX | `torch.onnx.export` | explicit `--trust-artifact` + shape |
| trusted TorchScript | Core ML ML Program | `coremltools` | explicit `--trust-artifact` + shape |

A conversion record contains source and target identities, tool/version, command, duration, companion artifacts, warnings and equivalence status.

Conversion success means the target tool produced an artifact. It does not mean application accuracy is unchanged.

## Generic correctness check

For ONNX -> OpenVINO, AutonomyFit can compare deterministic synthetic numeric outputs when all of these conditions hold:

- compatible numeric tensor input contract
- resolvable shapes
- preserved input mapping
- comparable numeric output shapes
- ONNX Runtime and OpenVINO are installed

The current generic tolerance is reported with the result. A pass means the sampled outputs matched within that tolerance. It is not task-level accuracy validation, dataset evaluation or statistical certification.

## Benchmarking and local evidence

`--benchmark` always measures the current machine. A profile can be used for screening, but AutonomyFit refuses to write local benchmark evidence for a profile that does not match the detected target.

Successful reports preserve latency distribution, throughput, process RSS, power/energy where available, hardware identity, runtime/provider versions, precision, input shapes, warmup/iteration counts, deterministic seed and reproduction command.

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

The JSON report is validated against `deployment-report-v1.schema.json`. It records all information needed to distinguish identity, compatibility and measurement scope, including reproduction commands.

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

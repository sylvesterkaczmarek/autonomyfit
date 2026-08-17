# Benchmarking

AutonomyFit 0.4 provides backend abstractions for ONNX Runtime, TensorRT, OpenVINO and Core ML.

## ONNX Runtime

`OnnxRuntimeBackend` runs deterministic synthetic inputs in-process, separates model/session load time from timed inference, records mean/median/p50/p90/p95/p99/min/max/stdev latency, derived throughput, process RSS, runtime/provider version and supported NVIDIA/Jetson power samples.

Install the ONNX benchmark extra:

```bash
pip install 'autonomyfit[benchmark]'
```

## TensorRT

`TensorRTBackend` uses NVIDIA `trtexec`. It supports ONNX input and serialized TensorRT engine/plan artifacts, records the native throughput and latency percentile summary, and preserves the actual command and TensorRT version. TensorRT engines remain hardware/software specific evidence.

## OpenVINO

`OpenVINOBackend` uses Intel `benchmark_app` with a latency performance hint and explicit iteration count. It parses native latency/throughput output. Install OpenVINO using Intel's supported installation route so `benchmark_app` is available.

## Core ML

`CoreMLBackend` benchmarks numeric `MLMultiArray` inputs through `MLModel.predict` on macOS. Image-typed or otherwise unsupported generic inputs fail explicitly instead of generating synthetic semantics that may be wrong. Detailed Apple compute-unit profiling should still be done with Xcode/Instruments; Apple is also moving new custom neural-network deployment toward Core AI.

## Local report

```bash
autonomyfit benchmark model.onnx --model-id my-model --model-revision COMMIT -o result.json
autonomyfit benchmark-inspect result.json
autonomyfit benchmark-import result.json
```

A report includes an artifact SHA-256, model revision, hardware identity, runtime/provider versions, deterministic seed, input shapes, load time, latency distribution, throughput, process-RSS peak, scoped power where available and a reproducibility fingerprint. Full local filesystem paths are not embedded in the report.

Imported reports are schema validated, benchmark IDs are constrained to cache-safe filenames, and writes are atomic. Corrupt, path-escaping or malformed reports are not added to the local evidence store.

The artifact digest is established before native benchmarking and checked again after execution. If the artifact changes during the run, AutonomyFit discards the benchmark instead of attaching the measurements to the wrong bytes.

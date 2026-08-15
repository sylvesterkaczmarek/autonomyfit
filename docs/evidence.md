# Evidence rules

AutonomyFit distinguishes three kinds of information.

## Published measurements

A benchmark record is bundled only when a source identifies the model, hardware, runtime, precision and latency sufficiently for an exact tuple match.

Current sources include:

- Ultralytics YOLO26 model metadata and T4 latency table: https://docs.ultralytics.com/models/yolo26
- Ultralytics NVIDIA Jetson benchmark guide: https://docs.ultralytics.com/guides/nvidia-jetson
- Hugging Face SmolVLM 256M model card: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- Hugging Face SmolVLM2 2.2B model card: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct

The first release uses Jetson latency records for YOLO26n where the upstream guide exposes an explicit runtime and precision. It does not infer missing model/hardware combinations.

## Screening estimates

For models without an upstream memory figure, AutonomyFit estimates a memory screen from parameter count and precision, then adds a task-specific allowance for runtime workspace and activations.

The estimate is intentionally used only for first-pass fit screening. Actual peak memory can differ because of model graph structure, input dimensions, batching, context length, runtime allocator behavior and concurrent processes.

## Local measurements

`autonomyfit benchmark` measures execution latency on a supplied ONNX graph using an ONNX Runtime provider available on the local machine. It uses deterministic synthetic inputs and records mean, p50, p95 and p99 latency.

On supported NVIDIA systems it also attempts power sampling. Absence of a usable power reader produces a missing power value rather than a fabricated estimate.

## Constraint policy

Latency, FPS and power constraints require matching measurements. If evidence is absent, the outcome is `BENCHMARK_REQUIRED`.

Accuracy constraints use the metric stored in the model profile. The caller is responsible for comparing like-for-like metrics and datasets when using a custom catalog.

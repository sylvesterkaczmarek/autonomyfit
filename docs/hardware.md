# Hardware and runtime model

AutonomyFit separates four levels that should not be conflated:

1. **profile support**: a bundled target description exists
2. **runtime detection**: the installed machine exposes a runtime/provider/device
3. **native benchmark execution**: an exact artifact successfully ran through that backend
4. **physical target evidence**: a benchmark report was measured on the named detected hardware

## Target categories

Bundled profiles cover NVIDIA Jetson, discrete NVIDIA GPUs, Apple Silicon, Intel CPU/GPU/NPU systems, AMD Ryzen AI, Qualcomm Snapdragon X Elite and Arm CPU targets. Profile capability is not proof that a specific model graph runs.

ONNX Runtime bridge providers such as CUDA, TensorRT, OpenVINO, Core ML, QNN, XNNPACK and Vitis AI remain `verified=false` for model coverage until the exact graph is exercised.

## Detected identity

Local evidence uses a host-specific hashed machine identity plus detected CPU/GPU/memory topology rather than collapsing a real machine to a coarse bundled profile ID. Driver, JetPack/L4T, power mode, runtime/provider versions and software-stack signals are tracked separately so material stack changes can invalidate exact evidence without pretending the hardware itself changed.

Jetson detection records JetPack package metadata where available, L4T information, `nvpmodel` power mode and VDD_IN telemetry when exposed. NVIDIA discrete GPUs use `nvidia-smi` identity/driver and board-power telemetry. Intel detection records OpenVINO CPU/GPU/NPU devices exposed by the installed runtime. Apple detection records the Apple chip and explicit Core ML compute-unit choice in benchmark evidence.

## Physical validation

A hardware profile must never create a local measurement. `validate --benchmark --hardware-profile PROFILE` is permitted only when the detected machine matches that profile, and the resulting report uses detected hardware identity.

Use:

```bash
autonomyfit scan --json
autonomyfit benchmark-backends --json
autonomyfit benchmark ARTIFACT --model-id MODEL --model-revision REVISION -o benchmark.json
autonomyfit benchmark-inspect benchmark.json
autonomyfit benchmark-matrix --local-only
```

CI/native-backend success is not automatically called Jetson/NVIDIA/Apple/Intel physical validation. Those claims require evidence generated on the corresponding detected target.

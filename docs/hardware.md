# Hardware and runtime model

AutonomyFit models hardware capability separately from model compatibility.

## First-class target categories

The bundled profile set covers:

- NVIDIA Jetson AGX Thor, AGX Orin, Orin NX and Orin Nano Super
- NVIDIA T4 and current GeForce RTX examples
- Apple M4, M4 Pro and M4 Max unified-memory systems
- Intel Core Ultra systems with CPU/GPU/NPU and OpenVINO
- AMD Ryzen AI systems with ONNX Runtime/Vitis AI paths where exposed
- Qualcomm Snapdragon X Elite with ONNX Runtime QNN
- Arm CPU systems, including Raspberry Pi 5 and XNNPACK-capable ONNX Runtime paths

Profiles record accelerator type, memory topology, precision capability, runtime capability, software-stack notes and relevant limits when a stable public limit exists.

## Capability is not compatibility

For native runtimes such as TensorRT, OpenVINO and Core ML, a hardware profile can state that the runtime is a normal target for the platform. Model-level compatibility still comes from the registry and evidence.

For ONNX Runtime execution providers, AutonomyFit is stricter. QNN, XNNPACK, OpenVINO EP, CoreML EP, TensorRT EP, CUDA EP and Vitis AI EP are reported as provider capabilities with `verified=false` for model coverage. Operator support, graph partitioning and fallback are model-dependent.

A provider should therefore be benchmarked on the exact artifact:

```bash
autonomyfit benchmark model.onnx \
  --backend onnxruntime \
  --provider QNNExecutionProvider \
  --model-id my-model \
  --model-revision COMMIT \
  -o result.json
```

The resulting artifact hash and runtime/provider information can then contribute exact evidence.

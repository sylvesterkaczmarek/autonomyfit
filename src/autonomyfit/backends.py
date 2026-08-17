from __future__ import annotations

import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .benchmark import (
    MemorySampler,
    PowerSampler,
    _shape_for_input,
    latency_summary,
    make_benchmark_report,
    power_reader,
)
from .integrity import artifact_sha256
from .models import HardwareProfile


class BackendError(RuntimeError):
    """Benchmark backend failed or is unavailable."""


@dataclass(frozen=True)
class BenchmarkRequest:
    model_path: Path
    model_id: str
    model_revision: str | None
    hardware: HardwareProfile
    iterations: int = 50
    warmup: int = 10
    shape_override: list[int] | None = None
    input_shapes: dict[str, list[int]] = field(default_factory=dict)
    provider: str | None = None
    device: str | None = None
    precision: str = "artifact"
    quantization: str | None = None
    batch_size: int | None = None
    command: str | None = None
    trusted_artifact: bool = False
    expected_sha256: str | None = None
    compute_units: str | None = None


@dataclass(frozen=True)
class BackendAvailability:
    name: str
    available: bool
    version: str | None = None
    executable: str | None = None
    detail: str | None = None


class BenchmarkBackend(ABC):
    name: str

    @abstractmethod
    def availability(self) -> BackendAvailability: ...

    @abstractmethod
    def benchmark(self, request: BenchmarkRequest) -> dict[str, Any]: ...


def _numpy_dtype(ort_type: str):
    import numpy as np

    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(bool)": np.bool_,
    }
    if ort_type not in mapping:
        raise ValueError(f"Unsupported ONNX input type: {ort_type}")
    return mapping[ort_type]



def _infer_batch_size(
    explicit: int | None, input_shapes: dict[str, list[int]]
) -> int | None:
    if explicit is not None:
        return explicit
    first_dims = {dims[0] for dims in input_shapes.values() if dims}
    return next(iter(first_dims)) if len(first_dims) == 1 else None


def _resolve_onnx_input_shapes(request: BenchmarkRequest) -> dict[str, list[int]]:
    if request.input_shapes:
        return dict(request.input_shapes)
    try:
        import onnx
    except ImportError as exc:
        raise BackendError(
            "native ONNX shape discovery requires the benchmark extra with onnx installed"
        ) from exc
    try:
        model = onnx.load(str(request.model_path), load_external_data=False)
    except Exception as exc:
        raise BackendError(f"could not inspect ONNX inputs: {exc}") from exc
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    if request.shape_override is not None:
        if len(inputs) != 1:
            raise BackendError(
                "--shape is supported only for single-input ONNX models; provide named input shapes for multi-input graphs"
            )
        return {inputs[0].name: list(request.shape_override)}
    resolved: dict[str, list[int]] = {}
    for item in inputs:
        dims: list[int] = []
        tensor_type = item.type.tensor_type
        for index, dim in enumerate(tensor_type.shape.dim):
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            elif index == 0:
                dims.append(request.batch_size or 1)
            else:
                raise BackendError(
                    f"ONNX input {item.name!r} has a dynamic non-batch dimension; use --shape for a single-input graph"
                )
        if not dims:
            raise BackendError(f"ONNX input {item.name!r} has no resolvable shape")
        resolved[item.name] = dims
    if not resolved:
        raise BackendError("ONNX graph exposes no benchmarkable inputs")
    return resolved


class OnnxRuntimeBackend(BenchmarkBackend):
    name = "onnxruntime"

    def availability(self) -> BackendAvailability:
        try:
            import onnxruntime as ort
        except ImportError:
            return BackendAvailability(self.name, False, detail="onnxruntime is not installed")
        providers = ", ".join(ort.get_available_providers())
        return BackendAvailability(self.name, True, ort.__version__, detail=providers)

    def benchmark(self, request: BenchmarkRequest) -> dict[str, Any]:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendError(
                "ONNX Runtime benchmarking requires: pip install 'autonomyfit[benchmark]'"
            ) from exc
        available = ort.get_available_providers()
        if not available:
            raise BackendError("ONNX Runtime reported no execution providers")
        provider = request.provider
        if provider is None:
            preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
            provider = next((name for name in preferred if name in available), available[0])
        if provider not in available:
            raise BackendError(f"Provider {provider!r} unavailable. Available: {', '.join(available)}")

        memory = MemorySampler()
        memory.start()
        started = time.perf_counter_ns()
        try:
            session_options = ort.SessionOptions()
            if provider != "CPUExecutionProvider":
                try:
                    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                except Exception as exc:
                    raise BackendError(
                        "could not disable ONNX Runtime CPU fallback for an accelerator provider"
                    ) from exc
            session = ort.InferenceSession(
                str(request.model_path), sess_options=session_options, providers=[provider]
            )
            active_providers = session.get_providers()
            if not active_providers or active_providers[0] != provider:
                raise BackendError(
                    f"requested provider {provider!r} was not the active primary provider: {active_providers}"
                )
            load_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            inputs = session.get_inputs()
            if request.shape_override is not None and len(inputs) != 1:
                raise BackendError("--shape is supported only for single-input ONNX models")
            feeds: dict[str, object] = {}
            input_shapes: dict[str, list[int]] = {}
            rng = np.random.default_rng(0)
            for input_meta in inputs:
                resolved = _shape_for_input(list(input_meta.shape), request.shape_override)
                dtype = _numpy_dtype(input_meta.type)
                if np.issubdtype(dtype, np.floating):
                    value = rng.random(resolved, dtype=np.float32).astype(dtype)
                elif dtype == np.bool_:
                    value = np.zeros(resolved, dtype=dtype)
                else:
                    value = np.zeros(resolved, dtype=dtype)
                feeds[input_meta.name] = value
                input_shapes[input_meta.name] = resolved

            for _ in range(request.warmup):
                session.run(None, feeds)

            reader, scope = power_reader(request.hardware.platform)
            sampler = PowerSampler(reader) if reader else None
            if sampler:
                sampler.start()
            latencies: list[float] = []
            try:
                for _ in range(request.iterations):
                    start = time.perf_counter_ns()
                    session.run(None, feeds)
                    latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
            finally:
                power = sampler.stop() if sampler else None
        finally:
            peak_memory = memory.stop()

        summary = latency_summary(latencies)
        mean_ms = summary["mean_ms"]
        return make_benchmark_report(
            model_path=request.model_path,
            model_id=request.model_id,
            model_revision=request.model_revision,
            hardware=request.hardware,
            runtime="onnxruntime",
            runtime_version=ort.__version__,
            provider=provider,
            provider_version=f"onnxruntime-{ort.__version__}",
            precision=request.precision,
            quantization=request.quantization,
            batch_size=_infer_batch_size(request.batch_size, input_shapes),
            input_shapes=input_shapes,
            warmup=request.warmup,
            iterations=request.iterations,
            backend_options={
                "requested_provider": provider,
                "active_providers": active_providers,
                "provider_options": session.get_provider_options().get(provider, {}),
                "ort_build_info": getattr(ort, "get_build_info", lambda: None)(),
                "cpu_fallback_disabled": provider != "CPUExecutionProvider",
                "synthetic_inputs": True,
            },
            load_ms=load_ms,
            latency=summary,
            throughput_fps=1000.0 / mean_ms if mean_ms else None,
            peak_memory_mb=peak_memory,
            power=power,
            power_scope=scope,
            command=request.command,
            notes="Synthetic deterministic inputs measure runtime execution, not task accuracy.",
        )


def parse_trtexec_output(text: str) -> dict[str, Any]:
    throughput = None
    match = re.search(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:qps|QPS)", text)
    if match:
        throughput = float(match.group(1))
    latency_match = re.search(
        r"Latency:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*"
        r"mean\s*=\s*([0-9.]+)\s*ms,\s*median\s*=\s*([0-9.]+)\s*ms,\s*"
        r"percentile\(90%\)\s*=\s*([0-9.]+)\s*ms,\s*percentile\(95%\)\s*=\s*"
        r"([0-9.]+)\s*ms,\s*percentile\(99%\)\s*=\s*([0-9.]+)\s*ms",
        text,
    )
    if not latency_match:
        raise BackendError("could not parse TensorRT trtexec latency summary")
    minimum, maximum, mean, median, p90, p95, p99 = map(float, latency_match.groups())
    return {
        "latency": {
            "min_ms": minimum,
            "mean_ms": mean,
            "median_ms": median,
            "p50_ms": median,
            "p90_ms": p90,
            "p95_ms": p95,
            "p99_ms": p99,
            "max_ms": maximum,
            "stdev_ms": None,
        },
        "throughput_fps": throughput,
    }


def build_trtexec_command(request: BenchmarkRequest, executable: str = "trtexec") -> list[str]:
    suffix = request.model_path.suffix.casefold()
    if suffix in {".engine", ".plan"}:
        command = [executable, f"--loadEngine={request.model_path}"]
    elif suffix == ".onnx":
        command = [executable, f"--onnx={request.model_path}"]
    else:
        raise BackendError("TensorRT backend supports ONNX or serialized engine/plan files")
    command += [f"--iterations={request.iterations}", "--duration=0", f"--warmUp={request.warmup}"]
    precision = request.precision.casefold()
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.append("--int8")
    if request.input_shapes:
        shape_arg = ",".join(
            f"{name}:{'x'.join(str(dim) for dim in dims)}"
            for name, dims in sorted(request.input_shapes.items())
        )
        command.append(f"--shapes={shape_arg}")
    return command


class TensorRTBackend(BenchmarkBackend):
    name = "tensorrt"

    def availability(self) -> BackendAvailability:
        executable = shutil.which("trtexec")
        if not executable:
            return BackendAvailability(self.name, False, detail="trtexec not found")
        try:
            result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=4, check=False)
            detail = (result.stdout or result.stderr).strip().splitlines()
            version_text = detail[0] if detail else None
        except (OSError, subprocess.SubprocessError):
            version_text = None
        return BackendAvailability(self.name, True, version_text, executable)

    def benchmark(self, request: BenchmarkRequest) -> dict[str, Any]:
        availability = self.availability()
        if not availability.available or not availability.executable:
            raise BackendError("TensorRT backend unavailable: trtexec not found")
        if request.model_path.suffix.casefold() in {".engine", ".plan"} and not request.trusted_artifact:
            raise BackendError(
                "refusing to deserialize an untrusted TensorRT engine; use --trust-artifact only for an engine you built or independently trust"
            )
        native_request = request
        if request.model_path.suffix.casefold() == ".onnx" and not request.input_shapes:
            native_request = replace(request, input_shapes=_resolve_onnx_input_shapes(request))
        command = build_trtexec_command(native_request, availability.executable)
        reader, scope = power_reader(request.hardware.platform)
        sampler = PowerSampler(reader) if reader else None
        if sampler:
            sampler.start()
        started = time.perf_counter_ns()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
        finally:
            power = sampler.stop() if sampler else None
        wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode != 0:
            raise BackendError(f"trtexec failed with exit code {result.returncode}: {output[-1000:]}")
        parsed = parse_trtexec_output(output)
        return make_benchmark_report(
            model_path=request.model_path,
            model_id=request.model_id,
            model_revision=request.model_revision,
            hardware=request.hardware,
            runtime="tensorrt",
            runtime_version=availability.version,
            provider="trtexec",
            provider_version=availability.version,
            precision=request.precision,
            quantization=request.quantization,
            batch_size=_infer_batch_size(native_request.batch_size, native_request.input_shapes),
            input_shapes=native_request.input_shapes,
            warmup=request.warmup,
            iterations=request.iterations,
            backend_options={"native_warmup_semantics": "milliseconds", "wall_ms": wall_ms},
            load_ms=None,
            latency=parsed["latency"],
            throughput_fps=parsed["throughput_fps"],
            peak_memory_mb=None,
            power=power,
            power_scope=scope,
            command=request.command or " ".join(command),
            notes="trtexec native benchmark; ONNX engine build time is not reported as inference load time.",
        )


def parse_openvino_output(text: str) -> dict[str, Any]:
    def value(label: str) -> float | None:
        match = re.search(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    throughput_match = re.search(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:FPS|fps)", text)
    avg = value("Average") or value("Latency")
    median = value("Median")
    minimum = value("Min")
    maximum = value("Max")
    if avg is None and median is None:
        raise BackendError("could not parse OpenVINO benchmark_app latency summary")
    representative = median or avg
    return {
        "latency": {
            "min_ms": minimum,
            "mean_ms": avg,
            "median_ms": median,
            "p50_ms": median,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": maximum,
            "stdev_ms": None,
        },
        "throughput_fps": float(throughput_match.group(1)) if throughput_match else (
            1000.0 / representative if representative else None
        ),
    }


def build_openvino_command(request: BenchmarkRequest, executable: str = "benchmark_app") -> list[str]:
    command = [executable, "-m", str(request.model_path), "-hint", "latency", "-niter", str(request.iterations)]
    if request.device:
        command += ["-d", request.device]
    if request.batch_size:
        command += ["-b", str(request.batch_size)]
    if request.input_shapes:
        shapes = ",".join(
            f"{name}[{','.join(str(dim) for dim in dims)}]"
            for name, dims in sorted(request.input_shapes.items())
        )
        command += ["-shape", shapes]
    return command


class OpenVINOBackend(BenchmarkBackend):
    name = "openvino"

    def availability(self) -> BackendAvailability:
        executable = shutil.which("benchmark_app")
        if not executable:
            return BackendAvailability(self.name, False, detail="benchmark_app not found")
        try:
            result = subprocess.run([executable, "-h"], capture_output=True, text=True, timeout=4, check=False)
            match = re.search(r"OpenVINO[^\n]*?([0-9]{4}\.[0-9.]+)", result.stdout + result.stderr)
            version_text = match.group(1) if match else None
        except (OSError, subprocess.SubprocessError):
            version_text = None
        return BackendAvailability(self.name, True, version_text, executable)

    def benchmark(self, request: BenchmarkRequest) -> dict[str, Any]:
        availability = self.availability()
        if not availability.available or not availability.executable:
            raise BackendError("OpenVINO backend unavailable: benchmark_app not found")
        native_request = request
        if request.model_path.suffix.casefold() == ".onnx" and not request.input_shapes:
            native_request = replace(request, input_shapes=_resolve_onnx_input_shapes(request))
        device = request.device or "CPU"
        try:
            import openvino as ov
            available_devices = tuple(ov.Core().available_devices)
        except Exception:  # noqa: BLE001
            available_devices = ()
        if available_devices and device not in available_devices:
            raise BackendError(
                f"OpenVINO device {device!r} unavailable. Available: {', '.join(available_devices)}"
            )
        native_request = replace(native_request, device=device)
        command = build_openvino_command(native_request, availability.executable)
        started = time.perf_counter_ns()
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
        wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode != 0:
            raise BackendError(f"benchmark_app failed with exit code {result.returncode}: {output[-1000:]}")
        parsed = parse_openvino_output(output)
        return make_benchmark_report(
            model_path=request.model_path,
            model_id=request.model_id,
            model_revision=request.model_revision,
            hardware=request.hardware,
            runtime="openvino",
            runtime_version=availability.version,
            provider=device,
            provider_version=availability.version,
            precision=request.precision,
            quantization=request.quantization,
            batch_size=_infer_batch_size(native_request.batch_size, native_request.input_shapes),
            input_shapes=native_request.input_shapes,
            warmup=0,
            iterations=request.iterations,
            backend_options={
                "performance_hint": "latency",
                "native_warmup": True,
                "requested_warmup_count_not_applied": request.warmup,
                "device": device,
                "available_devices": list(available_devices),
                "wall_ms": wall_ms,
            },
            load_ms=None,
            latency=parsed["latency"],
            throughput_fps=parsed["throughput_fps"],
            peak_memory_mb=None,
            power=None,
            power_scope=None,
            command=request.command or " ".join(command),
            notes="OpenVINO benchmark_app measures inference without application pre/post-processing.",
        )


class CoreMLBackend(BenchmarkBackend):
    name = "coreml"

    def availability(self) -> BackendAvailability:
        import platform
        if platform.system() != "Darwin":
            return BackendAvailability(self.name, False, detail="Core ML requires macOS")
        try:
            import coremltools as ct
        except ImportError:
            return BackendAvailability(self.name, False, detail="coremltools is not installed")
        return BackendAvailability(self.name, True, getattr(ct, "__version__", None))

    def benchmark(self, request: BenchmarkRequest) -> dict[str, Any]:
        availability = self.availability()
        if not availability.available:
            raise BackendError(f"Core ML backend unavailable: {availability.detail}")
        try:
            import coremltools as ct
            import numpy as np
        except ImportError as exc:
            raise BackendError("Core ML benchmarking requires coremltools and numpy") from exc
        memory = MemorySampler()
        memory.start()
        started = time.perf_counter_ns()
        try:
            compute_name = (request.compute_units or "ALL").strip().upper()
            compute_map = {
                "ALL": ct.ComputeUnit.ALL,
                "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
                "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
                "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            }
            if compute_name not in compute_map:
                raise BackendError(
                    "Core ML compute units must be ALL, CPU_ONLY, CPU_AND_GPU or CPU_AND_NE"
                )
            model = ct.models.MLModel(
                str(request.model_path), compute_units=compute_map[compute_name]
            )
            load_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            spec = model.get_spec()
            feeds: dict[str, Any] = {}
            input_shapes: dict[str, list[int]] = {}
            rng = np.random.default_rng(0)
            for item in spec.description.input:
                kind = item.type.WhichOneof("Type")
                if kind != "multiArrayType":
                    raise BackendError(
                        "CoreMLBackend currently benchmarks numeric MLMultiArray inputs only; "
                        f"input {item.name!r} is {kind}"
                    )
                shape = [int(dim) for dim in item.type.multiArrayType.shape]
                if not shape or any(dim <= 0 for dim in shape):
                    raise BackendError(f"Core ML input {item.name!r} has unresolved shape")
                feeds[item.name] = rng.random(shape, dtype=np.float32)
                input_shapes[item.name] = shape
            for _ in range(request.warmup):
                model.predict(feeds)
            latencies=[]
            for _ in range(request.iterations):
                start=time.perf_counter_ns()
                model.predict(feeds)
                latencies.append((time.perf_counter_ns()-start)/1_000_000.0)
        finally:
            peak_memory=memory.stop()
        summary=latency_summary(latencies)
        mean_ms=summary["mean_ms"]
        return make_benchmark_report(
            model_path=request.model_path, model_id=request.model_id,
            model_revision=request.model_revision, hardware=request.hardware,
            runtime="coreml", runtime_version=availability.version,
            provider="MLModel.predict", provider_version=availability.version,
            precision=request.precision, quantization=request.quantization,
            batch_size=_infer_batch_size(request.batch_size, input_shapes), input_shapes=input_shapes,
            warmup=request.warmup, iterations=request.iterations,
            backend_options={"compute_units":compute_name,"synthetic_inputs":True},
            load_ms=load_ms, latency=summary,
            throughput_fps=1000.0/mean_ms if mean_ms else None,
            peak_memory_mb=peak_memory, power=None, power_scope=None,
            command=request.command,
            notes="Core ML numeric-input benchmark. Use Xcode/Instruments for detailed compute-unit profiling.",
        )


BACKENDS: dict[str, type[BenchmarkBackend]] = {
    "onnxruntime": OnnxRuntimeBackend,
    "onnx": OnnxRuntimeBackend,
    "tensorrt": TensorRTBackend,
    "openvino": OpenVINOBackend,
    "coreml": CoreMLBackend,
}


def get_backend(name: str) -> BenchmarkBackend:
    normalized = name.strip().casefold()
    backend_type = BACKENDS.get(normalized)
    if backend_type is None:
        raise BackendError(f"unknown benchmark backend: {name}")
    return backend_type()


def infer_backend(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".onnx":
        return "onnxruntime"
    if suffix in {".engine", ".plan"}:
        return "tensorrt"
    if suffix in {".mlmodel", ".mlpackage"}:
        return "coreml"
    if suffix in {".xml", ".bin"}:
        return "openvino"
    raise BackendError("cannot infer benchmark backend from artifact extension; use --backend")


def backend_status() -> list[BackendAvailability]:
    seen: set[type[BenchmarkBackend]] = set()
    result=[]
    for backend_type in BACKENDS.values():
        if backend_type in seen:
            continue
        seen.add(backend_type)
        result.append(backend_type().availability())
    return sorted(result, key=lambda item: item.name)


def run_benchmark(request: BenchmarkRequest, backend_name: str | None = None) -> dict[str, Any]:
    if request.hardware.os_name == "profile":
        raise BackendError(
            "local benchmark evidence requires detected hardware; bundled profiles are screening targets only"
        )
    try:
        before = artifact_sha256(request.model_path)
    except (OSError, ValueError) as exc:
        raise BackendError(f"artifact identity could not be established before benchmark: {exc}") from exc
    if request.expected_sha256 and before.casefold() != request.expected_sha256.casefold():
        raise BackendError(
            f"artifact identity changed before benchmark: expected {request.expected_sha256}, got {before}"
        )
    backend = get_backend(backend_name or infer_backend(request.model_path))
    report = backend.benchmark(request)
    try:
        after = artifact_sha256(request.model_path)
    except (OSError, ValueError) as exc:
        raise BackendError(f"artifact identity could not be revalidated after benchmark: {exc}") from exc
    if after.casefold() != before.casefold():
        raise BackendError(
            f"artifact changed during benchmark: expected {before}, got {after}"
        )
    reported = (report.get("artifact") or {}).get("sha256")
    if not isinstance(reported, str) or reported.casefold() != before.casefold():
        raise BackendError("benchmark report artifact identity does not match the measured input")
    return report

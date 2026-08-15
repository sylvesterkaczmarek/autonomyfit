from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class BenchmarkResult:
    model_path: str
    provider: str
    iterations: int
    warmup: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    power_w_mean: float | None
    input_shapes: dict[str, list[int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def parse_tegrastats_power_w(line: str) -> float | None:
    match = re.search(r"VDD_IN\s+(\d+)mW", line)
    return float(match.group(1)) / 1000.0 if match else None


def parse_nvidia_smi_power_w(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def _read_nvidia_power() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi_power_w(result.stdout)


def _read_ina3221_vdd_in_power_w(hwmon_dir: Path) -> float | None:
    for channel in range(1, 4):
        label = hwmon_dir / f"in{channel}_label"
        voltage = hwmon_dir / f"in{channel}_input"
        current = hwmon_dir / f"curr{channel}_input"
        try:
            if label.read_text().strip() != "VDD_IN":
                continue
            millivolts = float(voltage.read_text().strip())
            milliamps = float(current.read_text().strip())
        except (OSError, ValueError):
            continue
        return (millivolts * milliamps) / 1_000_000.0
    return None


def _read_jetson_power() -> float | None:
    roots = [
        Path("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"),
        Path("/sys/class/hwmon"),
    ]
    seen: set[Path] = set()
    for root in roots:
        try:
            candidates = sorted(root.glob("hwmon*"))
        except OSError:
            continue
        for hwmon_dir in candidates:
            resolved = hwmon_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            power = _read_ina3221_vdd_in_power_w(hwmon_dir)
            if power is not None:
                return power
    return None



class PowerSampler:
    def __init__(self, reader: Callable[[], float | None], interval: float = 0.15) -> None:
        self.reader = reader
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.reader()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return statistics.mean(self.samples) if self.samples else None


def _power_reader(platform_kind: str) -> Callable[[], float | None] | None:
    if platform_kind == "jetson":
        return _read_jetson_power
    if platform_kind == "nvidia":
        return _read_nvidia_power
    return None


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


def _shape_for_input(shape: list[object], override: list[int] | None) -> list[int]:
    if override:
        return override
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif len(resolved) == 0:
            resolved.append(1)
        else:
            raise ValueError("Dynamic non-batch input shape requires --shape")
    return resolved


def benchmark_onnx(
    model_path: Path,
    platform_kind: str,
    iterations: int = 50,
    warmup: int = 10,
    shape_override: list[int] | None = None,
    provider: str | None = None,
) -> BenchmarkResult:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX benchmarking requires the benchmark extra: pip install 'autonomyfit[benchmark]'"
        ) from exc

    available = ort.get_available_providers()
    if provider is None:
        preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        provider = next((name for name in preferred if name in available), available[0])
    if provider not in available:
        raise ValueError(f"Provider {provider!r} is unavailable. Available: {', '.join(available)}")

    session = ort.InferenceSession(str(model_path), providers=[provider])
    feeds: dict[str, object] = {}
    input_shapes: dict[str, list[int]] = {}
    rng = np.random.default_rng(0)
    for input_meta in session.get_inputs():
        resolved = _shape_for_input(list(input_meta.shape), shape_override)
        dtype = _numpy_dtype(input_meta.type)
        if np.issubdtype(dtype, np.floating):
            value = rng.random(resolved, dtype=np.float32).astype(dtype)
        elif dtype == np.bool_:
            value = np.zeros(resolved, dtype=dtype)
        else:
            value = np.zeros(resolved, dtype=dtype)
        feeds[input_meta.name] = value
        input_shapes[input_meta.name] = resolved

    for _ in range(warmup):
        session.run(None, feeds)

    sampler = None
    reader = _power_reader(platform_kind)
    if reader is not None:
        sampler = PowerSampler(reader)
        sampler.start()

    latencies_ms: list[float] = []
    try:
        for _ in range(iterations):
            start = time.perf_counter_ns()
            session.run(None, feeds)
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            latencies_ms.append(elapsed_ms)
    finally:
        power = sampler.stop() if sampler else None

    mean_ms = statistics.mean(latencies_ms)
    return BenchmarkResult(
        model_path=str(model_path),
        provider=provider,
        iterations=iterations,
        warmup=warmup,
        mean_ms=mean_ms,
        p50_ms=percentile(latencies_ms, 0.50),
        p95_ms=percentile(latencies_ms, 0.95),
        p99_ms=percentile(latencies_ms, 0.99),
        fps=1000.0 / mean_ms,
        power_w_mean=power,
        input_shapes=input_shapes,
    )


def save_result(result: BenchmarkResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")

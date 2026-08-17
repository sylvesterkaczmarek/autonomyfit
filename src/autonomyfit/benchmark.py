from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil

from .integrity import artifact_sha256, artifact_size_bytes
from .integrity import sha256_file as _sha256_file
from .models import HardwareProfile

RANDOM_SEED = 0


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Backward-compatible SHA-256 helper delegated to the integrity layer."""
    return _sha256_file(path, chunk_size)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _shape_for_input(shape: list[object], override: list[int] | None) -> list[int]:
    """Resolve an ONNX-style shape while preserving the 0.3 compatibility helper."""
    if override:
        if len(override) != len(shape):
            raise ValueError(
                f"--shape has {len(override)} dimensions but the model input expects {len(shape)}"
            )
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


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency samples cannot be empty")
    return {
        "min_ms": min(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
        "stdev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def parse_tegrastats_power_w(line: str) -> float | None:
    match = re.search(r"VDD_IN\s+(\d+)mW", line)
    return float(match.group(1)) / 1000.0 if match else None


def parse_nvidia_smi_power_w(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def _read_nvidia_power() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
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
    roots = [Path("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"), Path("/sys/class/hwmon")]
    seen: set[Path] = set()
    for root in roots:
        try:
            candidates = sorted(root.glob("hwmon*"))
        except OSError:
            continue
        for hwmon_dir in candidates:
            try:
                resolved = hwmon_dir.resolve()
            except OSError:
                resolved = hwmon_dir
            if resolved in seen:
                continue
            seen.add(resolved)
            power = _read_ina3221_vdd_in_power_w(hwmon_dir)
            if power is not None:
                return power
    return None



def read_thermal_c(platform_kind: str) -> dict[str, float]:
    if platform_kind == "nvidia":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                check=False, capture_output=True, text=True, timeout=1.0,
            )
            value = parse_nvidia_smi_power_w(result.stdout)
            return {"gpu": value} if value is not None else {}
        except (OSError, subprocess.SubprocessError):
            return {}
    if platform_kind == "jetson":
        values: dict[str, float] = {}
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            try:
                name = (zone / "type").read_text().strip()
                milli_c = float((zone / "temp").read_text().strip())
            except (OSError, ValueError):
                continue
            values[name] = milli_c / 1000.0
        return values
    return {}


def power_reader(platform_kind: str) -> tuple[Callable[[], float | None] | None, str | None]:
    if platform_kind == "jetson":
        return _read_jetson_power, "Jetson VDD_IN rail"
    if platform_kind == "nvidia":
        return _read_nvidia_power, "NVIDIA GPU board power.draw"
    return None, None


class PowerSampler:
    def __init__(self, reader: Callable[[], float | None], interval: float = 0.15) -> None:
        self.reader = reader
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self.reader()
            if value is not None:
                self.samples.append((time.monotonic(), value))
            self._stop.wait(self.interval)

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {"mean_w": None, "max_w": None, "energy_j": None}
        values = [value for _, value in self.samples]
        energy_j = None
        if len(self.samples) >= 2:
            energy_j = 0.0
            for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
                energy_j += (p0 + p1) * 0.5 * (t1 - t0)
        return {
            "mean_w": statistics.mean(values),
            "max_w": max(values),
            "energy_j": energy_j,
        }


class MemorySampler:
    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            except (psutil.Error, OSError):
                pass
            self._stop.wait(self.interval)

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.peak_rss / (1024 * 1024) if self.peak_rss else None



def _machine_identity_hash() -> str:
    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return hashlib.sha256(value.encode()).hexdigest()[:20]
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                check=False, capture_output=True, text=True, timeout=2.0,
            )
            match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', result.stdout)
            if match:
                return hashlib.sha256(match.group(1).encode()).hexdigest()[:20]
        except (OSError, subprocess.SubprocessError):
            pass
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:20]


def hardware_evidence_id(hardware: HardwareProfile) -> str:
    if hardware.os_name == "profile" and hardware.matched_profile:
        return hardware.matched_profile
    payload = {
        "machine": _machine_identity_hash(),
        "platform": hardware.platform,
        "architecture": hardware.architecture,
        "cpu": hardware.cpu,
        "gpu": hardware.gpu,
        "ram_total_gb": round(hardware.ram_total_gb, 2),
        "accelerator_memory_gb": (
            round(hardware.accelerator_memory_gb, 2)
            if hardware.accelerator_memory_gb is not None
            else None
        ),
        "memory_topology": hardware.memory_topology,
        "matched_profile": hardware.matched_profile,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"local-{hardware.platform}-{digest}"


def _hostname_hash() -> str:
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16]


def software_stack_fingerprint(
    hardware: HardwareProfile,
    *,
    runtime: str,
    runtime_version: str | None,
    provider: str | None,
    provider_version: str | None,
) -> str:
    payload = {
        "platform": hardware.platform,
        "os": hardware.os_name,
        "architecture": hardware.architecture,
        "driver": hardware.driver,
        "jetpack": hardware.jetpack,
        "power_mode": hardware.power_mode,
        "software_stack": list(hardware.software_stack),
        "runtime": runtime,
        "runtime_version": runtime_version,
        "provider": provider,
        "provider_version": provider_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def environment_fingerprint(
    *, hardware: dict[str, Any], software: dict[str, Any], execution: dict[str, Any]
) -> str:
    payload = json.dumps(
        {"hardware": hardware, "software": software, "execution": execution},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def artifact_format(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".engine", ".plan"}:
        return "tensorrt-engine"
    if suffix == ".mlmodel":
        return "coreml-model"
    if suffix == ".mlpackage" or path.name.endswith(".mlpackage"):
        return "coreml-package"
    if suffix in {".xml", ".bin"}:
        return "openvino-ir"
    return suffix.lstrip(".") or "unknown"


def make_benchmark_report(
    *,
    model_path: Path,
    model_id: str,
    model_revision: str | None,
    hardware: HardwareProfile,
    runtime: str,
    runtime_version: str | None,
    provider: str | None,
    provider_version: str | None,
    precision: str,
    quantization: str | None,
    batch_size: int | None,
    input_shapes: dict[str, list[int]],
    warmup: int,
    iterations: int,
    backend_options: dict[str, Any],
    load_ms: float | None,
    latency: dict[str, float | None],
    throughput_fps: float | None,
    peak_memory_mb: float | None,
    power: dict[str, float | None] | None,
    power_scope: str | None,
    command: str | None,
    notes: str | None = None,
) -> dict[str, Any]:
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    artifact_hash = artifact_sha256(model_path)
    hardware_dict = {
        "id": hardware_evidence_id(hardware),
        "source": "profile" if hardware.os_name == "profile" else "detected",
        "profile_id": hardware.matched_profile,
        "platform": hardware.platform,
        "device": hardware.gpu,
        "cpu": hardware.cpu,
        "ram_total_gb": hardware.ram_total_gb,
        "os": hardware.os_name,
        "architecture": hardware.architecture,
        "driver": hardware.driver,
        "jetpack": hardware.jetpack,
        "power_mode": hardware.power_mode,
        "software_stack": list(hardware.software_stack),
        "clocks": {},
        "thermal_c": read_thermal_c(hardware.platform),
    }
    software_dict = {
        "runtime": runtime,
        "runtime_version": runtime_version,
        "provider": provider,
        "provider_version": provider_version,
        "python_version": sys.version.split()[0],
        "autonomyfit_version": _package_version("autonomyfit") or "source-tree",
    }
    execution = {
        "precision": precision,
        "quantization": quantization,
        "batch_size": batch_size,
        "input_shapes": input_shapes,
        "warmup": warmup,
        "iterations": iterations,
        "random_seed": RANDOM_SEED,
        "backend_options": backend_options,
    }
    fingerprint = environment_fingerprint(
        hardware=hardware_dict, software=software_dict, execution=execution
    )
    stack_fingerprint = software_stack_fingerprint(
        hardware,
        runtime=runtime,
        runtime_version=runtime_version,
        provider=provider,
        provider_version=provider_version,
    )
    benchmark_id = "local-" + hashlib.sha256(
        f"{created}|{artifact_hash}|{hardware_dict['id']}|{runtime}|{precision}".encode()
    ).hexdigest()[:24]
    power_payload = None
    if power is not None:
        power_payload = {
            "mean_w": power.get("mean_w"),
            "max_w": power.get("max_w"),
            "energy_j": power.get("energy_j"),
            "scope": power_scope,
        }
    return {
        "schema_version": 2,
        "benchmark_id": benchmark_id,
        "created_at": created,
        "quality": "local-measured",
        "measurement": {
            "machine_source": "profile" if hardware.os_name == "profile" else "detected",
            "profile_only": hardware.os_name == "profile",
            "artifact_identity_verified": True,
        },
        "notes": notes,
        "model": {"id": model_id, "revision": model_revision},
        "artifact": {
            "path": model_path.name,
            "format": artifact_format(model_path),
            "sha256": artifact_hash,
            "size_bytes": artifact_size_bytes(model_path),
        },
        "hardware": hardware_dict,
        "software": software_dict,
        "execution": execution,
        "metrics": {
            "load_ms": load_ms,
            "latency": latency,
            "throughput_fps": throughput_fps,
            "peak_memory_mb": peak_memory_mb,
            "peak_memory_scope": "process RSS" if peak_memory_mb is not None else None,
            "power": power_payload,
        },
        "reproducibility": {
            "command": command,
            "hostname_hash": _hostname_hash(),
            "environment_fingerprint": fingerprint,
            "software_stack_fingerprint": stack_fingerprint,
        },
    }


def save_result(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

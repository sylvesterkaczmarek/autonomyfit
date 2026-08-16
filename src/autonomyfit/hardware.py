from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

import psutil

from .catalog import load_hardware_profiles
from .models import HardwareProfile, RuntimeCapability

_GIB = 1024**3


def _run(command: list[str], timeout: float = 2.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _detect_runtimes(platform_kind: str) -> tuple[RuntimeCapability, ...]:
    capabilities: list[RuntimeCapability] = []

    ort_version = _package_version("onnxruntime") or _package_version("onnxruntime-gpu")
    ort_detail = None
    if ort_version:
        try:
            import onnxruntime as ort

            ort_detail = ", ".join(ort.get_available_providers())
        except Exception:  # noqa: BLE001
            ort_detail = "installed, provider query failed"
    capabilities.append(
        RuntimeCapability("onnxruntime", bool(ort_version), ort_version, ort_detail)
    )

    torch_version = _package_version("torch")
    torch_detail = None
    if torch_version:
        try:
            import torch

            if torch.cuda.is_available():
                torch_detail = f"CUDA {torch.version.cuda or 'available'}"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch_detail = "MPS available"
            else:
                torch_detail = "CPU"
        except Exception:  # noqa: BLE001
            torch_detail = "installed, backend query failed"
    capabilities.append(
        RuntimeCapability("pytorch", bool(torch_version), torch_version, torch_detail)
    )

    trt_version = _package_version("tensorrt")
    trtexec = shutil.which("trtexec")
    if not trt_version and trtexec:
        version_text = _run([trtexec, "--version"], timeout=4.0)
        trt_version = version_text.splitlines()[0] if version_text else "CLI"
    capabilities.append(
        RuntimeCapability(
            "tensorrt",
            bool(trt_version or trtexec),
            trt_version,
            "Jetson/NVIDIA GPU" if platform_kind in {"jetson", "nvidia"} else None,
        )
    )

    coreml_version = _package_version("coremltools")
    capabilities.append(
        RuntimeCapability(
            "coreml",
            bool(coreml_version),
            coreml_version,
            "Apple platform" if platform_kind == "apple" else None,
        )
    )

    transformers_version = _package_version("transformers")
    capabilities.append(
        RuntimeCapability("transformers", bool(transformers_version), transformers_version)
    )
    return tuple(capabilities)


def _jetson_model() -> str | None:
    candidates = [Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")]
    for path in candidates:
        try:
            return path.read_bytes().decode("utf-8", errors="ignore").strip("\x00\n ")
        except OSError:
            continue
    return None


def _jetpack_version() -> str | None:
    release = Path("/etc/nv_tegra_release")
    if release.exists():
        text = release.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"R(\d+)\s*\(release\).*REVISION:\s*([\d.]+)", text)
        if match:
            return f"L4T R{match.group(1)}.{match.group(2)}"
        return text.splitlines()[0].strip()
    return None


def _jetson_power_mode() -> str | None:
    if not shutil.which("nvpmodel"):
        return None
    text = _run(["nvpmodel", "-q"], timeout=3.0)
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("NV Power Mode") and not stripped.isdigit():
            return stripped
    return text.splitlines()[0].strip()


def _nvidia_smi() -> tuple[str | None, float | None, str | None]:
    if not shutil.which("nvidia-smi"):
        return None, None, None
    text = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=4.0,
    )
    if not text:
        return None, None, None
    first = text.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return first, None, None
    try:
        memory_gb = float(parts[1]) / 1024.0
    except ValueError:
        memory_gb = None
    return parts[0], memory_gb, parts[2]


def _apple_chip() -> str | None:
    text = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if text:
        return text
    text = _run(["system_profiler", "SPHardwareDataType"], timeout=5.0)
    if text:
        match = re.search(r"Chip:\s*(.+)", text)
        if match:
            return match.group(1).strip()
    return None


def match_hardware_profile(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.lower()
    profiles = load_hardware_profiles()
    best: tuple[int, str] | None = None
    for profile_id, item in profiles.items():
        aliases = [profile_id, item["display_name"], *item.get("aliases", [])]
        for alias in aliases:
            alias_norm = alias.lower()
            if alias_norm in normalized or normalized in alias_norm:
                score = len(alias_norm)
                if best is None or score > best[0]:
                    best = (score, profile_id)
    return best[1] if best else None


def hardware_from_profile(profile_id: str) -> HardwareProfile:
    profiles = load_hardware_profiles()
    if profile_id not in profiles:
        valid = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown hardware profile '{profile_id}'. Available: {valid}")
    item = profiles[profile_id]
    platform_kind = item["platform"]
    ram = float(item["memory_gb"])
    return HardwareProfile(
        platform=platform_kind,
        os_name="profile",
        architecture=item.get("architecture", "unknown"),
        cpu=item.get("cpu", "profile-defined"),
        ram_total_gb=ram,
        ram_available_gb=ram * 0.8,
        gpu=item.get("gpu"),
        accelerator_memory_gb=(ram * 0.8) if item.get("unified_memory") else item.get("vram_gb"),
        unified_memory=bool(item.get("unified_memory", False)),
        jetpack=item.get("jetpack"),
        matched_profile=profile_id,
        runtimes=tuple(
            RuntimeCapability(name, True, detail="profile capability")
            for name in item.get("supported_runtimes", [])
        ),
    )


def detect_hardware() -> HardwareProfile:
    vm = psutil.virtual_memory()
    total_gb = vm.total / _GIB
    available_gb = vm.available / _GIB
    machine = platform.machine() or "unknown"
    os_name = platform.system() or os.name
    cpu = platform.processor() or platform.uname().processor or "unknown"

    jetson = _jetson_model()
    if jetson or Path("/etc/nv_tegra_release").exists():
        gpu = jetson or "NVIDIA Jetson"
        matched = match_hardware_profile(gpu)
        return HardwareProfile(
            platform="jetson",
            os_name=os_name,
            architecture=machine,
            cpu=cpu,
            ram_total_gb=total_gb,
            ram_available_gb=available_gb,
            gpu=gpu,
            accelerator_memory_gb=available_gb,
            unified_memory=True,
            jetpack=_jetpack_version(),
            power_mode=_jetson_power_mode(),
            matched_profile=matched,
            runtimes=_detect_runtimes("jetson"),
        )

    if platform.system() == "Darwin" and machine in {"arm64", "aarch64"}:
        chip = _apple_chip() or cpu or "Apple Silicon"
        return HardwareProfile(
            platform="apple",
            os_name=os_name,
            architecture=machine,
            cpu=chip,
            ram_total_gb=total_gb,
            ram_available_gb=available_gb,
            gpu=chip,
            accelerator_memory_gb=available_gb,
            unified_memory=True,
            matched_profile=match_hardware_profile(chip),
            runtimes=_detect_runtimes("apple"),
        )

    gpu, vram_gb, driver = _nvidia_smi()
    if gpu:
        return HardwareProfile(
            platform="nvidia",
            os_name=os_name,
            architecture=machine,
            cpu=cpu,
            ram_total_gb=total_gb,
            ram_available_gb=available_gb,
            gpu=gpu,
            accelerator_memory_gb=vram_gb,
            driver=driver,
            matched_profile=match_hardware_profile(gpu),
            runtimes=_detect_runtimes("nvidia"),
        )

    return HardwareProfile(
        platform="cpu",
        os_name=os_name,
        architecture=machine,
        cpu=cpu,
        ram_total_gb=total_gb,
        ram_available_gb=available_gb,
        runtimes=_detect_runtimes("cpu"),
    )

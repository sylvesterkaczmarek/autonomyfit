from __future__ import annotations

import importlib.metadata
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
_BRIDGE_NAMES = {"qnn", "xnnpack", "openvino-ep", "coreml-ep", "tensorrt-ep", "cuda-ep", "vitisai"}


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
    ort_version = (
        _package_version("onnxruntime")
        or _package_version("onnxruntime-gpu")
        or _package_version("onnxruntime-openvino")
        or _package_version("onnxruntime-qnn")
    )
    providers: list[str] = []
    if ort_version:
        try:
            import onnxruntime as ort

            providers = list(ort.get_available_providers())
        except Exception:  # noqa: BLE001
            providers = []
    capabilities.append(
        RuntimeCapability(
            "onnxruntime",
            bool(ort_version),
            ort_version,
            ", ".join(providers) if providers else None,
            provider="CPUExecutionProvider" if ort_version else None,
            verified=True,
        )
    )
    provider_map = {
        "TensorrtExecutionProvider": "tensorrt-ep",
        "CUDAExecutionProvider": "cuda-ep",
        "OpenVINOExecutionProvider": "openvino-ep",
        "CoreMLExecutionProvider": "coreml-ep",
        "QNNExecutionProvider": "qnn",
        "XNNPACKExecutionProvider": "xnnpack",
        "VitisAIExecutionProvider": "vitisai",
    }
    for provider, name in provider_map.items():
        capabilities.append(
            RuntimeCapability(
                name,
                provider in providers,
                ort_version if provider in providers else None,
                "execution-provider availability; model operator coverage must be verified",
                provider=provider,
                verified=False,
            )
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
    capabilities.append(RuntimeCapability("pytorch", bool(torch_version), torch_version, torch_detail))

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
            "native TensorRT" if platform_kind in {"jetson", "nvidia"} else None,
        )
    )

    openvino_version = _package_version("openvino")
    benchmark_app = shutil.which("benchmark_app")
    capabilities.append(
        RuntimeCapability(
            "openvino",
            bool(openvino_version or benchmark_app),
            openvino_version,
            "native OpenVINO runtime",
        )
    )

    coreml_version = _package_version("coremltools")
    capabilities.append(
        RuntimeCapability(
            "coreml",
            bool(coreml_version and platform_kind == "apple"),
            coreml_version,
            "Apple Core ML" if platform_kind == "apple" else None,
        )
    )
    transformers_version = _package_version("transformers")
    capabilities.append(RuntimeCapability("transformers", bool(transformers_version), transformers_version))
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
    parts = [part.strip() for part in text.splitlines()[0].split(",")]
    if len(parts) < 3:
        return text.splitlines()[0], None, None
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


def _cpu_brand() -> str:
    value = platform.processor() or platform.uname().processor
    if value:
        return value
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "unknown"
    for key in ("model name", "Hardware", "Processor"):
        match = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "unknown"


def _openvino_devices() -> tuple[str, ...]:
    try:
        import openvino as ov

        return tuple(sorted(ov.Core().available_devices))
    except Exception:  # noqa: BLE001
        return ()


def match_hardware_profile(name: str | None, memory_gb: float | None = None) -> str | None:
    if not name:
        return None
    normalized = name.casefold()
    profiles = load_hardware_profiles()
    best: tuple[float, str] | None = None
    for profile_id, item in profiles.items():
        aliases = [profile_id, item["display_name"], *item.get("aliases", [])]
        for alias in aliases:
            alias_norm = alias.casefold()
            if alias_norm in normalized or normalized in alias_norm:
                score = float(len(alias_norm))
                if memory_gb is not None and item.get("memory_gb") is not None:
                    delta = abs(float(item["memory_gb"]) - memory_gb)
                    score -= min(delta, 64.0) / 100.0
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
    runtimes = tuple(
        RuntimeCapability(
            name,
            True,
            detail=(
                "profile execution-provider capability; model coverage unverified"
                if name in _BRIDGE_NAMES
                else "profile capability"
            ),
            verified=name not in _BRIDGE_NAMES,
        )
        for name in item.get("supported_runtimes", [])
    )
    accelerator_memory = item.get("vram_gb")
    if accelerator_memory is None and item.get("unified_memory"):
        accelerator_memory = ram * 0.8
    return HardwareProfile(
        platform=platform_kind,
        os_name="profile",
        architecture=item.get("architecture", "unknown"),
        cpu=item.get("cpu", "profile-defined"),
        ram_total_gb=ram,
        ram_available_gb=ram * 0.8,
        gpu=item.get("gpu"),
        accelerator_memory_gb=float(accelerator_memory) if accelerator_memory is not None else None,
        unified_memory=bool(item.get("unified_memory", False)),
        jetpack=item.get("jetpack"),
        power_mode=item.get("power_mode"),
        driver=item.get("driver"),
        matched_profile=profile_id,
        runtimes=runtimes,
        accelerator_type=item.get("accelerator_type", "cpu"),
        memory_topology=item.get("memory_topology", "system"),
        supported_precisions=tuple(item.get("supported_precisions", [])),
        max_power_w=float(item["max_power_w"]) if item.get("max_power_w") is not None else None,
        compute_tops=float(item["compute_tops"]) if item.get("compute_tops") is not None else None,
        hardware_limits=tuple(item.get("hardware_limits", [])),
        software_stack=tuple(item.get("software_stack", [])),
    )


def detect_hardware() -> HardwareProfile:
    vm = psutil.virtual_memory()
    total_gb = vm.total / _GIB
    available_gb = vm.available / _GIB
    machine = platform.machine() or "unknown"
    os_name = platform.system() or os.name
    cpu = _cpu_brand()

    jetson = _jetson_model()
    if jetson or Path("/etc/nv_tegra_release").exists():
        gpu = jetson or "NVIDIA Jetson"
        matched = match_hardware_profile(gpu, total_gb)
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
            accelerator_type="gpu",
            memory_topology="unified",
            supported_precisions=("fp32", "fp16", "bf16", "int8"),
            software_stack=tuple(value for value in (_jetpack_version(),) if value),
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
            matched_profile=match_hardware_profile(chip, total_gb),
            runtimes=_detect_runtimes("apple"),
            accelerator_type="gpu+npu",
            memory_topology="unified",
            supported_precisions=("fp32", "fp16", "int8"),
            software_stack=("Core ML", "Metal", "Apple Neural Engine"),
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
            matched_profile=match_hardware_profile(gpu, vram_gb),
            runtimes=_detect_runtimes("nvidia"),
            accelerator_type="gpu",
            memory_topology="discrete-vram",
            supported_precisions=("fp32", "fp16", "bf16", "int8"),
            software_stack=tuple(value for value in (f"NVIDIA driver {driver}" if driver else None,) if value),
        )

    cpu_lower = cpu.casefold()
    ov_devices = _openvino_devices()
    if "intel" in cpu_lower:
        accelerator = "npu+gpu+cpu" if any("NPU" in value for value in ov_devices) else "gpu+cpu"
        return HardwareProfile(
            platform="intel",
            os_name=os_name,
            architecture=machine,
            cpu=cpu,
            ram_total_gb=total_gb,
            ram_available_gb=available_gb,
            unified_memory=True,
            matched_profile=match_hardware_profile(cpu, total_gb),
            runtimes=_detect_runtimes("intel"),
            accelerator_type=accelerator,
            memory_topology="shared-system",
            supported_precisions=("fp32", "fp16", "int8"),
            software_stack=tuple(f"OpenVINO device {value}" for value in ov_devices),
        )
    if "amd" in cpu_lower or "ryzen" in cpu_lower:
        return HardwareProfile(
            platform="amd",
            os_name=os_name,
            architecture=machine,
            cpu=cpu,
            ram_total_gb=total_gb,
            ram_available_gb=available_gb,
            unified_memory=True,
            matched_profile=match_hardware_profile(cpu, total_gb),
            runtimes=_detect_runtimes("amd"),
            accelerator_type="cpu",
            memory_topology="shared-system",
            supported_precisions=("fp32", "fp16", "int8"),
            hardware_limits=("NPU/GPU capability is reported only when an execution provider exposes it",),
        )
    if any(value in cpu_lower for value in ("qualcomm", "snapdragon")):
        return HardwareProfile(
            platform="qualcomm",
            os_name=os_name,
            architecture=machine,
            cpu=cpu,
            ram_total_gb=total_gb,
            ram_available_gb=available_gb,
            unified_memory=True,
            matched_profile=match_hardware_profile(cpu, total_gb),
            runtimes=_detect_runtimes("qualcomm"),
            accelerator_type="cpu+npu",
            memory_topology="shared-system",
            supported_precisions=("fp32", "fp16", "int8"),
        )
    if machine.casefold() in {"arm64", "aarch64", "armv7l"}:
        return HardwareProfile(
            platform="arm",
            os_name=os_name,
            architecture=machine,
            cpu=cpu,
            ram_total_gb=total_gb,
            ram_available_gb=available_gb,
            unified_memory=True,
            matched_profile=match_hardware_profile(cpu, total_gb),
            runtimes=_detect_runtimes("arm"),
            accelerator_type="cpu",
            memory_topology="shared-system",
            supported_precisions=("fp32", "fp16", "int8"),
        )

    return HardwareProfile(
        platform="cpu",
        os_name=os_name,
        architecture=machine,
        cpu=cpu,
        ram_total_gb=total_gb,
        ram_available_gb=available_gb,
        runtimes=_detect_runtimes("cpu"),
        accelerator_type="cpu",
        memory_topology="system",
        supported_precisions=("fp32", "int8"),
    )

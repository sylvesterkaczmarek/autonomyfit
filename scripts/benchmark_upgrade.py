from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if addition.strip() in text:
        return
    target.write_text(text.rstrip() + "\n\n" + addition.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Exact machine and software-stack identity in benchmark.py
# ---------------------------------------------------------------------------
replace_once(
    "src/autonomyfit/benchmark.py",
    '''def hardware_evidence_id(hardware: HardwareProfile) -> str:\n    if hardware.matched_profile:\n        return hardware.matched_profile\n    payload = {\n        "platform": hardware.platform,\n        "architecture": hardware.architecture,\n        "cpu": hardware.cpu,\n        "gpu": hardware.gpu,\n        "ram_total_gb": round(hardware.ram_total_gb, 2),\n    }\n    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]\n    return f"local-{hardware.platform}-{digest}"\n\n\ndef _hostname_hash() -> str:\n    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16]\n''',
    '''def _machine_identity_hash() -> str:\n    for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):\n        try:\n            value = candidate.read_text(encoding="utf-8").strip()\n        except OSError:\n            continue\n        if value:\n            return hashlib.sha256(value.encode()).hexdigest()[:20]\n    if sys.platform == "darwin":\n        try:\n            result = subprocess.run(\n                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],\n                check=False, capture_output=True, text=True, timeout=2.0,\n            )\n            match = re.search(r'"IOPlatformUUID"\\s*=\\s*"([^"]+)"', result.stdout)\n            if match:\n                return hashlib.sha256(match.group(1).encode()).hexdigest()[:20]\n        except (OSError, subprocess.SubprocessError):\n            pass\n    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:20]\n\n\ndef hardware_evidence_id(hardware: HardwareProfile) -> str:\n    if hardware.os_name == "profile" and hardware.matched_profile:\n        return hardware.matched_profile\n    payload = {\n        "machine": _machine_identity_hash(),\n        "platform": hardware.platform,\n        "architecture": hardware.architecture,\n        "cpu": hardware.cpu,\n        "gpu": hardware.gpu,\n        "ram_total_gb": round(hardware.ram_total_gb, 2),\n        "accelerator_memory_gb": (\n            round(hardware.accelerator_memory_gb, 2)\n            if hardware.accelerator_memory_gb is not None\n            else None\n        ),\n        "memory_topology": hardware.memory_topology,\n        "matched_profile": hardware.matched_profile,\n    }\n    digest = hashlib.sha256(\n        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()\n    ).hexdigest()[:20]\n    return f"local-{hardware.platform}-{digest}"\n\n\ndef _hostname_hash() -> str:\n    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16]\n\n\ndef software_stack_fingerprint(\n    hardware: HardwareProfile,\n    *,\n    runtime: str,\n    runtime_version: str | None,\n    provider: str | None,\n    provider_version: str | None,\n) -> str:\n    payload = {\n        "platform": hardware.platform,\n        "os": hardware.os_name,\n        "architecture": hardware.architecture,\n        "driver": hardware.driver,\n        "jetpack": hardware.jetpack,\n        "power_mode": hardware.power_mode,\n        "software_stack": list(hardware.software_stack),\n        "runtime": runtime,\n        "runtime_version": runtime_version,\n        "provider": provider,\n        "provider_version": provider_version,\n    }\n    return hashlib.sha256(\n        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()\n    ).hexdigest()\n''',
)

replace_once(
    "src/autonomyfit/benchmark.py",
    '''    hardware_dict = {\n        "id": hardware_evidence_id(hardware),\n        "platform": hardware.platform,\n        "device": hardware.gpu,\n        "cpu": hardware.cpu,\n        "ram_total_gb": hardware.ram_total_gb,\n        "os": hardware.os_name,\n        "architecture": hardware.architecture,\n        "driver": hardware.driver,\n        "power_mode": hardware.power_mode,\n        "clocks": {},\n        "thermal_c": read_thermal_c(hardware.platform),\n    }\n''',
    '''    hardware_dict = {\n        "id": hardware_evidence_id(hardware),\n        "source": "profile" if hardware.os_name == "profile" else "detected",\n        "profile_id": hardware.matched_profile,\n        "platform": hardware.platform,\n        "device": hardware.gpu,\n        "cpu": hardware.cpu,\n        "ram_total_gb": hardware.ram_total_gb,\n        "os": hardware.os_name,\n        "architecture": hardware.architecture,\n        "driver": hardware.driver,\n        "jetpack": hardware.jetpack,\n        "power_mode": hardware.power_mode,\n        "software_stack": list(hardware.software_stack),\n        "clocks": {},\n        "thermal_c": read_thermal_c(hardware.platform),\n    }\n''',
)

replace_once(
    "src/autonomyfit/benchmark.py",
    '''    fingerprint = environment_fingerprint(\n        hardware=hardware_dict, software=software_dict, execution=execution\n    )\n''',
    '''    fingerprint = environment_fingerprint(\n        hardware=hardware_dict, software=software_dict, execution=execution\n    )\n    stack_fingerprint = software_stack_fingerprint(\n        hardware,\n        runtime=runtime,\n        runtime_version=runtime_version,\n        provider=provider,\n        provider_version=provider_version,\n    )\n''',
)

replace_once(
    "src/autonomyfit/benchmark.py",
    '''        "quality": "local-measured",\n        "notes": notes,\n''',
    '''        "quality": "local-measured",\n        "measurement": {\n            "machine_source": "profile" if hardware.os_name == "profile" else "detected",\n            "profile_only": hardware.os_name == "profile",\n            "artifact_identity_verified": True,\n        },\n        "notes": notes,\n''',
)

replace_once(
    "src/autonomyfit/benchmark.py",
    '''            "hostname_hash": _hostname_hash(),\n            "environment_fingerprint": fingerprint,\n''',
    '''            "hostname_hash": _hostname_hash(),\n            "environment_fingerprint": fingerprint,\n            "software_stack_fingerprint": stack_fingerprint,\n''',
)


# ---------------------------------------------------------------------------
# Hardware/runtime detection hardening
# ---------------------------------------------------------------------------
replace_once(
    "src/autonomyfit/hardware.py",
    '''    capabilities.append(\n        RuntimeCapability(\n            "onnxruntime",\n            bool(ort_version),\n            ort_version,\n            ", ".join(providers) if providers else None,\n            provider="CPUExecutionProvider" if ort_version else None,\n            verified=True,\n        )\n    )\n''',
    '''    primary_provider = (\n        "CPUExecutionProvider"\n        if "CPUExecutionProvider" in providers\n        else (providers[0] if providers else None)\n    )\n    capabilities.append(\n        RuntimeCapability(\n            "onnxruntime",\n            bool(ort_version and providers),\n            ort_version,\n            ", ".join(providers) if providers else ("installed but no providers exposed" if ort_version else None),\n            provider=primary_provider,\n            verified=bool(ort_version and providers),\n        )\n    )\n''',
)

replace_once(
    "src/autonomyfit/hardware.py",
    '''def _jetpack_version() -> str | None:\n    release = Path("/etc/nv_tegra_release")\n    if release.exists():\n        text = release.read_text(encoding="utf-8", errors="ignore")\n        match = re.search(r"R(\\d+)\\s*\\(release\\).*REVISION:\\s*([\\d.]+)", text)\n        if match:\n            return f"L4T R{match.group(1)}.{match.group(2)}"\n        return text.splitlines()[0].strip()\n    return None\n''',
    '''def _jetpack_version() -> str | None:\n    jetpack = None\n    if shutil.which("dpkg-query"):\n        jetpack = _run(\n            ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],\n            timeout=2.0,\n        )\n    l4t = None\n    release = Path("/etc/nv_tegra_release")\n    if release.exists():\n        text = release.read_text(encoding="utf-8", errors="ignore")\n        match = re.search(r"R(\\d+)\\s*\\(release\\).*REVISION:\\s*([\\d.]+)", text)\n        l4t = f"L4T R{match.group(1)}.{match.group(2)}" if match else text.splitlines()[0].strip()\n    if jetpack and l4t:\n        return f"JetPack {jetpack}; {l4t}"\n    if jetpack:\n        return f"JetPack {jetpack}"\n    return l4t\n''',
)

replace_once(
    "src/autonomyfit/hardware.py",
    '''    for line in text.splitlines():\n        stripped = line.strip()\n        if stripped and not stripped.startswith("NV Power Mode") and not stripped.isdigit():\n            return stripped\n    return text.splitlines()[0].strip()\n''',
    '''    for line in text.splitlines():\n        stripped = line.strip()\n        if stripped.casefold().startswith("nv power mode") and ":" in stripped:\n            return stripped.split(":", 1)[1].strip() or stripped\n    for line in text.splitlines():\n        stripped = line.strip()\n        if stripped and not stripped.isdigit():\n            return stripped\n    return None\n''',
)

replace_once(
    "src/autonomyfit/hardware.py",
    '''    openvino_version = _package_version("openvino")\n    benchmark_app = shutil.which("benchmark_app")\n    capabilities.append(\n        RuntimeCapability(\n            "openvino",\n            bool(openvino_version or benchmark_app),\n            openvino_version,\n            "native OpenVINO runtime",\n        )\n    )\n''',
    '''    openvino_version = _package_version("openvino")\n    benchmark_app = shutil.which("benchmark_app")\n    ov_devices = _openvino_devices()\n    ov_detail = "native OpenVINO runtime"\n    if ov_devices:\n        ov_detail += "; devices=" + ",".join(ov_devices)\n    capabilities.append(\n        RuntimeCapability(\n            "openvino",\n            bool(openvino_version or benchmark_app),\n            openvino_version,\n            ov_detail,\n        )\n    )\n''',
)


# ---------------------------------------------------------------------------
# Backend reproducibility and exact native input/device handling
# ---------------------------------------------------------------------------
replace_once(
    "src/autonomyfit/backends.py",
    "from dataclasses import dataclass, field\n",
    "from dataclasses import dataclass, field, replace\n",
)
replace_once(
    "src/autonomyfit/backends.py",
    '''    expected_sha256: str | None = None\n''',
    '''    expected_sha256: str | None = None\n    compute_units: str | None = None\n''',
)

insert_marker = '''class OnnxRuntimeBackend(BenchmarkBackend):\n'''
helper = '''def _infer_batch_size(\n    explicit: int | None, input_shapes: dict[str, list[int]]\n) -> int | None:\n    if explicit is not None:\n        return explicit\n    first_dims = {dims[0] for dims in input_shapes.values() if dims}\n    return next(iter(first_dims)) if len(first_dims) == 1 else None\n\n\ndef _resolve_onnx_input_shapes(request: BenchmarkRequest) -> dict[str, list[int]]:\n    if request.input_shapes:\n        return dict(request.input_shapes)\n    try:\n        import onnx\n    except ImportError as exc:\n        raise BackendError(\n            "native ONNX shape discovery requires the benchmark extra with onnx installed"\n        ) from exc\n    try:\n        model = onnx.load(str(request.model_path), load_external_data=False)\n    except Exception as exc:  # noqa: BLE001\n        raise BackendError(f"could not inspect ONNX inputs: {exc}") from exc\n    initializer_names = {item.name for item in model.graph.initializer}\n    inputs = [item for item in model.graph.input if item.name not in initializer_names]\n    if request.shape_override is not None:\n        if len(inputs) != 1:\n            raise BackendError(\n                "--shape is supported only for single-input ONNX models; provide named input shapes for multi-input graphs"\n            )\n        return {inputs[0].name: list(request.shape_override)}\n    resolved: dict[str, list[int]] = {}\n    for item in inputs:\n        dims: list[int] = []\n        tensor_type = item.type.tensor_type\n        for index, dim in enumerate(tensor_type.shape.dim):\n            if dim.HasField("dim_value") and dim.dim_value > 0:\n                dims.append(int(dim.dim_value))\n            elif index == 0:\n                dims.append(request.batch_size or 1)\n            else:\n                raise BackendError(\n                    f"ONNX input {item.name!r} has a dynamic non-batch dimension; use --shape for a single-input graph"\n                )\n        if not dims:\n            raise BackendError(f"ONNX input {item.name!r} has no resolvable shape")\n        resolved[item.name] = dims\n    if not resolved:\n        raise BackendError("ONNX graph exposes no benchmarkable inputs")\n    return resolved\n\n\n'''
replace_once("src/autonomyfit/backends.py", insert_marker, helper + insert_marker)

replace_once(
    "src/autonomyfit/backends.py",
    '''        memory = MemorySampler()\n        memory.start()\n        started = time.perf_counter_ns()\n        try:\n            session = ort.InferenceSession(str(request.model_path), providers=[provider])\n''',
    '''        memory = MemorySampler()\n        memory.start()\n        started = time.perf_counter_ns()\n        try:\n            session_options = ort.SessionOptions()\n            if provider != "CPUExecutionProvider":\n                try:\n                    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")\n                except Exception as exc:  # noqa: BLE001\n                    raise BackendError(\n                        "could not disable ONNX Runtime CPU fallback for an accelerator provider"\n                    ) from exc\n            session = ort.InferenceSession(\n                str(request.model_path), sess_options=session_options, providers=[provider]\n            )\n            active_providers = session.get_providers()\n            if not active_providers or active_providers[0] != provider:\n                raise BackendError(\n                    f"requested provider {provider!r} was not the active primary provider: {active_providers}"\n                )\n''',
)

replace_once(
    "src/autonomyfit/backends.py",
    '''            provider_version=ort.__version__,\n            precision=request.precision,\n            quantization=request.quantization,\n            batch_size=request.batch_size,\n            input_shapes=input_shapes,\n''',
    '''            provider_version=f"onnxruntime-{ort.__version__}",\n            precision=request.precision,\n            quantization=request.quantization,\n            batch_size=_infer_batch_size(request.batch_size, input_shapes),\n            input_shapes=input_shapes,\n''',
)

replace_once(
    "src/autonomyfit/backends.py",
    '''            backend_options={"providers": [provider], "synthetic_inputs": True},\n''',
    '''            backend_options={\n                "requested_provider": provider,\n                "active_providers": active_providers,\n                "provider_options": session.get_provider_options().get(provider, {}),\n                "ort_build_info": getattr(ort, "get_build_info", lambda: None)(),\n                "cpu_fallback_disabled": provider != "CPUExecutionProvider",\n                "synthetic_inputs": True,\n            },\n''',
)

replace_once(
    "src/autonomyfit/backends.py",
    '''        command = build_trtexec_command(request, availability.executable)\n''',
    '''        native_request = request\n        if request.model_path.suffix.casefold() == ".onnx" and not request.input_shapes:\n            native_request = replace(request, input_shapes=_resolve_onnx_input_shapes(request))\n        command = build_trtexec_command(native_request, availability.executable)\n''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''            batch_size=request.batch_size,\n            input_shapes=request.input_shapes,\n            warmup=request.warmup,\n''',
    '''            batch_size=_infer_batch_size(native_request.batch_size, native_request.input_shapes),\n            input_shapes=native_request.input_shapes,\n            warmup=request.warmup,\n''',
)

replace_once(
    "src/autonomyfit/backends.py",
    '''        command = build_openvino_command(request, availability.executable)\n        started = time.perf_counter_ns()\n''',
    '''        native_request = request\n        if request.model_path.suffix.casefold() == ".onnx" and not request.input_shapes:\n            native_request = replace(request, input_shapes=_resolve_onnx_input_shapes(request))\n        device = request.device or "CPU"\n        try:\n            import openvino as ov\n            available_devices = tuple(ov.Core().available_devices)\n        except Exception:  # noqa: BLE001\n            available_devices = ()\n        if available_devices and device not in available_devices:\n            raise BackendError(\n                f"OpenVINO device {device!r} unavailable. Available: {', '.join(available_devices)}"\n            )\n        native_request = replace(native_request, device=device)\n        command = build_openvino_command(native_request, availability.executable)\n        started = time.perf_counter_ns()\n''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''            provider=request.device or "AUTO",\n            provider_version=availability.version,\n            precision=request.precision,\n            quantization=request.quantization,\n            batch_size=request.batch_size,\n            input_shapes=request.input_shapes,\n''',
    '''            provider=device,\n            provider_version=availability.version,\n            precision=request.precision,\n            quantization=request.quantization,\n            batch_size=_infer_batch_size(native_request.batch_size, native_request.input_shapes),\n            input_shapes=native_request.input_shapes,\n''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''            backend_options={"performance_hint": "latency", "native_warmup": True, "wall_ms": wall_ms},\n''',
    '''            backend_options={\n                "performance_hint": "latency",\n                "native_warmup": True,\n                "device": device,\n                "available_devices": list(available_devices),\n                "wall_ms": wall_ms,\n            },\n''',
)

replace_once(
    "src/autonomyfit/backends.py",
    '''            model = ct.models.MLModel(str(request.model_path))\n''',
    '''            compute_name = (request.compute_units or "ALL").strip().upper()\n            compute_map = {\n                "ALL": ct.ComputeUnit.ALL,\n                "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,\n                "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,\n                "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,\n            }\n            if compute_name not in compute_map:\n                raise BackendError(\n                    "Core ML compute units must be ALL, CPU_ONLY, CPU_AND_GPU or CPU_AND_NE"\n                )\n            model = ct.models.MLModel(\n                str(request.model_path), compute_units=compute_map[compute_name]\n            )\n''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''            batch_size=request.batch_size, input_shapes=input_shapes,\n''',
    '''            batch_size=_infer_batch_size(request.batch_size, input_shapes), input_shapes=input_shapes,\n''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''            backend_options={"compute_units":"model-default","synthetic_inputs":True},\n''',
    '''            backend_options={"compute_units":compute_name,"synthetic_inputs":True},\n''',
)

replace_once(
    "src/autonomyfit/backends.py",
    '''def run_benchmark(request: BenchmarkRequest, backend_name: str | None = None) -> dict[str, Any]:\n    try:\n''',
    '''def run_benchmark(request: BenchmarkRequest, backend_name: str | None = None) -> dict[str, Any]:\n    if request.hardware.os_name == "profile":\n        raise BackendError(\n            "local benchmark evidence requires detected hardware; bundled profiles are screening targets only"\n        )\n    try:\n''',
)


# ---------------------------------------------------------------------------
# Evidence model and strict matrix-context matching
# ---------------------------------------------------------------------------
replace_once(
    "src/autonomyfit/evidence.py",
    '''    software_stack_id: str | None\n    notes: str | None = None\n    verified_identity: bool = False\n''',
    '''    software_stack_id: str | None\n    provider_version: str | None = None\n    notes: str | None = None\n    verified_identity: bool = False\n''',
)

replace_once(
    "src/autonomyfit/evidence.py",
    '''            and bool(self.model_revision)\n            and bool(self.artifact_sha256)\n        )\n''',
    '''            and bool(self.model_revision)\n            and bool(self.artifact_sha256)\n            and bool(self.runtime_version)\n            and bool(self.provider)\n            and bool(self.provider_version)\n            and self.precision.casefold() != "artifact"\n            and self.batch_size is not None\n            and bool(self.input_shapes)\n            and bool(self.software_stack_id)\n        )\n''',
)

replace_once(
    "src/autonomyfit/evidence.py",
    '''                software_stack_id=item.get("software_stack_id"),\n                notes=item.get("notes"),\n''',
    '''                software_stack_id=item.get("software_stack_id"),\n                provider_version=runtime.get("version"),\n                notes=item.get("notes"),\n''',
)

replace_once(
    "src/autonomyfit/evidence.py",
    '''        software_stack_id=None,\n        notes=document.get("notes"),\n''',
    '''        software_stack_id=(document.get("reproducibility") or {}).get("software_stack_fingerprint"),\n        provider_version=software.get("provider_version"),\n        notes=document.get("notes"),\n''',
)

replace_once(
    "src/autonomyfit/evidence.py",
    '''    provider: str | None = None,\n    max_age_days: int = 730,\n''',
    '''    provider: str | None = None,\n    provider_version: str | None = None,\n    quantization: str | None = None,\n    batch_size: int | None = None,\n    input_shapes: dict[str, list[int]] | None = None,\n    power_mode: str | None = None,\n    software_stack_id: str | None = None,\n    max_age_days: int = 730,\n''',
)

replace_once(
    "src/autonomyfit/evidence.py",
    '''        if provider and evidence.provider and evidence.provider.casefold() != provider.casefold():\n            continue\n\n        reasons: list[str] = []\n        identity_complete = True\n''',
    '''        if provider and evidence.provider and evidence.provider.casefold() != provider.casefold():\n            continue\n        if (\n            provider_version\n            and evidence.provider_version\n            and evidence.provider_version.casefold() != provider_version.casefold()\n        ):\n            continue\n        if (quantization is not None or evidence.quantization is not None) and (\n            (quantization or "").casefold() != (evidence.quantization or "").casefold()\n        ):\n            continue\n        if batch_size is not None and evidence.batch_size is not None and evidence.batch_size != batch_size:\n            continue\n        if input_shapes and evidence.input_shapes and evidence.input_shapes != input_shapes:\n            continue\n        if (power_mode is not None or evidence.power_mode is not None) and (\n            (power_mode or "").casefold() != (evidence.power_mode or "").casefold()\n        ):\n            continue\n        if (\n            software_stack_id\n            and evidence.software_stack_id\n            and evidence.software_stack_id.casefold() != software_stack_id.casefold()\n        ):\n            continue\n\n        reasons: list[str] = []\n        identity_complete = True\n''',
)

replace_once(
    "src/autonomyfit/evidence.py",
    '''        if provider and not evidence.provider:\n            identity_complete = False\n            reasons.append("target execution provider is pinned but evidence provider is unknown")\n        if _date_is_stale(evidence.source_date, max_age_days, today=today):\n''',
    '''        if provider and not evidence.provider:\n            identity_complete = False\n            reasons.append("target execution provider is pinned but evidence provider is unknown")\n        strict_context = evidence.quality in {"local-measured", "standardized"}\n        if strict_context:\n            if runtime_version is None or evidence.runtime_version is None:\n                identity_complete = False\n                reasons.append("runtime version is not pinned on both sides")\n            if provider is None or evidence.provider is None:\n                identity_complete = False\n                reasons.append("execution provider is not pinned on both sides")\n            if provider_version is None or evidence.provider_version is None:\n                identity_complete = False\n                reasons.append("provider version is not pinned on both sides")\n            if batch_size is None or evidence.batch_size is None:\n                identity_complete = False\n                reasons.append("batch size is not pinned on both sides")\n            if not input_shapes or not evidence.input_shapes:\n                identity_complete = False\n                reasons.append("input shapes are not pinned on both sides")\n            if software_stack_id is None or evidence.software_stack_id is None:\n                identity_complete = False\n                reasons.append("software stack is not pinned on both sides")\n        if _date_is_stale(evidence.source_date, max_age_days, today=today):\n''',
)

replace_once(
    "src/autonomyfit/evidence.py",
    '''        "provider": evidence.provider,\n        "precision": evidence.precision,\n''',
    '''        "provider": evidence.provider,\n        "provider_version": evidence.provider_version,\n        "precision": evidence.precision,\n''',
)


# ---------------------------------------------------------------------------
# Constraint and scoring context binding
# ---------------------------------------------------------------------------
replace_once(
    "src/autonomyfit/models.py",
    '''    include_experimental: bool = False\n''',
    '''    include_experimental: bool = False\n    provider: str | None = None\n    provider_version: str | None = None\n    quantization: str | None = None\n    batch_size: int | None = None\n    input_shapes: dict[str, list[int]] = field(default_factory=dict)\n    power_mode: str | None = None\n    software_stack_id: str | None = None\n''',
)

replace_once(
    "src/autonomyfit/scoring.py",
    '''    artifact_sha256: str | None,\n) -> EvidenceMatch | None:\n    provider = _BRIDGE_RUNTIMES.get(runtime)\n''',
    '''    artifact_sha256: str | None,\n    provider_override: str | None = None,\n    provider_version: str | None = None,\n    quantization: str | None = None,\n    batch_size: int | None = None,\n    input_shapes: dict[str, list[int]] | None = None,\n    power_mode: str | None = None,\n    software_stack_id: str | None = None,\n) -> EvidenceMatch | None:\n    provider = provider_override or _BRIDGE_RUNTIMES.get(runtime)\n''',
)

replace_once(
    "src/autonomyfit/scoring.py",
    '''            runtime_version=_runtime_version(hardware, runtime),\n            provider=provider,\n        )\n''',
    '''            runtime_version=_runtime_version(hardware, runtime),\n            provider=provider,\n            provider_version=provider_version,\n            quantization=quantization,\n            batch_size=batch_size,\n            input_shapes=input_shapes,\n            power_mode=power_mode,\n            software_stack_id=software_stack_id,\n        )\n''',
)

replace_once(
    "src/autonomyfit/scoring.py",
    '''            precision=precision,\n            artifact_sha256=artifact_sha,\n        )\n''',
    '''            precision=precision,\n            artifact_sha256=artifact_sha,\n            provider_override=constraints.provider,\n            provider_version=constraints.provider_version,\n            quantization=constraints.quantization,\n            batch_size=constraints.batch_size,\n            input_shapes=constraints.input_shapes,\n            power_mode=constraints.power_mode,\n            software_stack_id=constraints.software_stack_id,\n        )\n''',
)

# Prefer exact detected-machine local evidence, then permit matched-profile evidence as context.
old_select_return = '''        if matches:\n            return matches[0]\n    return None\n'''
new_select_return = '''        if matches:\n            return matches[0]\n        if hardware.os_name != "profile" and hardware.matched_profile:\n            profile_matches = match_benchmarks(\n                store.benchmarks,\n                model_id=model.id,\n                model_revision=model_revision,\n                hardware_id=hardware.matched_profile,\n                runtime=runtime_alias,\n                precision=precision,\n                artifact_sha256=artifact_sha256,\n                runtime_version=_runtime_version(hardware, runtime),\n                provider=provider,\n            )\n            if profile_matches:\n                contextual = profile_matches[0]\n                return EvidenceMatch(\n                    evidence=contextual.evidence,\n                    exact=False,\n                    identity_complete=False,\n                    reasons=("profile-level evidence is contextual for this detected machine", *contextual.reasons),\n                )\n    return None\n'''
replace_once("src/autonomyfit/scoring.py", old_select_return, new_select_return)


# ---------------------------------------------------------------------------
# Deployment: bind recommendation to exact benchmark context
# ---------------------------------------------------------------------------
replace_once(
    "src/autonomyfit/deployment.py",
    '''    options: ValidationOptions,\n) -> dict[str, Any] | None:\n    constraints = Constraints(\n''',
    '''    options: ValidationOptions,\n    benchmark: dict[str, Any],\n) -> dict[str, Any] | None:\n    software = benchmark.get("software") or {}\n    execution = benchmark.get("execution") or {}\n    benchmark_hardware = benchmark.get("hardware") or {}\n    reproducibility = benchmark.get("reproducibility") or {}\n    constraints = Constraints(\n''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''        include_experimental=True,\n    )\n''',
    '''        include_experimental=True,\n        provider=software.get("provider"),\n        provider_version=software.get("provider_version"),\n        quantization=execution.get("quantization"),\n        batch_size=execution.get("batch_size"),\n        input_shapes=execution.get("input_shapes") or {},\n        power_mode=benchmark_hardware.get("power_mode"),\n        software_stack_id=reproducibility.get("software_stack_fingerprint"),\n    )\n''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''                    precision=precision,\n                    options=options,\n                )\n''',
    '''                    precision=precision,\n                    options=options,\n                    benchmark=benchmark_report,\n                )\n''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''    hardware = _resolve_hardware(hardware_profile)\n    loaded = load_model_catalog(offline=offline)\n''',
    '''    hardware = detect_hardware()\n    if hardware_profile and hardware.matched_profile != hardware_profile:\n        raise DeploymentValidationError(\n            "candidate assessment measured the current machine but the requested hardware profile no longer matches it"\n        )\n    loaded = load_model_catalog(offline=offline)\n''',
)

# Bind candidate reranking to every measured execution dimension.
replace_once(
    "src/autonomyfit/deployment.py",
    '''        artifact = report.get("artifact") or {}\n        exact = recommend_models(\n''',
    '''        artifact = report.get("artifact") or {}\n        benchmark = report.get("benchmark") or {}\n        software = benchmark.get("software") or {}\n        execution = benchmark.get("execution") or {}\n        benchmark_hardware = benchmark.get("hardware") or {}\n        reproducibility = benchmark.get("reproducibility") or {}\n        exact = recommend_models(\n''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''                precision=precision,\n                include_experimental=True,\n            ),\n''',
    '''                precision=precision,\n                include_experimental=True,\n                provider=software.get("provider"),\n                provider_version=software.get("provider_version"),\n                quantization=execution.get("quantization"),\n                batch_size=execution.get("batch_size"),\n                input_shapes=execution.get("input_shapes") or {},\n                power_mode=benchmark_hardware.get("power_mode"),\n                software_stack_id=reproducibility.get("software_stack_fingerprint"),\n            ),\n''',
)


# ---------------------------------------------------------------------------
# Local-result invalidation for power mode and stack fingerprint
# ---------------------------------------------------------------------------
replace_once(
    "src/autonomyfit/local_results.py",
    "from .benchmark import hardware_evidence_id\n",
    "from .benchmark import hardware_evidence_id, software_stack_fingerprint\n",
)
replace_once(
    "src/autonomyfit/local_results.py",
    '''    software = document.get("software") or {}\n\n    provider = software.get("provider")\n''',
    '''    report_power_mode = report_hardware.get("power_mode")\n    if report_power_mode and hardware.power_mode and str(report_power_mode) != hardware.power_mode:\n        reasons.append(\n            f"power mode changed ({report_power_mode} -> {hardware.power_mode})"\n        )\n\n    software = document.get("software") or {}\n\n    provider = software.get("provider")\n''',
)
replace_once(
    "src/autonomyfit/local_results.py",
    '''    if report_runtime_version and current_runtime_version:\n        report_major = _major(str(report_runtime_version))\n        current_major = _major(current_runtime_version)\n        if report_major is not None and current_major is not None and report_major != current_major:\n            reasons.append(\n                "runtime major version changed "\n                f"({report_runtime_version} -> {current_runtime_version})"\n            )\n\n    return not reasons, tuple(reasons)\n''',
    '''    if report_runtime_version and current_runtime_version:\n        report_major = _major(str(report_runtime_version))\n        current_major = _major(current_runtime_version)\n        if report_major is not None and current_major is not None and report_major != current_major:\n            reasons.append(\n                "runtime major version changed "\n                f"({report_runtime_version} -> {current_runtime_version})"\n            )\n\n    recorded_stack = (document.get("reproducibility") or {}).get(\n        "software_stack_fingerprint"\n    )\n    if recorded_stack and runtime and current_runtime_version:\n        current_stack = software_stack_fingerprint(\n            hardware,\n            runtime=str(runtime),\n            runtime_version=current_runtime_version,\n            provider=str(provider) if provider else None,\n            provider_version=current_runtime_version if provider else None,\n        )\n        if recorded_stack != current_stack:\n            reasons.append("material software stack fingerprint changed")\n\n    return not reasons, tuple(reasons)\n''',
)


# ---------------------------------------------------------------------------
# Benchmark matrix tooling
# ---------------------------------------------------------------------------
matrix_path = ROOT / "src/autonomyfit/benchmark_matrix.py"
matrix_path.write_text('''from __future__ import annotations\n\nimport hashlib\nimport json\nfrom typing import Any\n\nfrom .evidence import BenchmarkEvidence, EvidenceStore, load_evidence_store\n\n\ndef matrix_key(evidence: BenchmarkEvidence) -> str:\n    payload = {\n        "model": evidence.model_id,\n        "revision": evidence.model_revision,\n        "artifact_sha256": evidence.artifact_sha256,\n        "hardware": evidence.hardware_id,\n        "runtime": evidence.runtime,\n        "runtime_version": evidence.runtime_version,\n        "provider": evidence.provider,\n        "provider_version": evidence.provider_version,\n        "precision": evidence.precision,\n        "quantization": evidence.quantization,\n        "batch_size": evidence.batch_size,\n        "input_shapes": evidence.input_shapes,\n        "power_mode": evidence.power_mode,\n        "software_stack_id": evidence.software_stack_id,\n    }\n    digest = hashlib.sha256(\n        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()\n    ).hexdigest()[:24]\n    return f"matrix-{digest}"\n\n\ndef _row(evidence: BenchmarkEvidence) -> dict[str, Any]:\n    return {\n        "matrix_key": matrix_key(evidence),\n        "benchmark_id": evidence.id,\n        "model_id": evidence.model_id,\n        "model_revision": evidence.model_revision,\n        "artifact_sha256": evidence.artifact_sha256,\n        "hardware_id": evidence.hardware_id,\n        "runtime": evidence.runtime,\n        "runtime_version": evidence.runtime_version,\n        "provider": evidence.provider,\n        "provider_version": evidence.provider_version,\n        "precision": evidence.precision,\n        "quantization": evidence.quantization,\n        "batch_size": evidence.batch_size,\n        "input_shapes": evidence.input_shapes,\n        "power_mode": evidence.power_mode,\n        "software_stack_id": evidence.software_stack_id,\n        "quality": evidence.quality,\n        "verified_identity": evidence.verified_identity,\n        "exact_context_complete": evidence.eligible_for_verified_fit,\n        "latency_ms": evidence.latency_ms,\n        "throughput_fps": evidence.fps,\n        "power_mean_w": evidence.power.mean_w,\n        "power_scope": evidence.power.scope,\n        "peak_memory_mb": evidence.peak_memory_mb,\n        "source_date": evidence.source_date,\n    }\n\n\ndef benchmark_matrix(\n    *,\n    store: EvidenceStore | None = None,\n    local_only: bool = False,\n    model_id: str | None = None,\n    hardware_id: str | None = None,\n) -> dict[str, Any]:\n    evidence_store = store or load_evidence_store(include_local=True)\n    items = list(evidence_store.benchmarks)\n    if local_only:\n        items = [item for item in items if item.quality == "local-measured"]\n    if model_id:\n        items = [item for item in items if item.model_id.casefold() == model_id.casefold()]\n    if hardware_id:\n        items = [item for item in items if item.hardware_id.casefold() == hardware_id.casefold()]\n    rows = [_row(item) for item in items]\n    rows.sort(key=lambda item: (item["model_id"], item["hardware_id"], item["runtime"], item["matrix_key"]))\n    return {\n        "record_count": len(rows),\n        "exact_context_complete": sum(bool(row["exact_context_complete"]) for row in rows),\n        "local_measured": sum(row["quality"] == "local-measured" for row in rows),\n        "vendor_published": sum(row["quality"] == "vendor-published" for row in rows),\n        "standardized": sum(row["quality"] == "standardized" for row in rows),\n        "matrix": rows,\n    }\n''', encoding="utf-8")

replace_once(
    "src/autonomyfit/cli.py",
    "from .benchmark import save_result\n",
    "from .benchmark import save_result\nfrom .benchmark_matrix import benchmark_matrix\n",
)
replace_once(
    "src/autonomyfit/cli.py",
    '''    trust_artifact: Annotated[\n        bool,\n        typer.Option(\n            "--trust-artifact",\n            help="Explicitly trust a serialized executable artifact such as a TensorRT engine you control.",\n        ),\n    ] = False,\n) -> None:\n''',
    '''    trust_artifact: Annotated[\n        bool,\n        typer.Option(\n            "--trust-artifact",\n            help="Explicitly trust a serialized executable artifact such as a TensorRT engine you control.",\n        ),\n    ] = False,\n    compute_units: Annotated[\n        str | None,\n        typer.Option("--compute-units", help="Core ML compute units: ALL, CPU_ONLY, CPU_AND_GPU or CPU_AND_NE."),\n    ] = None,\n) -> None:\n''',
)
replace_once(
    "src/autonomyfit/cli.py",
    '''        trusted_artifact=trust_artifact,\n    )\n''',
    '''        trusted_artifact=trust_artifact,\n        compute_units=compute_units,\n        command=(\n            f"autonomyfit benchmark {model} --model-id {resolved_id} "\n            f"--model-revision {resolved_revision or 'unknown'} "\n            f"--backend {backend or 'auto'} --precision {precision.strip().casefold()}"\n        ),\n    )\n''',
)

matrix_command = '''@app.command("benchmark-matrix")\ndef benchmark_matrix_command(\n    local_only: Annotated[bool, typer.Option("--local-only", help="Show only locally measured evidence.")] = False,\n    model_id: Annotated[str | None, typer.Option("--model-id", help="Filter one model ID.")] = None,\n    hardware_id: Annotated[str | None, typer.Option("--hardware-id", help="Filter one exact hardware ID.")] = None,\n    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,\n) -> None:\n    \"\"\"Inspect the exact model/artifact/hardware/runtime benchmark evidence matrix.\"\"\"\n    payload = benchmark_matrix(\n        local_only=local_only, model_id=model_id, hardware_id=hardware_id\n    )\n    if json_output:\n        console.print_json(json.dumps(payload))\n        return\n    table = Table(title="Benchmark evidence matrix")\n    table.add_column("Model")\n    table.add_column("Hardware")\n    table.add_column("Runtime/provider")\n    table.add_column("Precision")\n    table.add_column("Batch/shape")\n    table.add_column("Quality")\n    table.add_column("Exact context")\n    for item in payload["matrix"]:\n        shapes = ";".join(\n            f"{name}={','.join(str(dim) for dim in dims)}"\n            for name, dims in sorted(item["input_shapes"].items())\n        ) or "unknown"\n        table.add_row(\n            str(item["model_id"]),\n            str(item["hardware_id"]),\n            f"{item['runtime']}/{item.get('provider') or '-'}",\n            str(item["precision"]),\n            f"{item.get('batch_size') or '?'} / {shapes}",\n            str(item["quality"]),\n            "yes" if item["exact_context_complete"] else "no",\n        )\n    console.print(table)\n    console.print(\n        f"{payload['record_count']} records; {payload['exact_context_complete']} exact-context complete"\n    )\n'''
replace_once(
    "src/autonomyfit/cli.py",
    '\n\nregister_deployment_commands(app, console)\n',
    '\n\n' + matrix_command + '\n\nregister_deployment_commands(app, console)\n',
)


# ---------------------------------------------------------------------------
# Optional backwards-compatible report schema fields
# ---------------------------------------------------------------------------
import json
schema_path = ROOT / "src/autonomyfit/data/benchmark-report-v2.schema.json"
schema = json.loads(schema_path.read_text(encoding="utf-8"))
schema["properties"]["measurement"] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["machine_source", "profile_only", "artifact_identity_verified"],
    "properties": {
        "machine_source": {"enum": ["detected", "profile"]},
        "profile_only": {"type": "boolean"},
        "artifact_identity_verified": {"type": "boolean"},
    },
}
hw = schema["properties"]["hardware"]["properties"]
hw["source"] = {"enum": ["detected", "profile"]}
hw["profile_id"] = {"type": ["string", "null"]}
hw["jetpack"] = {"type": ["string", "null"]}
hw["software_stack"] = {"type": "array", "items": {"type": "string"}}
schema["properties"]["reproducibility"]["properties"]["software_stack_fingerprint"] = {
    "type": "string",
    "pattern": "^[a-fA-F0-9]{64}$",
}
schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

# Include ONNX structural metadata support in the benchmark extra.
replace_once(
    "pyproject.toml",
    '''benchmark = [\n  "numpy>=1.26",\n  "onnxruntime>=1.19",\n]\n''',
    '''benchmark = [\n  "numpy>=1.26",\n  "onnx>=1.16",\n  "onnxruntime>=1.19",\n]\n''',
)

print("AutonomyFit benchmark evidence core upgrade applied")

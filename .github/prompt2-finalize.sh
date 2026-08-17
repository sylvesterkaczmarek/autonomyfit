#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


# Evidence classification and artifact-identity trust.
replace_once(
    "src/autonomyfit/evidence.py",
    '''    measurement = document.get("measurement")
    if measurement is not None and (
        measurement.get("machine_source") != "detected" or measurement.get("profile_only")
    ):
        raise EvidenceSchemaError(
            "local benchmark reports cannot claim measurements from a profile-only hardware target"
        )
''',
    '''    measurement = document.get("measurement")
    if measurement is None:
        raise EvidenceSchemaError(
            "local benchmark reports require explicit measurement classification"
        )
    if measurement.get("machine_source") != "detected" or measurement.get("profile_only"):
        raise EvidenceSchemaError(
            "local benchmark reports cannot claim measurements from a profile-only hardware target"
        )
''',
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''    software = document["software"]
    execution = document["execution"]
    metrics = document["metrics"]
    return BenchmarkEvidence(
''',
    '''    software = document["software"]
    execution = document["execution"]
    metrics = document["metrics"]
    measurement = document["measurement"]
    return BenchmarkEvidence(
''',
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''        provider_version=software.get("provider_version"),
        machine_source=(document.get("measurement") or {}).get("machine_source"),
        notes=document.get("notes"),
        verified_identity=bool(model.get("revision") and artifact.get("sha256")),
''',
    '''        provider_version=software.get("provider_version"),
        machine_source=measurement.get("machine_source"),
        notes=document.get("notes"),
        verified_identity=bool(
            measurement.get("artifact_identity_verified")
            and model.get("revision")
            and artifact.get("sha256")
        ),
''',
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''        "quality": evidence.quality,
        "identity_complete": evidence.eligible_for_verified_fit,
''',
    '''        "quality": evidence.quality,
        "machine_source": evidence.machine_source,
        "verified_identity": evidence.verified_identity,
        "identity_complete": evidence.eligible_for_verified_fit,
''',
)
replace_once(
    "src/autonomyfit/data/benchmark-report-v2.schema.json",
    '''    "metrics",
    "reproducibility"
''',
    '''    "metrics",
    "measurement",
    "reproducibility"
''',
)

# Local evidence invalidation must fail closed on native runtime/driver/power-state loss.
replace_once(
    "src/autonomyfit/local_results.py",
    '''def _runtime_capability_version(hardware: HardwareProfile, runtime: str | None) -> str | None:
    if not runtime:
        return None
    aliases = {"onnx": "onnxruntime", "onnxruntime": "onnxruntime"}
    target = aliases.get(runtime.casefold(), runtime.casefold())
    for capability in hardware.runtimes:
        if capability.name.casefold() == target:
            return capability.version
    return None
''',
    '''def _runtime_capability(hardware: HardwareProfile, runtime: str | None):
    if not runtime:
        return None
    aliases = {"onnx": "onnxruntime", "onnxruntime": "onnxruntime"}
    target = aliases.get(runtime.casefold(), runtime.casefold())
    return next(
        (item for item in hardware.runtimes if item.name.casefold() == target),
        None,
    )


def _runtime_capability_version(hardware: HardwareProfile, runtime: str | None) -> str | None:
    capability = _runtime_capability(hardware, runtime)
    return capability.version if capability else None
''',
)
replace_once(
    "src/autonomyfit/local_results.py",
    '''    report_driver = report_hardware.get("driver")
    if report_driver and hardware.driver:
        report_major = _major(str(report_driver))
        current_major = _major(hardware.driver)
        if report_major is not None and current_major is not None and report_major != current_major:
            reasons.append(
                f"driver major version changed ({report_driver} -> {hardware.driver})"
            )

    report_power_mode = report_hardware.get("power_mode")
    if report_power_mode and hardware.power_mode and str(report_power_mode) != hardware.power_mode:
        reasons.append(
            f"power mode changed ({report_power_mode} -> {hardware.power_mode})"
        )
''',
    '''    report_driver = report_hardware.get("driver")
    if report_driver and not hardware.driver:
        reasons.append("current driver version could not be established")
    elif report_driver and hardware.driver:
        report_major = _major(str(report_driver))
        current_major = _major(hardware.driver)
        if report_major is not None and current_major is not None and report_major != current_major:
            reasons.append(
                f"driver major version changed ({report_driver} -> {hardware.driver})"
            )

    report_power_mode = report_hardware.get("power_mode")
    if report_power_mode and not hardware.power_mode:
        reasons.append("current power mode could not be established")
    elif report_power_mode and hardware.power_mode and str(report_power_mode) != hardware.power_mode:
        reasons.append(
            f"power mode changed ({report_power_mode} -> {hardware.power_mode})"
        )
''',
)
replace_once(
    "src/autonomyfit/local_results.py",
    '''    runtime = software.get("runtime")
    report_runtime_version = software.get("runtime_version")
    current_runtime_version = _runtime_capability_version(
        hardware, str(runtime) if runtime else None
    )
    if report_runtime_version and current_runtime_version:
''',
    '''    runtime = software.get("runtime")
    runtime_name = str(runtime) if runtime else None
    runtime_capability = _runtime_capability(hardware, runtime_name)
    if runtime_name and (runtime_capability is None or not runtime_capability.available):
        reasons.append(f"runtime is no longer available ({runtime_name})")
    report_runtime_version = software.get("runtime_version")
    current_runtime_version = _runtime_capability_version(hardware, runtime_name)
    if (
        report_runtime_version
        and runtime_capability is not None
        and runtime_capability.available
        and not current_runtime_version
    ):
        reasons.append(f"current runtime version could not be established ({runtime_name})")
    if report_runtime_version and current_runtime_version:
''',
)

# CLI support for named multi-input shapes and complete reproduction commands.
replace_once(
    "src/autonomyfit/cli.py",
    '''import json
from dataclasses import asdict
''',
    '''import json
import shlex
from dataclasses import asdict
''',
)
replace_once(
    "src/autonomyfit/cli.py",
    '''def _parse_shape(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        dims = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(
            "shape must be comma-separated positive integers, for example 1,3,640,640",
            param_hint="--shape",
        ) from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise typer.BadParameter("shape dimensions must be positive integers", param_hint="--shape")
    return dims
''',
    '''def _parse_dims(value: str, *, param_hint: str) -> list[int]:
    try:
        dims = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise typer.BadParameter(
            "shape must be comma-separated positive integers, for example 1,3,640,640",
            param_hint=param_hint,
        ) from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise typer.BadParameter(
            "shape dimensions must be positive integers", param_hint=param_hint
        )
    return dims


def _parse_shape(value: str | None) -> list[int] | None:
    return None if value is None else _parse_dims(value, param_hint="--shape")


def _parse_named_shapes(values: list[str] | None) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for raw in values or []:
        if "=" not in raw:
            raise typer.BadParameter(
                "named input shape must use NAME=1,3,640,640",
                param_hint="--input-shape",
            )
        name, raw_dims = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise typer.BadParameter(
                "named input shape requires a non-empty input name",
                param_hint="--input-shape",
            )
        if name in result:
            raise typer.BadParameter(
                f"duplicate named input shape: {name}", param_hint="--input-shape"
            )
        result[name] = _parse_dims(raw_dims, param_hint="--input-shape")
    return result
''',
)
replace_once(
    "src/autonomyfit/cli.py",
    '''    shape: Annotated[str | None, typer.Option(help="ONNX dynamic input shape, e.g. 1,3,640,640.")] = None,
    provider: Annotated[
''',
    '''    shape: Annotated[str | None, typer.Option(help="ONNX dynamic input shape, e.g. 1,3,640,640.")] = None,
    input_shape: Annotated[
        list[str] | None,
        typer.Option(
            "--input-shape",
            help="Repeatable named input shape NAME=1,3,640,640 for multi-input ONNX graphs.",
        ),
    ] = None,
    provider: Annotated[
''',
)
replace_once(
    "src/autonomyfit/cli.py",
    '''    shape_override = _parse_shape(shape)
    resolved_id = model_id or model.stem
    resolved_revision = _resolve_model_revision(resolved_id, model_revision)
    hardware = detect_hardware()
    request = BenchmarkRequest(
''',
    '''    shape_override = _parse_shape(shape)
    named_input_shapes = _parse_named_shapes(input_shape)
    if shape_override is not None and named_input_shapes:
        raise typer.BadParameter(
            "--shape and --input-shape cannot be combined", param_hint="--input-shape"
        )
    resolved_id = model_id or model.stem
    resolved_revision = _resolve_model_revision(resolved_id, model_revision)
    normalized_precision = precision.strip().casefold()
    normalized_quantization = quantization.strip().casefold() if quantization else None
    command_parts = [
        "autonomyfit",
        "benchmark",
        str(model),
        "--model-id",
        resolved_id,
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--precision",
        normalized_precision,
    ]
    if resolved_revision:
        command_parts += ["--model-revision", resolved_revision]
    if backend:
        command_parts += ["--backend", backend]
    if shape:
        command_parts += ["--shape", shape]
    for name, dims in sorted(named_input_shapes.items()):
        command_parts += [
            "--input-shape",
            f"{name}={','.join(str(dim) for dim in dims)}",
        ]
    if provider:
        command_parts += ["--provider", provider]
    if device:
        command_parts += ["--device", device]
    if normalized_quantization:
        command_parts += ["--quantization", normalized_quantization]
    if batch_size is not None:
        command_parts += ["--batch-size", str(batch_size)]
    if trust_artifact:
        command_parts.append("--trust-artifact")
    if compute_units:
        command_parts += ["--compute-units", compute_units]
    hardware = detect_hardware()
    request = BenchmarkRequest(
''',
)
replace_once(
    "src/autonomyfit/cli.py",
    '''        shape_override=shape_override,
        provider=provider,
        device=device,
        precision=precision.strip().casefold(),
        quantization=quantization.strip().casefold() if quantization else None,
''',
    '''        shape_override=shape_override,
        input_shapes=named_input_shapes,
        provider=provider,
        device=device,
        precision=normalized_precision,
        quantization=normalized_quantization,
''',
)
replace_once(
    "src/autonomyfit/cli.py",
    '''        command=(
            f"autonomyfit benchmark {model} --model-id {resolved_id} "
            f"--model-revision {resolved_revision or 'unknown'} "
            f"--backend {backend or 'auto'} --precision {precision.strip().casefold()}"
        ),
''',
    '''        command=shlex.join(command_parts),
''',
)

# Native backend shape/provider hardening.
replace_once(
    "src/autonomyfit/backends.py",
    '''    PowerSampler,
    _shape_for_input,
    latency_summary,
''',
    '''    PowerSampler,
    latency_summary,
''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''def _infer_batch_size(
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
''',
    '''def _infer_batch_size(
    explicit: int | None, input_shapes: dict[str, list[int]]
) -> int | None:
    if explicit is not None:
        return explicit
    first_dims = {dims[0] for dims in input_shapes.values() if dims}
    return next(iter(first_dims)) if len(first_dims) == 1 else None


def _validate_onnx_shape(item: Any, shape: list[int]) -> list[int]:
    graph_dims = list(item.type.tensor_type.shape.dim)
    if len(shape) != len(graph_dims):
        raise BackendError(
            f"ONNX input {item.name!r} expects {len(graph_dims)} dimensions, got {len(shape)}"
        )
    for index, (dim, value) in enumerate(zip(graph_dims, shape)):
        if value <= 0:
            raise BackendError(f"ONNX input {item.name!r} dimension {index} must be positive")
        if dim.HasField("dim_value") and dim.dim_value > 0 and int(dim.dim_value) != value:
            raise BackendError(
                f"ONNX input {item.name!r} dimension {index} is fixed at "
                f"{int(dim.dim_value)}, not {value}"
            )
    return list(shape)


def _resolve_onnx_input_shapes(request: BenchmarkRequest) -> dict[str, list[int]]:
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
    if not inputs:
        raise BackendError("ONNX graph exposes no benchmarkable inputs")
    if request.shape_override is not None and request.input_shapes:
        raise BackendError("--shape and named input shapes cannot be combined")
    if request.shape_override is not None and len(inputs) != 1:
        raise BackendError(
            "--shape is supported only for single-input ONNX models; use --input-shape NAME=... for multi-input graphs"
        )
    input_by_name = {item.name: item for item in inputs}
    unknown = sorted(set(request.input_shapes) - set(input_by_name))
    if unknown:
        raise BackendError("unknown ONNX input shape name(s): " + ", ".join(unknown))

    resolved: dict[str, list[int]] = {}
    for item in inputs:
        override = request.input_shapes.get(item.name)
        if override is None and request.shape_override is not None:
            override = request.shape_override
        if override is not None:
            resolved[item.name] = _validate_onnx_shape(item, list(override))
            continue
        dims: list[int] = []
        for index, dim in enumerate(item.type.tensor_type.shape.dim):
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            elif index == 0:
                dims.append(request.batch_size or 1)
            else:
                raise BackendError(
                    f"ONNX input {item.name!r} has a dynamic non-batch dimension; "
                    "use --input-shape NAME=..."
                )
        if not dims:
            raise BackendError(f"ONNX input {item.name!r} has no resolvable shape")
        resolved[item.name] = dims

    if request.batch_size is not None:
        conflicts = {
            name: dims[0]
            for name, dims in resolved.items()
            if dims and dims[0] != request.batch_size
        }
        if conflicts:
            detail = ", ".join(f"{name}={value}" for name, value in sorted(conflicts.items()))
            raise BackendError(
                f"--batch-size {request.batch_size} conflicts with resolved first dimensions: {detail}"
            )
    return resolved


def _preferred_ort_providers(platform_kind: str) -> tuple[str, ...]:
    platform_map = {
        "jetson": ("CUDAExecutionProvider", "TensorrtExecutionProvider"),
        "nvidia": ("CUDAExecutionProvider", "TensorrtExecutionProvider"),
        "apple": ("CoreMLExecutionProvider",),
        "intel": ("OpenVINOExecutionProvider",),
        "qualcomm": ("QNNExecutionProvider",),
        "amd": ("VitisAIExecutionProvider",),
        "arm": ("XNNPACKExecutionProvider",),
    }
    return (*platform_map.get(platform_kind, ()), "CPUExecutionProvider")
''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''        provider = request.provider
        if provider is None:
            preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
            provider = next((name for name in preferred if name in available), available[0])
''',
    '''        provider = request.provider
        if provider is None:
            preferred = _preferred_ort_providers(request.hardware.platform)
            provider = next((name for name in preferred if name in available), available[0])
''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''        memory = MemorySampler()
        memory.start()
        started = time.perf_counter_ns()
        try:
            session_options = ort.SessionOptions()
''',
    '''        resolved_input_shapes = _resolve_onnx_input_shapes(request)
        memory = MemorySampler()
        memory.start()
        started = time.perf_counter_ns()
        try:
            session_options = ort.SessionOptions()
''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''            inputs = session.get_inputs()
            if request.shape_override is not None and len(inputs) != 1:
                raise BackendError("--shape is supported only for single-input ONNX models")
            feeds: dict[str, object] = {}
            input_shapes: dict[str, list[int]] = {}
            rng = np.random.default_rng(0)
            for input_meta in inputs:
                resolved = _shape_for_input(list(input_meta.shape), request.shape_override)
                dtype = _numpy_dtype(input_meta.type)
''',
    '''            inputs = session.get_inputs()
            runtime_input_names = {item.name for item in inputs}
            if runtime_input_names != set(resolved_input_shapes):
                raise BackendError("ONNX Runtime input names differ from the inspected ONNX graph")
            feeds: dict[str, object] = {}
            input_shapes: dict[str, list[int]] = {}
            rng = np.random.default_rng(0)
            for input_meta in inputs:
                resolved = resolved_input_shapes[input_meta.name]
                dtype = _numpy_dtype(input_meta.type)
''',
)
# TensorRT and OpenVINO must validate any provided named shapes against the ONNX graph.
p = Path("src/autonomyfit/backends.py")
text = p.read_text(encoding="utf-8")
old = '''        native_request = request
        if request.model_path.suffix.casefold() == ".onnx" and not request.input_shapes:
            native_request = replace(request, input_shapes=_resolve_onnx_input_shapes(request))
'''
if text.count(old) != 2:
    raise SystemExit(f"src/autonomyfit/backends.py: expected two native ONNX shape blocks, found {text.count(old)}")
text = text.replace(
    old,
    '''        native_request = request
        if request.model_path.suffix.casefold() == ".onnx":
            native_request = replace(request, input_shapes=_resolve_onnx_input_shapes(request))
''',
)
p.write_text(text, encoding="utf-8")
replace_once(
    "src/autonomyfit/backends.py",
    '''            backend_options={"native_warmup_semantics": "milliseconds", "wall_ms": wall_ms},
''',
    '''            backend_options={
                "native_warmup_semantics": "milliseconds",
                "native_command": command,
                "synthetic_inputs": True,
                "wall_ms": wall_ms,
            },
''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''        if available_devices and device not in available_devices:
            raise BackendError(
                f"OpenVINO device {device!r} unavailable. Available: {', '.join(available_devices)}"
            )
        native_request = replace(native_request, device=device)
''',
    '''        if available_devices:
            canonical_device = next(
                (value for value in available_devices if value.casefold() == device.casefold()),
                None,
            )
            if canonical_device is None:
                raise BackendError(
                    f"OpenVINO device {device!r} unavailable. Available: {', '.join(available_devices)}"
                )
            device = canonical_device
        native_request = replace(native_request, device=device)
''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''                "available_devices": list(available_devices),
                "wall_ms": wall_ms,
''',
    '''                "available_devices": list(available_devices),
                "native_command": command,
                "synthetic_inputs": True,
                "wall_ms": wall_ms,
''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''        memory = MemorySampler()
        memory.start()
        started = time.perf_counter_ns()
        try:
            compute_name = (request.compute_units or "ALL").strip().upper()
''',
    '''        if request.shape_override is not None or request.input_shapes:
            raise BackendError(
                "Core ML generic benchmarking reads shapes from the model contract; "
                "--shape/--input-shape overrides are not supported"
            )
        memory = MemorySampler()
        memory.start()
        started = time.perf_counter_ns()
        try:
            compute_name = (request.compute_units or "ALL").strip().upper()
''',
)
replace_once(
    "src/autonomyfit/backends.py",
    '''            batch_size=_infer_batch_size(request.batch_size, input_shapes), input_shapes=input_shapes,
''',
    '''            batch_size=request.batch_size, input_shapes=input_shapes,
''',
)

# Regression tests.
p = Path("tests/test_evidence.py")
text = p.read_text(encoding="utf-8")
if "test_false_artifact_identity_flag_cannot_be_verified_fit" not in text:
    text += '''\n\ndef test_false_artifact_identity_flag_cannot_be_verified_fit():
    report = _report()
    report["measurement"]["artifact_identity_verified"] = False
    validate_benchmark_report(report)
    evidence = benchmark_evidence_from_report(report)
    assert evidence.verified_identity is False
    assert evidence.eligible_for_verified_fit is False


def test_local_report_requires_explicit_measurement_classification():
    report = _report()
    report.pop("measurement")
    with pytest.raises(EvidenceSchemaError):
        validate_benchmark_report(report)
'''
p.write_text(text, encoding="utf-8")

p = Path("tests/test_local_results_stage5.py")
text = p.read_text(encoding="utf-8")
if "test_native_runtime_disappearance_invalidates_local_result" not in text:
    text += '''\n\ndef test_native_runtime_disappearance_invalidates_local_result():
    hardware = _hardware()
    document = _report(hardware)
    document["software"].update(
        {"runtime": "openvino", "runtime_version": "2026.3.0", "provider": "CPU"}
    )
    valid, reasons = local_report_compatibility(
        document, hardware, now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("runtime is no longer available" in reason for reason in reasons)


def test_unknown_current_runtime_version_invalidates_exact_local_context():
    hardware = _hardware()
    unknown = HardwareProfile(
        platform=hardware.platform,
        os_name=hardware.os_name,
        architecture=hardware.architecture,
        cpu=hardware.cpu,
        ram_total_gb=hardware.ram_total_gb,
        ram_available_gb=hardware.ram_available_gb,
        gpu=hardware.gpu,
        driver=hardware.driver,
        runtimes=(RuntimeCapability("onnxruntime", True, None, provider="CPUExecutionProvider"),),
    )
    document = _report(unknown)
    valid, reasons = local_report_compatibility(
        document, unknown, now=datetime(2026, 8, 16, tzinfo=timezone.utc)
    )
    assert valid is False
    assert any("runtime version could not be established" in reason for reason in reasons)
'''
p.write_text(text, encoding="utf-8")

replace_once(
    "tests/test_cli.py",
    '''from autonomyfit.cli import app
''',
    '''from autonomyfit.cli import _parse_named_shapes, app
''',
)
p = Path("tests/test_cli.py")
text = p.read_text(encoding="utf-8")
if "test_named_input_shapes_are_parsed_and_duplicates_rejected" not in text:
    text += '''\n\ndef test_named_input_shapes_are_parsed_and_duplicates_rejected():
    assert _parse_named_shapes(["image=1,3,640,640", "mask=1,1,640,640"]) == {
        "image": [1, 3, 640, 640],
        "mask": [1, 1, 640, 640],
    }
    try:
        _parse_named_shapes(["image=1,3,640,640", "image=1,3,224,224"])
    except Exception as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate named input shape was accepted")


def test_shape_and_named_input_shape_conflict_is_rejected(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"placeholder")
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(model),
            "--shape",
            "1,3,640,640",
            "--input-shape",
            "image=1,3,640,640",
        ],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output
'''
p.write_text(text, encoding="utf-8")

replace_once(
    "tests/test_backends.py",
    '''    CoreMLBackend,
    build_openvino_command,
''',
    '''    CoreMLBackend,
    _preferred_ort_providers,
    build_openvino_command,
''',
)
p = Path("tests/test_backends.py")
text = p.read_text(encoding="utf-8")
if "test_onnxruntime_default_provider_order_is_platform_specific" not in text:
    text += '''\n\ndef test_onnxruntime_default_provider_order_is_platform_specific():
    assert _preferred_ort_providers("nvidia")[:2] == (
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
    )
    assert _preferred_ort_providers("apple")[0] == "CoreMLExecutionProvider"
    assert _preferred_ort_providers("intel")[0] == "OpenVINOExecutionProvider"
    assert _preferred_ort_providers("qualcomm")[0] == "QNNExecutionProvider"
'''
p.write_text(text, encoding="utf-8")

# Documentation and the next pre-1.0 version.
p = Path("docs/benchmarking.md")
text = p.read_text(encoding="utf-8")
if "## Self-hosted target validation" not in text:
    text += '''\n\n## Self-hosted target validation\n\n`.github/workflows/hardware-validation.yml` is the permanent manual harness for Jetson, discrete NVIDIA, Intel and physical Apple target runs. Attach the target as a self-hosted GitHub runner, give it a dedicated label such as `autonomyfit-hardware`, preinstall the vendor runtime, and dispatch the workflow with an exact model path, immutable model revision, runtime/provider/device, precision, batch and shape context.\n\nThe workflow captures `scan`, backend availability, the schema-v2 benchmark report, an inspected summary and a separate machine-class attestation. `operator_machine_class=physical` is an operator attestation and is recorded as such; AutonomyFit does not independently infer that a self-hosted runner is bare-metal. VM/container runs remain useful native execution validation but must not be described as physical target evidence.\n\nMulti-input ONNX graphs use repeatable `--input-shape NAME=...` values (semicolon-separated in the workflow input). Named shapes are checked against graph input names, ranks and fixed dimensions before execution.\n'''
p.write_text(text, encoding="utf-8")

replace_once(
    "README.md",
    '''Local evidence is invalidated rather than silently reused when it becomes stale or a material execution identity changes, including hardware identity, OS identity, driver major version, runtime major version, or required ONNX Runtime execution-provider availability.
''',
    '''Local evidence is invalidated rather than silently reused when it becomes stale or a material execution identity changes, including hardware identity, OS identity, driver state, power mode, native runtime/provider availability, exact runtime/provider version when known, or the material software-stack fingerprint.
''',
)
replace_once(
    "README.md",
    '''See [docs/hardware.md](docs/hardware.md) and [docs/benchmarking.md](docs/benchmarking.md).
''',
    '''See [docs/hardware.md](docs/hardware.md) and [docs/benchmarking.md](docs/benchmarking.md). A manual self-hosted hardware-validation workflow is included for collecting exact target evidence without classifying VMs or containers as physical hardware.
''',
)
replace_once("pyproject.toml", 'version = "0.7.0"', 'version = "0.8.0"')
replace_once("src/autonomyfit/__init__.py", '__version__ = "0.7.0"', '__version__ = "0.8.0"')

p = Path(".github/workflows/ci.yml")
text = p.read_text(encoding="utf-8").replace('"0.7.0"', '"0.8.0"')
old = '''          assert len(report['reproducibility']['software_stack_fingerprint']) == 64
          assert matrix['local_measured'] == 1
'''
new = '''          assert len(report['reproducibility']['software_stack_fingerprint']) == 64
          command = report['reproducibility']['command'] or ''
          assert '--backend onnxruntime' in command
          assert '--provider CPUExecutionProvider' in command
          assert '--batch-size 1' in command
          assert '--backend auto' not in command
          assert matrix['local_measured'] == 1
'''
if text.count(old) != 1:
    raise SystemExit(f".github/workflows/ci.yml: native command assertion anchor count={text.count(old)}")
p.write_text(text.replace(old, new), encoding="utf-8")
PY

python3 -m compileall -q src
git diff --check

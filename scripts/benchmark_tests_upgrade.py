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


# Local-measured evidence must identify that it came from detected hardware.
replace_once(
    "src/autonomyfit/evidence.py",
    '''    provider_version: str | None = None\n    notes: str | None = None\n    verified_identity: bool = False\n''',
    '''    provider_version: str | None = None\n    machine_source: str | None = None\n    notes: str | None = None\n    verified_identity: bool = False\n''',
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''    def eligible_for_verified_fit(self) -> bool:\n        return (\n            self.quality in {"local-measured", "standardized"}\n''',
    '''    def eligible_for_verified_fit(self) -> bool:\n        local_machine_ok = self.quality != "local-measured" or self.machine_source == "detected"\n        return (\n            self.quality in {"local-measured", "standardized"}\n            and local_machine_ok\n''',
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''                provider_version=runtime.get("version"),\n                notes=item.get("notes"),\n''',
    '''                provider_version=runtime.get("version"),\n                machine_source=None,\n                notes=item.get("notes"),\n''',
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''        provider_version=software.get("provider_version"),\n        notes=document.get("notes"),\n''',
    '''        provider_version=software.get("provider_version"),\n        machine_source=(document.get("measurement") or {}).get("machine_source"),\n        notes=document.get("notes"),\n''',
)
replace_once(
    "src/autonomyfit/evidence.py",
    '''    artifact = document["artifact"]\n    if not artifact.get("sha256"):\n        raise EvidenceSchemaError("local benchmark reports require artifact.sha256")\n''',
    '''    artifact = document["artifact"]\n    if not artifact.get("sha256"):\n        raise EvidenceSchemaError("local benchmark reports require artifact.sha256")\n    measurement = document.get("measurement")\n    if measurement is not None and (\n        measurement.get("machine_source") != "detected" or measurement.get("profile_only")\n    ):\n        raise EvidenceSchemaError(\n            "local benchmark reports cannot claim measurements from a profile-only hardware target"\n        )\n''',
)

# Match the provider-version convention used by the ORT backend when checking current stack.
replace_once(
    "src/autonomyfit/local_results.py",
    '''            provider_version=current_runtime_version if provider else None,\n''',
    '''            provider_version=(\n                f"onnxruntime-{current_runtime_version}"\n                if provider and str(runtime).casefold() in {"onnx", "onnxruntime"}\n                else (current_runtime_version if provider else None)\n            ),\n''',
)

# Core ML compute-unit selection is available from deployment validation too.
replace_once(
    "src/autonomyfit/deployment.py",
    '''    allow_restricted_license: bool = False\n''',
    '''    allow_restricted_license: bool = False\n    compute_units: str | None = None\n''',
)
replace_once(
    "src/autonomyfit/deployment.py",
    '''                expected_sha256=final_artifact.sha256,\n            )\n''',
    '''                expected_sha256=final_artifact.sha256,\n                compute_units=options.compute_units,\n            )\n''',
)
replace_once(
    "src/autonomyfit/deployment_cli.py",
    '''        device: Annotated[str | None, typer.Option(help="Native runtime device, e.g. CPU/GPU/NPU.")] = None,\n        convert: Annotated[bool, typer.Option("--convert", help="Persist a supported local conversion before benchmarking.")] = False,\n''',
    '''        device: Annotated[str | None, typer.Option(help="Native runtime device, e.g. CPU/GPU/NPU.")] = None,\n        compute_units: Annotated[\n            str | None,\n            typer.Option("--compute-units", help="Core ML compute units: ALL, CPU_ONLY, CPU_AND_GPU or CPU_AND_NE."),\n        ] = None,\n        convert: Annotated[bool, typer.Option("--convert", help="Persist a supported local conversion before benchmarking.")] = False,\n''',
)
replace_once(
    "src/autonomyfit/deployment_cli.py",
    '''                    device=device,\n                    convert=convert,\n''',
    '''                    device=device,\n                    compute_units=compute_units,\n                    convert=convert,\n''',
)

# OpenVINO's benchmark_app owns native warmup; do not mislabel the CLI warmup count as executed.
replace_once(
    "src/autonomyfit/backends.py",
    '''            warmup=request.warmup,\n            iterations=request.iterations,\n            backend_options={\n                "performance_hint": "latency",\n                "native_warmup": True,\n''',
    '''            warmup=0,\n            iterations=request.iterations,\n            backend_options={\n                "performance_hint": "latency",\n                "native_warmup": True,\n                "requested_warmup_count_not_applied": request.warmup,\n''',
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
replace_once(
    "tests/test_evidence.py",
    '''        "software_stack_id": None,\n        "verified_identity": True,\n''',
    '''        "software_stack_id": "stack-1",\n        "provider_version": "1.0",\n        "machine_source": "detected",\n        "verified_identity": True,\n''',
)
replace_once(
    "tests/test_evidence.py",
    '''        "runtime_version": "1.0",\n        "today": date(2026, 8, 16),\n''',
    '''        "runtime_version": "1.0",\n        "provider": "CPUExecutionProvider",\n        "provider_version": "1.0",\n        "batch_size": 1,\n        "input_shapes": {"input": [1, 3, 640, 640]},\n        "software_stack_id": "stack-1",\n        "today": date(2026, 8, 16),\n''',
)
replace_once(
    "tests/test_evidence.py",
    '''        "reproducibility": {"command": "autonomyfit benchmark", "hostname_hash": "abc", "environment_fingerprint": "12345678"},\n''',
    '''        "measurement": {"machine_source": "detected", "profile_only": False, "artifact_identity_verified": True},\n        "reproducibility": {\n            "command": "autonomyfit benchmark",\n            "hostname_hash": "abc",\n            "environment_fingerprint": "12345678",\n            "software_stack_fingerprint": "1" * 64,\n        },\n''',
)

append_once(
    "tests/test_evidence.py",
    '''def test_local_evidence_requires_full_execution_context_for_exact_match():\n    evidence = _evidence()\n    assert _match(evidence, batch_size=2) == []\n    assert _match(evidence, input_shapes={"input": [1, 3, 224, 224]}) == []\n    assert _match(evidence, provider_version="2.0") == []\n    assert _match(evidence, software_stack_id="stack-2") == []\n\n    contextual = _match(evidence, software_stack_id=None)[0]\n    assert contextual.exact is False\n    assert any("software stack" in reason for reason in contextual.reasons)\n\n\ndef test_profile_only_local_report_is_rejected():\n    report = _report()\n    report["measurement"] = {\n        "machine_source": "profile",\n        "profile_only": True,\n        "artifact_identity_verified": True,\n    }\n    with pytest.raises(EvidenceSchemaError, match="profile-only"):\n        validate_benchmark_report(report)\n''',
)

append_once(
    "tests/test_backends.py",
    '''def test_run_benchmark_rejects_profile_only_hardware(tmp_path):\n    from autonomyfit.backends import run_benchmark\n\n    path = tmp_path / "model.onnx"\n    path.write_bytes(b"model")\n    profile = HardwareProfile(\n        platform="nvidia", os_name="profile", architecture="x86_64", cpu="profile",\n        ram_total_gb=16, ram_available_gb=12, matched_profile="nvidia-t4-16gb",\n    )\n    request = BenchmarkRequest(\n        model_path=path, model_id="demo", model_revision="r1", hardware=profile\n    )\n    with pytest.raises(BackendError, match="detected hardware"):\n        run_benchmark(request, "onnxruntime")\n\n\ndef test_coreml_compute_unit_validation_is_explicit(monkeypatch, tmp_path):\n    backend = CoreMLBackend()\n    monkeypatch.setattr(backend, "availability", lambda: type("A", (), {"available": True, "detail": None, "version": "9"})())\n    request = BenchmarkRequest(\n        model_path=tmp_path / "model.mlmodel", model_id="demo", model_revision="r1",\n        hardware=HARDWARE, compute_units="not-a-real-unit",\n    )\n    # Import may fail before compute-unit validation on non-macOS test hosts; the production macOS\n    # path is exercised separately. The request field itself must remain explicit and serialisable.\n    assert request.compute_units == "not-a-real-unit"\n''',
)

append_once(
    "tests/test_hardware.py",
    '''def test_detected_machine_identity_does_not_collapse_to_profile(monkeypatch):\n    from autonomyfit.benchmark import hardware_evidence_id\n    from autonomyfit.models import HardwareProfile\n\n    profile = hardware_from_profile("jetson-orin-nx-16gb")\n    detected = HardwareProfile(\n        platform="jetson", os_name="Linux", architecture="aarch64", cpu="Jetson",\n        ram_total_gb=16, ram_available_gb=12, gpu="Jetson Orin NX",\n        matched_profile="jetson-orin-nx-16gb", memory_topology="unified",\n    )\n    monkeypatch.setattr("autonomyfit.benchmark._machine_identity_hash", lambda: "machine-a")\n    assert hardware_evidence_id(profile) == "jetson-orin-nx-16gb"\n    assert hardware_evidence_id(detected).startswith("local-jetson-")\n    assert hardware_evidence_id(detected) != hardware_evidence_id(profile)\n''',
)

append_once(
    "tests/test_local_results_stage5.py",
    '''def test_power_mode_change_invalidates_local_result():\n    hardware = _hardware()\n    hardware = HardwareProfile(**{**hardware.to_dict(), "power_mode": "MODE_15W"})\n    report = _report(hardware)\n    report["hardware"]["power_mode"] = "MODE_30W"\n    valid, reasons = local_report_compatibility(\n        report, hardware, now=datetime(2026, 8, 16, tzinfo=timezone.utc)\n    )\n    assert valid is False\n    assert any("power mode" in reason for reason in reasons)\n''',
)

# Update the exact-local precedence test to reflect the complete matrix identity.
replace_once(
    "tests/test_deployment_stage5.py",
    '''        source_date="2026-08-16", software_stack_id=None, verified_identity=True,\n''',
    '''        source_date="2026-08-16", software_stack_id="stack-1", provider_version="trtexec-1",\n        machine_source="detected", verified_identity=True,\n''',
)
replace_once(
    "tests/test_deployment_stage5.py",
    '''            artifact_sha256="a" * 64, runtime="tensorrt", precision="fp16",\n            max_latency_ms=5.0,\n''',
    '''            artifact_sha256="a" * 64, runtime="tensorrt", precision="fp16",\n            provider="trtexec", provider_version="trtexec-1", batch_size=1,\n            input_shapes={"input": [1, 3, 640, 640]}, software_stack_id="stack-1",\n            max_latency_ms=5.0,\n''',
)

append_once(
    "tests/test_deployment_stage5.py",
    '''def test_profile_benchmark_mismatch_is_refused(monkeypatch):\n    actual = _fake_hardware()\n    monkeypatch.setattr("autonomyfit.deployment.detect_hardware", lambda: actual)\n    try:\n        validate_deployment(\n            ValidationOptions(\n                model_id="yolo26n", offline=True, benchmark=True,\n                hardware_profile="nvidia-t4-16gb",\n            )\n        )\n    except Exception as exc:\n        assert "actual machine" in str(exc) or "does not match" in str(exc)\n    else:\n        raise AssertionError("profile-only benchmark should have been refused")\n''',
)

matrix_test = ROOT / "tests/test_benchmark_matrix.py"
matrix_test.write_text('''from autonomyfit.benchmark_matrix import benchmark_matrix, matrix_key\nfrom autonomyfit.evidence import (\n    BenchmarkEvidence, EvidenceStore, LatencyStats, PowerStats\n)\n\n\ndef _evidence(**overrides):\n    values = dict(\n        id="local-a", model_id="demo", model_revision="r1", artifact_id="a",\n        artifact_sha256="a" * 64, artifact_format="onnx", hardware_id="local-cpu-a",\n        hardware_name="CPU", runtime="onnxruntime", runtime_version="1.20",\n        provider="CPUExecutionProvider", precision="fp32", quantization=None, batch_size=1,\n        input_shapes={"input": [1, 4]}, power_mode=None, clocks={}, warmup=2, iterations=5,\n        latency=LatencyStats(mean_ms=1.0, median_ms=1.0), throughput_fps=1000.0,\n        power=PowerStats(), peak_memory_mb=10, peak_memory_scope="process RSS",\n        quality="local-measured", source_id="local", source_url="local://benchmark",\n        source_date="2026-08-17", software_stack_id="stack-a",\n        provider_version="onnxruntime-1.20", machine_source="detected", verified_identity=True,\n    )\n    values.update(overrides)\n    return BenchmarkEvidence(**values)\n\n\ndef test_matrix_key_changes_for_material_execution_context():\n    base = _evidence()\n    assert matrix_key(base) != matrix_key(_evidence(id="b", batch_size=2))\n    assert matrix_key(base) != matrix_key(_evidence(id="c", input_shapes={"input": [1, 8]}))\n    assert matrix_key(base) != matrix_key(_evidence(id="d", software_stack_id="stack-b"))\n\n\ndef test_matrix_summary_reports_exact_context():\n    evidence = _evidence()\n    payload = benchmark_matrix(\n        store=EvidenceStore(document={}, benchmarks=(evidence,)), local_only=True\n    )\n    assert payload["record_count"] == 1\n    assert payload["local_measured"] == 1\n    assert payload["exact_context_complete"] == 1\n''', encoding="utf-8")

# ---------------------------------------------------------------------------
# Docs: distinguish architecture, detected runtime and physical validation.
# ---------------------------------------------------------------------------
benchmarking = '''# Benchmarking\n\nAutonomyFit benchmarks exact artifacts on detected hardware and records the execution context as evidence. A bundled hardware profile is a screening target only and cannot create `local-measured` evidence.\n\n## Evidence matrix identity\n\n`autonomyfit benchmark-matrix` exposes the dimensions that make a benchmark applicable:\n\n- model ID and immutable model revision\n- artifact SHA-256\n- exact detected machine identity\n- runtime and runtime version\n- provider/device and provider version\n- precision and quantisation\n- batch size and input shapes\n- power mode when relevant\n- material software-stack fingerprint\n\nLocal measured evidence is eligible for `VERIFIED_FIT` only when this execution context is complete and the caller requests a matching context. A benchmark from another machine, batch, shape, provider or material software stack remains contextual rather than silently overriding other evidence.\n\n```bash\nautonomyfit benchmark-matrix --local-only\nautonomyfit benchmark-matrix --model-id MODEL --json\n```\n\n## ONNX Runtime\n\n`OnnxRuntimeBackend` runs deterministic synthetic inputs in-process, separates session load from timed inference, records min/mean/median/p50/p90/p95/p99/max/stdev latency, throughput, process RSS and scoped NVIDIA/Jetson power where available. Accelerator-provider requests disable CPU execution-provider fallback and the report records requested/active providers, provider options and ONNX Runtime build information. Synthetic inputs measure execution, not task accuracy.\n\n## TensorRT\n\n`TensorRTBackend` uses `trtexec`. ONNX inputs are inspected to record concrete input names/shapes before the command runs. Dynamic non-batch shapes must be supplied explicitly. Serialized engine/plan artifacts require explicit trust. TensorRT engine evidence is tied to the recorded GPU/software stack and is not assumed portable. `trtexec --warmUp` is a millisecond setting and is labelled as such in backend options.\n\n## OpenVINO\n\n`OpenVINOBackend` uses `benchmark_app`, defaults to an explicit `CPU` device rather than `AUTO`, records exposed OpenVINO devices where available, and fails if an explicitly requested detected device is unavailable. Native benchmark-app warmup is owned by the tool; AutonomyFit does not misreport the generic CLI warmup count as executed OpenVINO warmup iterations.\n\n## Core ML\n\n`CoreMLBackend` supports numeric `MLMultiArray` inputs on macOS. Compute units are explicit (`ALL`, `CPU_ONLY`, `CPU_AND_GPU`, or `CPU_AND_NE`) and recorded in the evidence. Image-typed or otherwise unsupported generic inputs fail explicitly. Detailed Neural Engine/GPU attribution still requires Apple profiling tools.\n\n## Reproducible local run\n\n```bash\nautonomyfit benchmark model.onnx \\\n  --model-id my-model \\\n  --model-revision COMMIT \\\n  --backend onnxruntime \\\n  --provider CPUExecutionProvider \\\n  --precision fp32 \\\n  --batch-size 1 \\\n  -o result.json\n\nautonomyfit benchmark-inspect result.json\nautonomyfit benchmark-import result.json\nautonomyfit benchmark-matrix --local-only\n```\n\nReports include an artifact SHA-256, exact machine ID, runtime/provider versions, input shapes, batch, deterministic seed, latency distribution, throughput, process RSS, scoped power/energy when available, environment fingerprint and material software-stack fingerprint. The artifact digest is checked before and after native execution.\n\n## Validation status\n\nThe regular CI executes a real ONNX Runtime `CPUExecutionProvider` benchmark against a generated ONNX graph and validates/imports its report. That is native runtime validation on the GitHub-hosted CI machine, not evidence for Jetson, NVIDIA GPU, Apple Silicon or Intel GPU/NPU hardware. Platform-specific support must be described as physically validated only after a report was measured on that detected target.\n'''
(ROOT / "docs/benchmarking.md").write_text(benchmarking, encoding="utf-8")

hardware = '''# Hardware and runtime model\n\nAutonomyFit separates four levels that should not be conflated:\n\n1. **profile support**: a bundled target description exists\n2. **runtime detection**: the installed machine exposes a runtime/provider/device\n3. **native benchmark execution**: an exact artifact successfully ran through that backend\n4. **physical target evidence**: a benchmark report was measured on the named detected hardware\n\n## Target categories\n\nBundled profiles cover NVIDIA Jetson, discrete NVIDIA GPUs, Apple Silicon, Intel CPU/GPU/NPU systems, AMD Ryzen AI, Qualcomm Snapdragon X Elite and Arm CPU targets. Profile capability is not proof that a specific model graph runs.\n\nONNX Runtime bridge providers such as CUDA, TensorRT, OpenVINO, Core ML, QNN, XNNPACK and Vitis AI remain `verified=false` for model coverage until the exact graph is exercised.\n\n## Detected identity\n\nLocal evidence uses a host-specific hashed machine identity plus detected CPU/GPU/memory topology rather than collapsing a real machine to a coarse bundled profile ID. Driver, JetPack/L4T, power mode, runtime/provider versions and software-stack signals are tracked separately so material stack changes can invalidate exact evidence without pretending the hardware itself changed.\n\nJetson detection records JetPack package metadata where available, L4T information, `nvpmodel` power mode and VDD_IN telemetry when exposed. NVIDIA discrete GPUs use `nvidia-smi` identity/driver and board-power telemetry. Intel detection records OpenVINO CPU/GPU/NPU devices exposed by the installed runtime. Apple detection records the Apple chip and explicit Core ML compute-unit choice in benchmark evidence.\n\n## Physical validation\n\nA hardware profile must never create a local measurement. `validate --benchmark --hardware-profile PROFILE` is permitted only when the detected machine matches that profile, and the resulting report uses detected hardware identity.\n\nUse:\n\n```bash\nautonomyfit scan --json\nautonomyfit benchmark-backends --json\nautonomyfit benchmark ARTIFACT --model-id MODEL --model-revision REVISION -o benchmark.json\nautonomyfit benchmark-inspect benchmark.json\nautonomyfit benchmark-matrix --local-only\n```\n\nCI/native-backend success is not automatically called Jetson/NVIDIA/Apple/Intel physical validation. Those claims require evidence generated on the corresponding detected target.\n'''
(ROOT / "docs/hardware.md").write_text(hardware, encoding="utf-8")

# README wording for exact context and validation levels.
replace_once(
    "README.md",
    '''Successful validation benchmarks can be imported automatically into the local-results layer. Exact local measurements outrank generic vendor/reference evidence for the same artifact, hardware, runtime and precision.\n''',
    '''Successful validation benchmarks can be imported automatically into the local-results layer. Exact local measurements outrank generic vendor/reference evidence only when model revision, artifact hash, detected machine, runtime/provider versions, precision/quantisation, batch/input shapes, relevant power mode and material software stack match.\n\nUse `autonomyfit benchmark-matrix` to inspect those applicability dimensions and see which records are complete enough for exact-context use.\n''',
)
replace_once(
    "README.md",
    '''First-class profiles cover NVIDIA Jetson and discrete GPUs, Apple Silicon, Intel CPU/GPU/NPU systems, AMD Ryzen AI systems, Qualcomm Snapdragon X Elite and Arm CPU targets. Native benchmark paths include ONNX Runtime, TensorRT, OpenVINO and Core ML where the required tooling exists.\n''',
    '''First-class profiles cover NVIDIA Jetson and discrete GPUs, Apple Silicon, Intel CPU/GPU/NPU systems, AMD Ryzen AI systems, Qualcomm Snapdragon X Elite and Arm CPU targets. Native benchmark paths include ONNX Runtime, TensorRT, OpenVINO and Core ML where the required tooling exists. Profiles describe targets; detected runtime availability, successful native execution and physical target evidence are reported as separate levels.\n''',
)
replace_once(
    "README.md",
    '''- Exact deployment evidence is still sparse across the full hardware x runtime x model matrix.\n''',
    '''- Exact deployment evidence is still sparse across the full hardware x runtime x model matrix. The benchmark matrix tooling prevents sparse profile/vendor data from being presented as exact local measurements.\n''',
)

# ---------------------------------------------------------------------------
# Permanent CI native ONNX Runtime smoke on the actual hosted machine.
# ---------------------------------------------------------------------------
ci_path = ROOT / ".github/workflows/ci.yml"
ci = ci_path.read_text(encoding="utf-8")
marker = '''  package:\n    name: Build package\n'''
native_job = '''  native-benchmark:\n    name: Native ONNX Runtime benchmark\n    needs: test\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09 # v5\n      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6\n        with:\n          python-version: "3.12"\n      - name: Install native benchmark dependencies\n        run: python -m pip install -e '.[dev,benchmark,deployment]'\n      - name: Build deterministic ONNX identity graph\n        run: |\n          python - <<'PY'\n          import onnx\n          from onnx import TensorProto, helper\n          graph = helper.make_graph(\n              [helper.make_node("Identity", ["input"], ["output"])],\n              "autonomyfit-ci-identity",\n              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],\n              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],\n          )\n          model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])\n          model.ir_version = min(model.ir_version, 10)\n          onnx.save(model, "/tmp/autonomyfit-ci.onnx")\n          PY\n      - name: Execute and inspect native benchmark\n        env:\n          AUTONOMYFIT_EVIDENCE_DIR: ${{ runner.temp }}/autonomyfit-native-evidence\n        run: |\n          autonomyfit benchmark /tmp/autonomyfit-ci.onnx \\\n            --model-id autonomyfit-ci-identity \\\n            --model-revision "$GITHUB_SHA" \\\n            --backend onnxruntime \\\n            --provider CPUExecutionProvider \\\n            --precision fp32 \\\n            --batch-size 1 \\\n            --iterations 10 \\\n            --warmup 2 \\\n            -o /tmp/autonomyfit-native.json >/tmp/autonomyfit-native-output.txt\n          autonomyfit benchmark-inspect /tmp/autonomyfit-native.json >/dev/null\n          autonomyfit benchmark-import /tmp/autonomyfit-native.json >/dev/null\n          autonomyfit benchmark-matrix --local-only --json >/tmp/autonomyfit-matrix.json\n          python - <<'PY'\n          import json\n          report = json.load(open('/tmp/autonomyfit-native.json'))\n          matrix = json.load(open('/tmp/autonomyfit-matrix.json'))\n          assert report['measurement']['machine_source'] == 'detected'\n          assert report['measurement']['profile_only'] is False\n          assert report['software']['provider'] == 'CPUExecutionProvider'\n          assert report['software']['provider_version']\n          assert report['execution']['batch_size'] == 1\n          assert report['execution']['input_shapes'] == {'input': [1, 4]}\n          assert len(report['reproducibility']['software_stack_fingerprint']) == 64\n          assert matrix['local_measured'] == 1\n          assert matrix['exact_context_complete'] == 1\n          PY\n\n'''
if native_job.strip() not in ci:
    if marker not in ci:
        raise SystemExit("CI package marker not found")
    ci = ci.replace(marker, native_job + marker, 1)
    ci_path.write_text(ci, encoding="utf-8")

print("AutonomyFit benchmark evidence tests/docs upgrade applied")

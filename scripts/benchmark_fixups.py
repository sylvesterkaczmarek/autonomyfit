from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/autonomyfit/scoring.py",
    '    provider = provider_override or _BRIDGE_RUNTIMES.get(runtime)\n    runtime_candidates = ("onnxruntime",) if provider else (\n',
    '    bridge_provider = _BRIDGE_RUNTIMES.get(runtime)\n    provider = provider_override or bridge_provider\n    runtime_candidates = ("onnxruntime",) if bridge_provider else (\n',
)

replace_once(
    "tests/test_deployment_stage5.py",
    "from autonomyfit.deployment import (\n    ValidationOptions,",
    "from autonomyfit.deployment import (\n    DeploymentValidationError,\n    ValidationOptions,",
)
replace_once(
    "tests/test_deployment_stage5.py",
    "    except Exception as exc:\n",
    "    except DeploymentValidationError as exc:\n",
)
replace_once(
    "tests/test_deployment_stage5.py",
    '    hardware = hardware_from_profile("jetson-orin-nx-16gb")\n    local = BenchmarkEvidence(',
    '    hardware = replace(\n        hardware_from_profile("jetson-orin-nx-16gb"),\n        runtimes=(RuntimeCapability("tensorrt", True, "10.0", "local"),),\n    )\n    local = BenchmarkEvidence(',
)
replace_once(
    "tests/test_deployment_stage5.py",
    'runtime="tensorrt", runtime_version=None, provider="trtexec",\n        precision="fp16", quantization=None, batch_size=1, input_shapes={}, power_mode=None,',
    'runtime="tensorrt", runtime_version="10.0", provider="trtexec",\n        precision="fp16", quantization=None, batch_size=1,\n        input_shapes={"input": [1, 3, 640, 640]}, power_mode=None,',
)

replace_once(
    "tests/test_confidence_stage4.py",
    "        software_stack_id=None, verified_identity=True,",
    '        software_stack_id="stack-1", provider_version="10.0",\n        machine_source="detected", verified_identity=True,',
)
replace_once(
    "tests/test_confidence_stage4.py",
    '        "artifact_sha256": "a" * 64, "runtime": "tensorrt", "precision": "fp16",\n        "max_latency_ms": 5.0,',
    '        "artifact_sha256": "a" * 64, "runtime": "tensorrt", "precision": "fp16",\n        "provider": "trtexec", "provider_version": "10.0", "batch_size": 1,\n        "input_shapes": {"images": [1, 3, 640, 640]}, "power_mode": "MAXN",\n        "software_stack_id": "stack-1", "max_latency_ms": 5.0,',
)

replace_once(
    "tests/test_scoring.py",
    "        software_stack_id=None,\n        verified_identity=True,",
    '        software_stack_id="stack-1",\n        provider_version="10.0",\n        machine_source="detected",\n        verified_identity=True,',
)
replace_once(
    "tests/test_scoring.py",
    '            artifact_sha256="a" * 64,\n        ),',
    '            artifact_sha256="a" * 64,\n            provider="trtexec",\n            provider_version="10.0",\n            batch_size=1,\n            input_shapes={"images": [1, 3, 640, 640]},\n            power_mode="MAXN",\n            software_stack_id="stack-1",\n        ),',
)

replace_once(
    "tests/test_local_results_stage5.py",
    'def test_power_mode_change_invalidates_local_result():\n    hardware = _hardware()\n    hardware = HardwareProfile(**{**hardware.to_dict(), "power_mode": "MODE_15W"})',
    'def test_power_mode_change_invalidates_local_result():\n    from dataclasses import replace\n\n    hardware = replace(_hardware(), power_mode="MODE_15W")',
)

print("AutonomyFit benchmark validation fixups applied")

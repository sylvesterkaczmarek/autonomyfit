from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if addition.strip() not in text:
        target.write_text(text.rstrip() + "\n\n" + addition.strip() + "\n", encoding="utf-8")


replace_once(
    "src/autonomyfit/backends.py",
    "import re\n",
    "import importlib.metadata\nimport re\n",
)
replace_once(
    "src/autonomyfit/backends.py",
    '''        except (OSError, subprocess.SubprocessError):
            version_text = None
        return BackendAvailability(self.name, True, version_text, executable)

    def benchmark(self, request: BenchmarkRequest) -> dict[str, Any]:
        availability = self.availability()
        if not availability.available or not availability.executable:
            raise BackendError("OpenVINO backend unavailable: benchmark_app not found")
''',
    '''        except (OSError, subprocess.SubprocessError):
            version_text = None
        if not version_text:
            try:
                version_text = importlib.metadata.version("openvino")
            except importlib.metadata.PackageNotFoundError:
                version_text = None
        return BackendAvailability(self.name, True, version_text, executable)

    def benchmark(self, request: BenchmarkRequest) -> dict[str, Any]:
        availability = self.availability()
        if not availability.available or not availability.executable:
            raise BackendError("OpenVINO backend unavailable: benchmark_app not found")
''',
)

replace_once(
    "src/autonomyfit/hardware.py",
    '''def _cpu_brand() -> str:
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
''',
    '''def _cpu_brand() -> str:
    if platform.system() == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        for key in ("model name", "Hardware", "Processor"):
            match = re.search(
                rf"^{re.escape(key)}\s*:\s*(.+)$",
                text,
                re.MULTILINE | re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
    value = platform.processor() or platform.uname().processor
    return value or "unknown"
''',
)

append_once(
    "tests/test_backends.py",
    '''def test_openvino_version_falls_back_to_installed_package(monkeypatch):
    from autonomyfit.backends import OpenVINOBackend

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/benchmark_app")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: type("Result", (), {"stdout": "usage only", "stderr": ""})(),
    )
    monkeypatch.setattr(
        "autonomyfit.backends.importlib.metadata.version",
        lambda name: "2026.3.0" if name == "openvino" else "0",
    )
    availability = OpenVINOBackend().availability()
    assert availability.available is True
    assert availability.version == "2026.3.0"
''',
)

append_once(
    "tests/test_hardware.py",
    '''def test_linux_cpu_brand_prefers_cpuinfo_model(monkeypatch):
    from autonomyfit.hardware import _cpu_brand

    monkeypatch.setattr("autonomyfit.hardware.platform.system", lambda: "Linux")
    monkeypatch.setattr("autonomyfit.hardware.platform.processor", lambda: "x86_64")
    monkeypatch.setattr(
        "autonomyfit.hardware.Path.read_text",
        lambda self, **kwargs: "model name : Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz"
        if str(self) == "/proc/cpuinfo"
        else "",
    )
    assert _cpu_brand() == "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz"
''',
)

print("AutonomyFit native platform follow-up fixes applied")

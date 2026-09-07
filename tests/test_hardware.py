import pytest

from autonomyfit.hardware import hardware_from_profile, match_hardware_profile


def test_jetson_alias_matching():
    assert match_hardware_profile("NVIDIA Jetson Orin NX 16GB") == "jetson-orin-nx-16gb"
    assert match_hardware_profile("reComputer J4012 powered by Orin NX") == "jetson-orin-nx-16gb"


def test_unknown_hardware_does_not_fake_match():
    assert match_hardware_profile("Some Future Accelerator 123") is None


def test_profile_uses_unified_memory_for_jetson():
    profile = hardware_from_profile("jetson-orin-nano-super-8gb")
    assert profile.unified_memory is True
    assert profile.accelerator_memory_gb == profile.ram_available_gb == 6.4
    assert profile.matched_profile == "jetson-orin-nano-super-8gb"

def test_detected_machine_identity_does_not_collapse_to_profile(monkeypatch):
    from autonomyfit.benchmark import hardware_evidence_id
    from autonomyfit.models import HardwareProfile

    profile = hardware_from_profile("jetson-orin-nx-16gb")
    detected = HardwareProfile(
        platform="jetson", os_name="Linux", architecture="aarch64", cpu="Jetson",
        ram_total_gb=16, ram_available_gb=12, gpu="Jetson Orin NX",
        matched_profile="jetson-orin-nx-16gb", memory_topology="unified",
    )
    monkeypatch.setattr("autonomyfit.benchmark._machine_identity_hash", lambda: "machine-a")
    assert hardware_evidence_id(profile) == "jetson-orin-nx-16gb"
    assert hardware_evidence_id(detected).startswith("local-jetson-")
    assert hardware_evidence_id(detected) != hardware_evidence_id(profile)

def test_linux_cpu_brand_prefers_cpuinfo_model(monkeypatch):
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


@pytest.mark.parametrize(
    ("name", "memory", "expected"),
    [
        ("Apple M4", 16, "apple-m4-16gb"),
        ("Apple M4 Pro", 24, "apple-m4-pro-24gb"),
        ("Apple M4 Max", 36, "apple-m4-max-36gb"),
        ("Apple M4 Pro", 16, None),
        ("Apple M4 24GB", None, None),
        ("Apple M40", 16, None),
        ("NVIDIA", None, None),
        ("Intel", 32, None),
        ("", 16, None),
        ("---", None, None),
        ("Jetson Orin NX", 15.3, "jetson-orin-nx-16gb"),
        ("Jetson Orin NX 8GB", 8, None),
        ("NVIDIA T4", 15, "nvidia-t4-16gb"),
        ("Apple M4", float("nan"), None),
        ("Apple M4", float("inf"), None),
        ("Apple M4", 0, None),
    ],
)
def test_profile_match_requires_specific_model_and_compatible_capacity(name, memory, expected):
    assert match_hardware_profile(name, memory) == expected


def test_raspberry_pi_device_tree_does_not_imply_jetson(monkeypatch):
    from types import SimpleNamespace

    from autonomyfit.hardware import detect_hardware

    monkeypatch.setattr(
        "autonomyfit.hardware.Path.read_bytes",
        lambda self: b"Raspberry Pi 5 Model B Rev 1.0\x00",
    )
    monkeypatch.setattr("autonomyfit.hardware.Path.exists", lambda self: False)
    monkeypatch.setattr("autonomyfit.hardware.platform.system", lambda: "Linux")
    monkeypatch.setattr("autonomyfit.hardware.platform.machine", lambda: "aarch64")
    monkeypatch.setattr("autonomyfit.hardware._cpu_brand", lambda: "BCM2712")
    monkeypatch.setattr("autonomyfit.hardware._detect_runtimes", lambda _: ())
    monkeypatch.setattr("autonomyfit.hardware._openvino_devices", lambda: ())
    monkeypatch.setattr("autonomyfit.hardware._nvidia_smi", lambda: (None, None, None))
    monkeypatch.setattr(
        "autonomyfit.hardware.psutil.virtual_memory",
        lambda: SimpleNamespace(total=8 * 1024**3, available=6 * 1024**3),
    )
    profile = detect_hardware()
    assert profile.platform == "arm"
    assert profile.accelerator_type == "cpu"
    assert profile.gpu is None
    assert profile.matched_profile == "raspberry-pi-5-8gb"


@pytest.mark.parametrize("model", ["NVIDIA Jetson Orin NX 16GB", "NVIDIA JETSON AGX ORIN"])
def test_jetson_device_tree_is_recognised(monkeypatch, model):
    from autonomyfit.hardware import _jetson_model

    monkeypatch.setattr("autonomyfit.hardware._device_tree_model", lambda: model)
    assert _jetson_model() == model


def test_tegra_family_alone_does_not_establish_jetson(monkeypatch):
    from autonomyfit.hardware import _jetson_model

    monkeypatch.setattr("autonomyfit.hardware._device_tree_model", lambda: "NVIDIA Tegra234")
    assert _jetson_model() is None


def test_failed_hardware_probe_does_not_treat_error_as_device_name(monkeypatch):
    from types import SimpleNamespace

    from autonomyfit.hardware import _nvidia_smi

    monkeypatch.setattr("autonomyfit.hardware.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "autonomyfit.hardware.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=9, stdout="", stderr="NVIDIA-SMI has failed because the driver is unavailable"
        ),
    )
    assert _nvidia_smi() == (None, None, None)


def test_unified_memory_machine_identity_ignores_free_memory(monkeypatch):
    from dataclasses import replace

    from autonomyfit.benchmark import hardware_evidence_id

    monkeypatch.setattr("autonomyfit.benchmark._machine_identity_hash", lambda: "same-machine")
    profile = replace(hardware_from_profile("apple-m4-16gb"), os_name="Darwin")
    busy = replace(profile, ram_available_gb=4, accelerator_memory_gb=4)
    assert hardware_evidence_id(busy) == hardware_evidence_id(profile)
    upgraded = replace(profile, ram_total_gb=24)
    assert hardware_evidence_id(upgraded) != hardware_evidence_id(profile)


def test_discrete_memory_capacity_remains_part_of_machine_identity(monkeypatch):
    from dataclasses import replace

    from autonomyfit.benchmark import hardware_evidence_id

    monkeypatch.setattr("autonomyfit.benchmark._machine_identity_hash", lambda: "same-machine")
    profile = replace(hardware_from_profile("nvidia-t4-16gb"), os_name="Linux")
    replaced = replace(profile, accelerator_memory_gb=24)
    assert hardware_evidence_id(replaced) != hardware_evidence_id(profile)

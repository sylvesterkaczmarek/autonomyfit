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

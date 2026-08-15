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

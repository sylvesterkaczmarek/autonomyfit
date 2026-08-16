from autonomyfit.hardware import hardware_from_profile


def test_stage4_hardware_profiles_cover_heterogeneous_accelerators():
    intel = hardware_from_profile("intel-core-ultra-200v-32gb")
    amd = hardware_from_profile("amd-ryzen-ai-300-32gb")
    qualcomm = hardware_from_profile("qualcomm-snapdragon-x-elite-16gb")
    apple = hardware_from_profile("apple-m4-pro-24gb")
    assert intel.accelerator_type == "npu+gpu+cpu"
    assert amd.accelerator_type == "npu+gpu+cpu"
    assert qualcomm.accelerator_type == "npu+gpu+cpu"
    assert apple.memory_topology == "unified"


def test_execution_provider_profiles_are_explicitly_unverified_for_model_coverage():
    qualcomm = hardware_from_profile("qualcomm-snapdragon-x-elite-16gb")
    qnn = next(item for item in qualcomm.runtimes if item.name == "qnn")
    assert qnn.available is True
    assert qnn.verified is False
    assert "unverified" in (qnn.detail or "")


def test_nvidia_and_arm_profiles_keep_memory_topology_distinct():
    t4 = hardware_from_profile("nvidia-t4-16gb")
    pi = hardware_from_profile("raspberry-pi-5-8gb")
    assert t4.memory_topology == "discrete-vram"
    assert t4.accelerator_memory_gb == 16
    assert pi.memory_topology == "shared-system"

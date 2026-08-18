import inspect

import pytest

import autonomyfit


def test_public_api_surface_is_small_and_stable():
    assert autonomyfit.__all__ == [
        "DeploymentValidationError",
        "assess_deployment",
        "recommend",
    ]

    assessment_parameters = inspect.signature(autonomyfit.assess_deployment).parameters
    for advanced_cli_only in ("fetch", "convert", "benchmark", "artifact_url"):
        assert advanced_cli_only not in assessment_parameters


def test_recommend_uses_existing_offline_ranking_engine():
    recommendations = autonomyfit.recommend(
        task="detection",
        hardware_profile="nvidia-t4-16gb",
        objective="balanced",
        offline=True,
        limit=2,
    )

    assert recommendations
    assert len(recommendations) <= 2
    assert all(item.model.task == "detection" for item in recommendations)
    assert recommendations[0].objective_rank == 1


def test_assess_deployment_without_artifact_is_safe_and_non_executing():
    assessment = autonomyfit.assess_deployment(
        "yolo26n",
        hardware_profile="nvidia-t4-16gb",
        offline=True,
    )

    assert assessment["status"] == "artifact-selection-required"
    assert assessment["artifact"] is None
    assert assessment["model"]["id"] == "yolo26n"
    assert assessment["benchmark"] is None
    assert assessment["conversion"] is None


def test_recommend_rejects_invalid_limit():
    with pytest.raises(ValueError, match="positive integer"):
        autonomyfit.recommend(limit=0, offline=True)

import pytest

from autonomyfit.models import Constraints


@pytest.mark.parametrize("name", [
    "min_fps", "max_latency_ms", "max_power_w", "min_accuracy",
    "max_memory_gb", "max_params_m", "min_confidence",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1, True, "5"])
def test_constraint_limits_reject_invalid_numbers(name, value):
    with pytest.raises(ValueError, match=name):
        Constraints(**{name: value})


def test_confidence_limit_cannot_exceed_percentage_range():
    with pytest.raises(ValueError, match="between 0 and 100"):
        Constraints(min_confidence=101)


def test_constraints_preserve_zero_boundaries():
    constraints = Constraints(min_fps=0, max_power_w=0, min_confidence=0, min_accuracy=0)
    assert constraints.max_power_w == 0


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_size_requires_positive_integer(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        Constraints(batch_size=batch_size)


@pytest.mark.parametrize("shapes", [None, {"input": [True]}, {"input": [0]}, {"input": [1.5]}])
def test_constraint_shapes_require_concrete_integer_dimensions(shapes):
    with pytest.raises(ValueError, match="input_shapes"):
        Constraints(input_shapes=shapes)


def test_unknown_objective_fails_at_the_constraint_boundary():
    with pytest.raises(ValueError, match="objective"):
        Constraints(objective="typo")

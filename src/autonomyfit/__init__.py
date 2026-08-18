"""AutonomyFit public API."""

from .api import DeploymentValidationError, assess_deployment, recommend

__version__ = "0.8.0"

__all__ = [
    "DeploymentValidationError",
    "assess_deployment",
    "recommend",
]

# Python API

AutonomyFit remains CLI-first, with a small Python API for embedding model recommendation and safe deployment assessment in other software.

## Recommend models

```python
from autonomyfit import recommend

recommendations = recommend(
    task="detection",
    hardware_profile="nvidia-t4-16gb",
    objective="latency",
    offline=True,
    limit=3,
)

for item in recommendations:
    print(item.model.id, item.verdict, item.confidence.score if item.confidence else None)
```

Omit `hardware_profile` to assess the current machine. The function uses the same evidence, constraint, confidence, Pareto-ranking, and registry logic as the CLI recommendation path.

Common constraints such as latency, throughput, power, accuracy, memory, parameter count, runtime, precision, and minimum confidence can be supplied as keyword arguments.

## Assess deployment compatibility

Assess a model before selecting an artifact:

```python
from autonomyfit import assess_deployment

assessment = assess_deployment(
    "yolo26n",
    hardware_profile="nvidia-t4-16gb",
    offline=True,
)

print(assessment["status"])
```

Or inspect a local artifact:

```python
assessment = assess_deployment(
    "yolo26n",
    artifact="./yolo26n.onnx",
    runtime="onnx",
    expected_sha256="EXPECTED_SHA256",
)
```

`assess_deployment()` uses the existing deployment validator and returns its structured assessment dictionary.

## Deliberate API boundary

The small Python API does not expose remote artifact acquisition, conversion, or benchmarking. Those workflows have larger trust, execution, and reproducibility surfaces and remain available through the `autonomyfit` CLI.

The public package surface is intentionally limited to:

- `recommend()`
- `assess_deployment()`
- `DeploymentValidationError`

Internal modules remain implementation details while AutonomyFit is pre-1.0.

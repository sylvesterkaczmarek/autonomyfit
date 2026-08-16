# Ranking and confidence

AutonomyFit 0.5 uses a staged decision process instead of one opaque weighted score.

## 1. Hard feasibility

The engine first evaluates conditions that can make a deployment infeasible: model/runtime compatibility, listed precision support, accelerator-memory screening, explicit parameter and memory ceilings, licence filters and any exact measured threshold failures. Candidates that fail hard feasibility cannot outrank feasible candidates.

Unknown performance does not become a pass. If latency, throughput or power is requested but exact applicable evidence is absent, the result is `BENCHMARK_REQUIRED`.

## 2. Conservative Pareto layers

Among candidates that remain feasible, AutonomyFit computes non-dominated layers over the quantities it can actually compare:

- latency: lower is better
- throughput: higher is better
- task accuracy/quality metric: direction defined by the metric
- power: lower is better
- deployment-memory screen: lower is better

A dominance statement is made only when both candidates have the same known objective set. This avoids claiming that candidate A dominates candidate B merely because B is missing a measurement.

`pareto_rank=0` is the frontier. Later layers are dominated by at least one candidate in an earlier layer under the available comparable evidence.

## 3. Objective-specific order

The `--objective` flag controls the order inside the feasibility/Pareto structure:

- `latency`: lowest applicable latency
- `throughput`: highest applicable throughput
- `accuracy`: best comparable task metric
- `power`: lowest scoped measured power
- `memory`: lowest deployment-memory screen
- `balanced`: normalized utility across known latency, throughput, task metric, power and memory

For a single objective, displayed scores are min-max normalized within the feasible candidate set. Missing objective data receives no objective score and is resolved through Pareto/confidence/deterministic tie-breaking.

`balanced` gives each known objective equal weight after normalization, then multiplies by objective coverage. It does not impute missing measurements.

## 4. Deterministic tie-breaking

The stable order is:

1. fit verdict class
2. Pareto layer
3. selected objective value or balanced utility
4. numeric confidence
5. canonical model ID

This makes JSON output reproducible for automation.

## Confidence

Confidence is a disclosed evidence-coverage score, not a probability of real-world success. Six components are normalized to 0-1 and averaged equally:

1. hardware-match exactness
2. runtime/precision match
3. evidence quality
4. evidence freshness
5. revision/artifact identity
6. requested-quantity coverage

The final value is reported on a 0-100 scale. Any unresolved requested constraint caps it at 55. Compatibility labels are derived from the numeric score only for concise human output:

- `HIGH`: 85-100
- `MEDIUM`: 60-84.9
- `LOW`: 35-59.9
- `UNKNOWN`: below 35

Every component is present in JSON output. A high confidence score does not override a failed hard constraint.

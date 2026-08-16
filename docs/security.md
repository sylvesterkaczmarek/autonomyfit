# Artifact and supply-chain security

AutonomyFit treats model deployment artifacts as software supply-chain inputs, not as harmless downloads.

## Default policy

- no model repository code execution during discovery or download
- no Hugging Face `trust_remote_code`
- exact immutable Hugging Face commit resolution before acquisition
- SHA-256 verification before managed-cache admission
- upstream LFS SHA-256 verification when exposed by the Hub API
- cache digest revalidation on every cache load
- path-traversal checks for ONNX external tensor locations
- no automatic loading of pickle-style PyTorch checkpoints
- no automatic deserialization of untrusted TensorRT engines
- explicit licence status in artifact/deployment reports
- non-standard/restricted/unknown licences block automatic acquisition unless acknowledged

## Remote code

A Hugging Face repository containing Python files or `auto_map` metadata is marked as remote-code-bearing. AutonomyFit can still inspect and, where appropriate, acquire a self-contained static ONNX/safetensors file without importing that code. The report records the warning.

If a deployment path fundamentally requires repository Python or custom model code, AutonomyFit refuses to claim a secure generic automated validation. The user must establish a trusted, isolated export process outside the automatic path and provide the resulting artifact with identity metadata.

## TensorRT

Serialized TensorRT engines contain runtime-executable compiled state. AutonomyFit therefore distinguishes locally built engines from external engines. External engines are not benchmarked unless the user explicitly marks the local artifact trusted.

The trust flag is not a malware scanner. It is a boundary marker for artifacts the user independently trusts.

## Provenance

A deployment report records the upstream model source, requested and resolved revision, licence, artifact digest, conversion toolchain and measurement environment. Registry trust remains separate and uses the existing signed, rollback-protected registry mechanism.

## What AutonomyFit does not claim

- that HTTPS alone proves artifact provenance
- that a valid hash proves an artifact is benign
- that conversion proves semantic or accuracy equivalence
- that an installed execution provider supports every graph operator
- that model licence metadata is legal advice
- that synthetic inference timings measure application end-to-end latency

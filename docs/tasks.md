# Task system

AutonomyFit 0.5 replaces the original two-task branch logic with a registry-driven task layer.

Canonical tasks are:

| ID | Scope |
|---|---|
| `detection` | object detection |
| `classification` | image classification |
| `segmentation` | semantic, instance or promptable segmentation |
| `pose` | pose/keypoint estimation |
| `depth` | monocular depth estimation |
| `ocr` | OCR pipelines and OCR-oriented models |
| `vlm` | compact vision-language and multimodal generation |
| `anomaly` | visual anomaly detection/localization |
| `asr` | automatic speech recognition |
| `embedding` | edge-oriented representation/embedding models |

Aliases such as `object-detection`, `pose-estimation`, `depth-estimation`, `speech-recognition` and `embeddings` normalize to the canonical IDs.

The task layer holds task identity and input/output modality semantics separately from the ranking engine. Adding a future task therefore requires a new task definition and registry entries rather than a new branch through scoring code. Robotic/VLA policy selection is a planned example, but is not declared supported until deployable model/runtime evidence is good enough.

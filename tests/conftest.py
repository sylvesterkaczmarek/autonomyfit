from pathlib import Path

import pytest


@pytest.fixture
def write_onnx():
    """Write a valid small graph, optionally with externally stored weights."""
    def write(
        path: Path, *, external: bool = False, location: str = "weights.data", value: float = 1.0
    ) -> Path:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        path.parent.mkdir(parents=True, exist_ok=True)
        graph = helper.make_graph(
            [helper.make_node("Add", ["input", "weight"], ["output"])],
            "test-artifact",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
            [numpy_helper.from_array(np.array([value], dtype=np.float32), name="weight")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        if external:
            (path.parent / location).parent.mkdir(parents=True, exist_ok=True)
            onnx.save_model(
                model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                location=location, size_threshold=0,
            )
        else:
            onnx.save_model(model, str(path))
        return path

    return write

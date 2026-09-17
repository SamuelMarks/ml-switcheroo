"""Test suite for NN and Shape operations in the StableHLO backend."""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode

OPS: List[Tuple[str, str]] = [
  # NN
  ("BatchNormGrad", "stablehlo.batch_norm_grad"),
  ("BatchNormInference", "stablehlo.batch_norm_inference"),
  ("BatchNormTraining", "stablehlo.batch_norm_training"),
  ("Convolution", "stablehlo.convolution"),
  ("DynamicConv", "stablehlo.dynamic_conv"),
  # Shape
  ("BitcastConvert", "stablehlo.bitcast_convert"),
  ("BroadcastInDim", "stablehlo.broadcast_in_dim"),
  ("Concatenate", "stablehlo.concatenate"),
  ("Convert", "stablehlo.convert"),
  ("DynamicBroadcastInDim", "stablehlo.dynamic_broadcast_in_dim"),
  ("DynamicIota", "stablehlo.dynamic_iota"),
  ("DynamicPad", "stablehlo.dynamic_pad"),
  ("DynamicReshape", "stablehlo.dynamic_reshape"),
  ("DynamicSlice", "stablehlo.dynamic_slice"),
  ("DynamicUpdateSlice", "stablehlo.dynamic_update_slice"),
  ("Gather", "stablehlo.gather"),
  ("DynamicGather", "stablehlo.dynamic_gather"),
  ("GetDimensionSize", "stablehlo.get_dimension_size"),
  ("GetTupleElement", "stablehlo.get_tuple_element"),
  ("Iota", "stablehlo.iota"),
  ("Pad", "stablehlo.pad"),
  ("Reshape", "stablehlo.reshape"),
  ("Reverse", "stablehlo.reverse"),
  ("Scatter", "stablehlo.scatter"),
  ("Slice", "stablehlo.slice"),
  ("Sort", "stablehlo.sort"),
  ("Transpose", "stablehlo.transpose"),
  ("Tuple", "stablehlo.tuple"),
]


class NNShapeSemanticsMock:
  """Mock semantics."""

  def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Get mock definition for kind.

    Args:
        kind (str): Kind string.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition tuple.
    """
    for abstract, stablehlo_api in OPS:
      if kind == abstract:
        return abstract, {"variants": {"stablehlo": {"api": stablehlo_api}}}
    return None


@pytest.mark.parametrize("abstract_name, expected_api", OPS)
def test_nn_shape_operations_backend(abstract_name: str, expected_api: str) -> None:
  """Test generating StableHLO via backend.

  Args:
      abstract_name (str): Abstract name.
      expected_api (str): Expected API string.
  """
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="op1", op_type=abstract_name),
    LogicalNode(id="out1", op_type="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="in1", target="op1"),
    LogicalEdge(source="op1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=NNShapeSemanticsMock())  # type: ignore
  code: str = backend.compile(graph)
  assert expected_api in code

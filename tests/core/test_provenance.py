"""Test suite for the Provenance module."""

import libcst as cst
import typing
from ml_switcheroo.core.graph import GraphExtractor


def extract(code: str) -> GraphExtractor:
  """Extracts ."""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)
  return extractor


def test_provenance_input_args() -> None:
  """Verifies the behavior of provenance input arguments."""
  code: str = "\ndef forward(self, x):\n    pass\n"
  ex = extract(code)
  assert "Input_x" in ex.node_map
  node = ex.node_map["Input_x"]
  assert isinstance(node, cst.Param)
  assert node.name.value == "x"


def test_provenance_layer_definition() -> None:
  """Verifies the behavior of provenance layer definition."""
  code: str = "\nclass Net:\n    def __init__(self):\n        self.conv = nn.Conv2d(1, 1, 1)\n"
  ex = extract(code)
  assert "conv" in ex.node_map
  node = ex.node_map["conv"]
  assert isinstance(node, cst.Assign)
  assert "nn.Conv2d" in cst.Module([]).code_for_node(node.value)


def test_provenance_functional_call() -> None:
  """Verifies the behavior of provenance functional call."""
  code: str = "\ndef forward(self, x):\n    return F.relu(x, inplace=True)\n"
  ex = extract(code)
  assert "func_relu" in ex.node_map
  node = ex.node_map["func_relu"]
  assert isinstance(node, cst.Call)
  assert typing.cast(cst.Name, typing.cast(cst.Attribute, node.func).attr).value == "relu"
  logical_node = ex.layer_registry["func_relu"]
  assert logical_node.metadata.get("inplace") == "True"


def test_provenance_script_constant() -> None:
  """Verifies the behavior of provenance script constant."""
  code: str = "x = 1"
  ex = extract(code)
  assert "Input_x" in ex.node_map
  node = ex.node_map["Input_x"]
  assert isinstance(node, cst.Assign)


def test_provenance_return_output() -> None:
  """Verifies the behavior of provenance return output."""
  code: str = "\ndef forward(self, x):\n    return x\n"
  ex = extract(code)
  assert "output" in ex.node_map
  node = ex.node_map["output"]
  assert isinstance(node, cst.Return)


def test_provenance_implicit_script_input() -> None:
  """Verifies the behavior of provenance implicit script input."""
  code: str = "x = op(img)"
  ex = extract(code)
  assert "Input_img" in ex.node_map
  node = ex.node_map["Input_img"]
  assert isinstance(node, cst.Arg)
  assert typing.cast(cst.Name, node.value).value == "img"

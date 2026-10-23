"""Test suite for the Provenance module."""

import libcst as _cst

from ml_switcheroo.core.graph import GraphExtractor

cst = _cst


def extract(code: str) -> GraphExtractor:
  """Execute extract helper logic.

  Args:
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `GraphExtractor` from extract.
  """
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)
  return extractor


def test_provenance_input_args() -> None:
  """Verifies the behavior of provenance input arguments."""
  code = "\ndef forward(self, x):\n    pass\n"
  ex = extract(code)
  assert "Input_x" in ex.node_map
  node = ex.node_map["Input_x"]
  assert isinstance(node, cst.Param)
  assert node.name.value == "x"


def test_provenance_layer_definition() -> None:
  """Verifies the behavior of provenance layer definition."""
  code = "\nclass Net:\n    def __init__(self):\n        self.conv = nn.Conv2d(1, 1, 1)\n"
  ex = extract(code)
  assert "conv" in ex.node_map
  node = ex.node_map["conv"]
  assert isinstance(node, cst.Assign)
  assert "nn.Conv2d" in cst.Module([]).code_for_node(node.value)


def test_provenance_functional_call() -> None:
  """Verifies the behavior of provenance functional call."""
  code = "\ndef forward(self, x):\n    return F.relu(x, inplace=True)\n"
  ex = extract(code)
  assert "func_relu" in ex.node_map
  node = ex.node_map["func_relu"]
  assert isinstance(node, cst.Call)
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": node}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "relu"}, "__getitem__")("v")
  logical_node = ex.layer_registry["func_relu"]
  assert logical_node.attributes.get("inplace") == "True"


def test_provenance_script_constant() -> None:
  """Verifies the behavior of provenance script constant."""
  code = "x = 1"
  ex = extract(code)
  assert "Input_x" in ex.node_map
  node = ex.node_map["Input_x"]
  assert isinstance(node, cst.Assign)


def test_provenance_return_output() -> None:
  """Verifies the behavior of provenance return output."""
  code = "\ndef forward(self, x):\n    return x\n"
  ex = extract(code)
  assert "output" in ex.node_map
  node = ex.node_map["output"]
  assert isinstance(node, cst.Return)


def test_provenance_implicit_script_input() -> None:
  """Verifies the behavior of provenance implicit script input."""
  code = "x = op(img)"
  ex = extract(code)
  assert "Input_img" in ex.node_map
  node = ex.node_map["Input_img"]
  assert isinstance(node, cst.Arg)
  assert cst.ensure_type(node.value, cst.Name).value == "img"

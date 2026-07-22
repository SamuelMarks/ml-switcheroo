"""Test suite for the Graph Synthesizer module."""

import ast
import pytest
import libcst as cst
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.backends.python import PythonBackend


@pytest.fixture
def synthesizer() -> PythonBackend:
  """Provides a mock synthesizer for testing."""
  return PythonBackend()


def validate_python(code: str) -> None:
  """Validates python."""
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_synthesize_simple_chain(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of synthesize simple chain."""
  g = LogicalGraph()
  g.nodes = [LogicalNode("x", "Input"), LogicalNode("conv1", "Conv2d"), LogicalNode("output", "Output")]
  code = synthesizer.generate(g, "SimpleNet")
  validate_python(code)
  assert "class SimpleNet(nn.Module):" in code
  assert "self.conv1 = nn.Conv2d()" in code
  assert "x = self.conv1(x)" in code
  assert "return x" in code


def test_synthesize_functional_mix(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of synthesize functional mix."""
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input"),
    LogicalNode("fc", "Linear"),
    LogicalNode("flat", "torch.flatten"),
    LogicalNode("out", "Output"),
  ]
  code = synthesizer.generate(g)
  validate_python(code)
  assert "self.fc = nn.Linear()" in code
  assert "self.flat" not in code
  init_block = code.split("__init__")[1].split("forward")[0]
  assert "torch.flatten" not in init_block
  assert "x = self.fc(x)" in code
  assert "x = torch.flatten(x)" in code


def test_synthesize_metadata_args(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of synthesize metadata arguments."""
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input"),
    LogicalNode("c1", "Conv2d", {"arg_0": "1", "arg_1": "32", "kernel_size": "3"}),
    LogicalNode("out", "Output"),
  ]
  code = synthesizer.generate(g)
  validate_python(code)
  assert "nn.Conv2d(1, 32, kernel_size=3)" in code


def test_synthesize_custom_input_name(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of synthesize custom input name."""
  g = LogicalGraph()
  g.nodes = [LogicalNode("in_node", "Input", {"name": "img"}), LogicalNode("l1", "Linear"), LogicalNode("out", "Output")]
  code = synthesizer.generate(g)
  validate_python(code)
  assert "def forward(self, img):" in code
  assert "img = self.l1(img)" in code
  assert "return img" in code


def test_context_preservation(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of context preservation."""
  original_source = '\nimport torch\nimport torch.nn as nn\n\nclass MyNet(nn.Module):\n    """My Docstring."""\n    def __init__(self):\n        super().__init__()\n        self.old_layer = nn.Linear(1, 1)\n\n    def forward(self, x):\n        return self.old_layer(x)\n\n    def validation_step(self, batch):\n        print("I should survive")\n'
  original_tree = cst.parse_module(original_source)
  g = LogicalGraph(nodes=[LogicalNode("x", "Input"), LogicalNode("new_conv", "Conv2d"), LogicalNode("out", "Output")])
  new_code = synthesizer.generate(g, class_name="MyNet", original_tree=original_tree)
  validate_python(new_code)
  assert '"""My Docstring."""' in new_code
  assert "def validation_step(self, batch):" in new_code
  assert 'print("I should survive")' in new_code
  assert "self.old_layer" not in new_code
  assert "self.new_conv = nn.Conv2d" in new_code


def test_missing_class_fallback(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of missing class fallback."""
  original_tree = cst.parse_module("class OtherClass: pass")
  g = LogicalGraph(nodes=[LogicalNode("x", "Input"), LogicalNode("l1", "Linear"), LogicalNode("out", "Output")])
  new_code = synthesizer.generate(g, class_name="MissingClass", original_tree=original_tree)
  validate_python(new_code)
  assert "class MissingClass(nn.Module):" in new_code
  assert "class OtherClass" not in new_code


def test_format_args_helper(synthesizer: PythonBackend) -> None:
  """Formats arguments helper."""
  meta = {"arg_1": "b", "arg_0": "a", "key": "val", "dropout": "0.5"}
  res = synthesizer._format_args_from_metadata(meta)
  assert res == "a, b, dropout=0.5, key=val"


def test_is_stateful_layer_helper(synthesizer: PythonBackend) -> None:
  """Checks if is stateful layer helper."""
  n1 = LogicalNode("f", "torch.flatten", {})
  assert synthesizer._is_stateful_layer(n1) is False
  n2 = LogicalNode("n", "Input", {})
  assert synthesizer._is_stateful_layer(n2) is False
  n3 = LogicalNode("l", "Linear", {})
  assert synthesizer._is_stateful_layer(n3) is True
  n4 = LogicalNode("c", "nn.Conv2d", {})
  assert synthesizer._is_stateful_layer(n4) is True


def test_return_insertion(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of return insertion."""
  g = LogicalGraph(nodes=[LogicalNode("x", "Input", {"name": "y"}), LogicalNode("l1", "Linear")], edges=[])
  code = synthesizer.generate(g)
  assert "return y" in code

"""Test suite for the Graph Synthesizer module."""

import ast

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


@pytest.fixture
def synthesizer() -> PythonBackend:
  """Execute synthesizer helper logic.

  Returns:
      Computed test result or fixture value.
  """
  return PythonBackend()


def validate_python(code: str) -> None:
  """Validates python.

  Args:
      code: Test argument or fixture.
  """
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_synthesize_simple_chain(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of synthesize simple chain.

  Args:
      synthesizer: Test argument or fixture.
  """
  g = LogicalGraph(
    nodes={
      "x": LogicalNode("x", "Input"),
      "conv1": LogicalNode("conv1", "Conv2d"),
      "output": LogicalNode("output", "Output"),
    }
  )
  code = synthesizer.generate(g, "SimpleNet")
  validate_python(code)
  assert "class SimpleNet(nn.Module):" in code
  assert "self.conv1 = nn.Conv2d()" in code
  assert "x = self.conv1(x)" in code
  assert "return x" in code


def test_synthesize_functional_mix(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of synthesize functional mix.

  Args:
      synthesizer: Test argument or fixture.
  """
  g = LogicalGraph(
    nodes={
      "x": LogicalNode("x", "Input"),
      "fc": LogicalNode("fc", "Linear"),
      "flat": LogicalNode("flat", "torch.flatten"),
      "out": LogicalNode("out", "Output"),
    }
  )
  code = synthesizer.generate(g)
  validate_python(code)
  assert "self.fc = nn.Linear()" in code
  assert "self.flat" not in code
  init_block = code.split("__init__")[1].split("forward")[0]
  assert "torch.flatten" not in init_block
  assert "x = self.fc(x)" in code
  assert "x = torch.flatten(x)" in code


def test_synthesize_metadata_args(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of synthesize metadata arguments.

  Args:
      synthesizer: Test argument or fixture.
  """
  g = LogicalGraph(
    nodes={
      "x": LogicalNode("x", "Input"),
      "c1": LogicalNode("c1", "Conv2d", attributes={"arg_0": "1", "arg_1": "32", "kernel_size": "3"}),
      "out": LogicalNode("out", "Output"),
    }
  )
  code = synthesizer.generate(g)
  validate_python(code)
  assert "nn.Conv2d(1, 32, kernel_size=3)" in code


def test_synthesize_custom_input_name(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of synthesize custom input name.

  Args:
      synthesizer: Test argument or fixture.
  """
  g = LogicalGraph(
    nodes={
      "in_node": LogicalNode("in_node", "Input", attributes={"name": "img"}),
      "l1": LogicalNode("l1", "Linear"),
      "out": LogicalNode("out", "Output"),
    }
  )
  code = synthesizer.generate(g)
  validate_python(code)
  assert "def forward(self, img):" in code
  assert "img = self.l1(img)" in code
  assert "return img" in code


def test_context_preservation(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of context preservation.

  Args:
      synthesizer: Test argument or fixture.
  """
  original_source = '\nimport torch\nimport torch.nn as nn\n\nclass MyNet(nn.Module):\n    """My Docstring."""\n    def __init__(self):\n        super().__init__()\n        self.old_layer = nn.Linear(1, 1)\n\n    def forward(self, x):\n        return self.old_layer(x)\n\n    def validation_step(self, batch):\n        print("I should survive")\n'
  original_tree = cst.parse_module(original_source)
  g = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("x", "Input"), LogicalNode("new_conv", "Conv2d"), LogicalNode("out", "Output")]}
  )
  new_code = synthesizer.generate(g, class_name="MyNet", original_tree=original_tree)
  validate_python(new_code)
  assert '"""My Docstring."""' in new_code
  assert "def validation_step(self, batch):" in new_code
  assert 'print("I should survive")' in new_code
  assert "self.old_layer" not in new_code
  assert "self.new_conv = nn.Conv2d" in new_code


def test_missing_class_fallback(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of missing class fallback.

  Args:
      synthesizer: Test argument or fixture.
  """
  original_tree = cst.parse_module("class OtherClass: pass")
  g = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("x", "Input"), LogicalNode("l1", "Linear"), LogicalNode("out", "Output")]}
  )
  new_code = synthesizer.generate(g, class_name="MissingClass", original_tree=original_tree)
  validate_python(new_code)
  assert "class MissingClass(nn.Module):" in new_code
  assert "class OtherClass" not in new_code


def test_format_args_helper() -> None:
  """Formats arguments helper."""
  from ml_switcheroo.core.compiler.backends.python_strategies.init import format_args_from_metadata

  meta = {"arg_1": "b", "arg_0": "a", "key": "val", "dropout": "0.5"}
  res = format_args_from_metadata(meta)
  assert res == "a, b, dropout=0.5, key=val"


def test_is_stateful_layer_helper(synthesizer: PythonBackend) -> None:
  """Checks if is stateful layer helper.

  Args:
      synthesizer: Test argument or fixture.
  """
  n1 = LogicalNode("f", "torch.flatten", attributes={})
  assert synthesizer._is_stateful_layer(n1) is False
  n2 = LogicalNode("n", "Input", attributes={})
  assert synthesizer._is_stateful_layer(n2) is False
  n3 = LogicalNode("l", "Linear", attributes={})
  assert synthesizer._is_stateful_layer(n3) is True
  n4 = LogicalNode("c", "nn.Conv2d", attributes={})
  assert synthesizer._is_stateful_layer(n4) is True


def test_return_insertion(synthesizer: PythonBackend) -> None:
  """Verifies the behavior of return insertion.

  Args:
      synthesizer: Test argument or fixture.
  """
  g = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("x", "Input", attributes={"name": "y"}), LogicalNode("l1", "Linear")]},
    edges=[],
  )
  code = synthesizer.generate(g)
  assert "return y" in code

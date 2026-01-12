"""
Tests for Python Backend (formerly Graph Synthesizer).

Verifies that LogicalGraphs are correctly converted into valid Python/PyTorch source code.
Covers:
1. Basic Layer chaining (Fresh Synthesis).
2. Mix of Stateful Layers and Functional Operations.
3. Metadata argument rendering.
4. Non-destructive AST patching (Context Preservation).
5. Internal helper logic.
"""

import ast
import pytest
import libcst as cst
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.compiler.backends.python import PythonBackend


@pytest.fixture
def synthesizer() -> PythonBackend:
  return PythonBackend()


def validate_python(code: str) -> None:
  """Ensures generated code is syntactically valid Python."""
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_synthesize_simple_chain(synthesizer: PythonBackend) -> None:
  """
  Scenario: Input -> Conv2d -> Output.
  Expectation: Fresh class generation with __init__ and forward.
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input"),
    LogicalNode("conv1", "Conv2d"),
    LogicalNode("output", "Output"),
  ]
  # Simple implicit linear ordering via list order for synthesized output

  code = synthesizer.generate(g, "SimpleNet")

  validate_python(code)

  assert "class SimpleNet(nn.Module):" in code
  assert "self.conv1 = nn.Conv2d()" in code
  assert "x = self.conv1(x)" in code
  assert "return x" in code


def test_synthesize_functional_mix(synthesizer: PythonBackend) -> None:
  """
  Scenario: Input -> Layer -> Functional -> Output.
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input"),
    LogicalNode("fc", "Linear"),
    # Functional Op
    LogicalNode("flat", "torch.flatten"),
    LogicalNode("out", "Output"),
  ]

  code = synthesizer.generate(g)
  validate_python(code)

  # Init should only have fc
  assert "self.fc = nn.Linear()" in code
  assert "self.flat" not in code
  # Check that functional op is not in Init block (rough heurisitc)
  init_block = code.split("__init__")[1].split("forward")[0]
  assert "torch.flatten" not in init_block

  # Forward should have both
  assert "x = self.fc(x)" in code
  assert "x = torch.flatten(x)" in code


def test_synthesize_metadata_args(synthesizer: PythonBackend) -> None:
  """
  Scenario: Nodes have argument metadata.
  Expectation: Arguments injected into init/call.
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input"),
    # Positional args simulated via "arg_N" keys
    LogicalNode("c1", "Conv2d", {"arg_0": "1", "arg_1": "32", "kernel_size": "3"}),
    LogicalNode("out", "Output"),
  ]

  code = synthesizer.generate(g)
  validate_python(code)

  # Check for: nn.Conv2d(1, 32, kernel_size=3)
  # Exact ordering of kwargs depends on dict iteration, but positionals come first in implementation
  assert "nn.Conv2d(1, 32, kernel_size=3)" in code


def test_synthesize_custom_input_name(synthesizer: PythonBackend) -> None:
  """
  Scenario: Input node has specific name 'img'.
  Expectation: forward argument is 'img'.
  """
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("in_node", "Input", {"name": "img"}),
    LogicalNode("l1", "Linear"),
    LogicalNode("out", "Output"),
  ]

  code = synthesizer.generate(g)
  validate_python(code)

  assert "def forward(self, img):" in code
  assert "img = self.l1(img)" in code
  assert "return img" in code


def test_context_preservation(synthesizer: PythonBackend) -> None:
  """
  Scenario: Patching an existing class while keeping other methods/docstrings.
  Expectation: __init__ and forward are replaced, but helper methods and docs remain.
  """
  original_source = """
import torch
import torch.nn as nn

class MyNet(nn.Module):
    \"\"\"My Docstring.\"\"\"
    def __init__(self):
        super().__init__()
        self.old_layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.old_layer(x)

    def validation_step(self, batch):
        print("I should survive")
"""
  original_tree = cst.parse_module(original_source)

  # New graph logic to inject
  g = LogicalGraph(
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("new_conv", "Conv2d"),
      LogicalNode("out", "Output"),
    ]
  )

  # Run Synthesizer in Patch Mode
  new_code = synthesizer.generate(g, class_name="MyNet", original_tree=original_tree)
  validate_python(new_code)

  # Verify Persistence
  assert '"""My Docstring."""' in new_code
  assert "def validation_step(self, batch):" in new_code
  assert 'print("I should survive")' in new_code

  # Verify Replacement
  assert "self.old_layer" not in new_code
  assert "self.new_conv = nn.Conv2d" in new_code


def test_missing_class_fallback(synthesizer: PythonBackend) -> None:
  """
  Scenario: Original tree provided, but target class name not found.
  Expectation: Fallback to fresh generation (or appending, depending on implementation).
  Current impl falls back to returning a fresh module string if patcher fails to return modified code?
  Actually, implementation creates new module if logic branches that way.
  """
  original_tree = cst.parse_module("class OtherClass: pass")

  g = LogicalGraph(
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("l1", "Linear"),
      LogicalNode("out", "Output"),
    ]
  )

  new_code = synthesizer.generate(g, class_name="MissingClass", original_tree=original_tree)
  validate_python(new_code)

  # Since class 'MissingClass' wasn't found in 'OtherClass',
  # the synthesizer falls through to fresh generation logic.
  assert "class MissingClass(nn.Module):" in new_code
  assert "class OtherClass" not in new_code  # Fresh generation replaces file currently


def test_format_args_helper(synthesizer: PythonBackend) -> None:
  """Test metadata formatting private helper directly."""
  meta = {"arg_1": "b", "arg_0": "a", "key": "val", "dropout": "0.5"}
  # Explicitly verify sorting: arg_0, arg_1, then alphabetical kwargs
  res = synthesizer._format_args_from_metadata(meta)
  assert res == "a, b, dropout=0.5, key=val"


def test_is_stateful_layer_helper(synthesizer: PythonBackend) -> None:
  """Test differentiation between layers and functional ops."""
  # Functional cases
  n1 = LogicalNode("f", "torch.flatten", {})
  assert synthesizer._is_stateful_layer(n1) is False

  n2 = LogicalNode("n", "Input", {})
  assert synthesizer._is_stateful_layer(n2) is False

  # Stateful cases
  n3 = LogicalNode("l", "Linear", {})
  assert synthesizer._is_stateful_layer(n3) is True

  n4 = LogicalNode("c", "nn.Conv2d", {})
  assert synthesizer._is_stateful_layer(n4) is True


def test_return_insertion(synthesizer: PythonBackend) -> None:
  """Ensure return statement is added if graph ends without explicit Output node logic."""
  # Graph with no Output node kind (stops at last computation)
  g = LogicalGraph(
    nodes=[LogicalNode("x", "Input", {"name": "y"}), LogicalNode("l1", "Linear")],
    edges=[],
  )
  code = synthesizer.generate(g)
  # The synthesizer should append a return for the last current_var
  assert "return y" in code

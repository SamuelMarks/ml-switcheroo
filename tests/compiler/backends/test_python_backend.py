"""
Tests for Python Backend.
"""

import pytest
import ast
import libcst as cst
from ml_switcheroo.compiler.backends.python import PythonBackend
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


@pytest.fixture
def backend() -> PythonBackend:
  return PythonBackend()


def validate_python(code: str) -> None:
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_compile_interface_implementation(backend: PythonBackend) -> None:
  g = LogicalGraph()
  res = backend.compile(g)
  assert isinstance(res, str)
  # LogicalGraph defaults name to "Model", so class will be "class Model" unless overwritten
  # backend.compile uses g.name if present.
  assert "class Model" in res


def test_synthesize_torch_chain(backend: PythonBackend) -> None:
  g = LogicalGraph(
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("conv1", "Conv2d"),
      LogicalNode("output", "Output"),
    ],
    edges=[LogicalEdge("x", "conv1"), LogicalEdge("conv1", "output")],
  )
  code = backend.generate(g, "SimpleNet")
  validate_python(code)
  assert "import torch" in code
  assert "class SimpleNet(nn.Module):" in code
  assert "self.conv1 = nn.Conv2d()" in code
  assert "return x" in code


def test_synthesize_flax_chain() -> None:
  backend = PythonBackend(framework="flax_nnx")
  g = LogicalGraph(
    nodes=[
      LogicalNode("x", "Input"),
      LogicalNode("fc", "Linear", {"out": "10"}),
    ]
  )
  code = backend.generate(g, "FlaxNet")
  validate_python(code)
  assert "class FlaxNet(nnx.Module):" in code
  assert "self.fc = nnx.Linear(out=10, rngs=rngs)" in code


def test_context_preservation(backend: PythonBackend) -> None:
  orig = "class MyNet(nn.Module): pass"
  tree = cst.parse_module(orig)
  g = LogicalGraph(nodes=[LogicalNode("x", "Input")])
  code = backend.generate(g, class_name="MyNet", original_tree=tree)
  validate_python(code)
  assert "class MyNet" in code

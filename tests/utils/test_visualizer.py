"""Test suite for the Visualizer module."""

import libcst as cst
from ml_switcheroo.utils.visualizer import MermaidGenerator


def test_visualizer_basic_flow():
  """Verifies the behavior of visualizer basic flow."""
  code = "x = 1"
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)
  assert "graph TD" in mermaid
  assert "classDef" in mermaid
  assert "::modNode" in mermaid
  assert "Assign (=)" in mermaid
  assert "::stmtNode" in mermaid
  assert "-->" in mermaid


def test_visualizer_function_def():
  """Verifies the behavior of visualizer function def."""
  code = "def f(a, b=2): pass"
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)
  assert "Def: f" in mermaid
  assert "::funcNode" in mermaid
  assert mermaid.count(":::funcNode") == 1


def test_visualizer_call_structure():
  """Verifies the behavior of visualizer call structure."""
  code = "fn(x, y=z)"
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)
  assert "Call" in mermaid
  assert "fn()" in mermaid
  assert "::callNode" in mermaid
  assert "arg=" in mermaid or "arg" in mermaid


def test_visualizer_truncated_labels():
  """Verifies the behavior of visualizer truncated labels."""
  long_str = "A" * 100
  code = f"x = '{long_str}'"
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)
  assert "..." in mermaid


def test_visualizer_escapes_quotes():
  """Verifies the behavior of visualizer escapes quotes."""
  code = 'x = "quote"'
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)
  assert "quote" in mermaid


def test_node_to_str_robustness():
  """Verifies the behavior of node to string robustness."""
  gen = MermaidGenerator()
  assert gen._node_to_str(cst.Name("x")) == "x"
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  assert gen._node_to_str(attr) == "a.b"
  assert gen._node_to_str(cst.Integer("1")) == "1"
  assert gen._node_to_str(cst.Float("1.5")) == "1.5"
  tup = cst.Tuple(elements=[])
  res = gen._node_to_str(tup)
  assert res == "()"

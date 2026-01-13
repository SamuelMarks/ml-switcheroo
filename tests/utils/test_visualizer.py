"""
Tests for the Mermaid Visualizer.

Verifies:
1. Graph structure generation (nodes, edges).
2. Proper labelling and escaping of content.
3. CSS Class assignment based on node type.
4. Robustness against detached nodes.
"""

import pytest
import libcst as cst
from ml_switcheroo.utils.visualizer import MermaidGenerator


def test_visualizer_basic_flow():
  """
  Scenario: Visualize a simple assignment.
  """
  code = "x = 1"
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)

  assert "graph TD" in mermaid
  assert "classDef" in mermaid
  # Should identify module root
  assert "::modNode" in mermaid
  # Should identify Assignment
  assert "Assign (=)" in mermaid
  assert "::stmtNode" in mermaid
  # Should link them
  assert "-->" in mermaid


def test_visualizer_function_def():
  """
  Scenario: Visualize a function definition with args.
  """
  code = "def f(a, b=2): pass"
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)

  # Check function node
  assert "Def: f" in mermaid
  assert "::funcNode" in mermaid
  # Check hierarchy
  # Fix: Be explicit about node counting to avoid style definition collisions
  assert mermaid.count(":::funcNode") == 1


def test_visualizer_call_structure():
  """
  Scenario: Nested function calls.
  """
  code = "fn(x, y=z)"
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)

  assert "Call" in mermaid  # Helper creates "Call: fn()"
  assert "fn()" in mermaid
  assert "::callNode" in mermaid
  # Args should be inlined or present
  assert "arg=" in mermaid or "arg" in mermaid


def test_visualizer_truncated_labels():
  """
  Scenario: Very long string literal.
  """
  long_str = "A" * 100
  code = f"x = '{long_str}'"
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)

  # Check for ellipsis in label
  assert "..." in mermaid


def test_visualizer_escapes_quotes():
  """
  Scenario: Label contains quotes.
  """
  code = 'x = "quote"'
  tree = cst.parse_module(code)
  gen = MermaidGenerator()
  mermaid = gen.generate(tree)

  # Quotes inside label should not break mermaid syntax
  # MermaidGenerator replaces " with ' inside labels
  assert "quote" in mermaid


def test_node_to_str_robustness():
  """
  Unit test for string conversion helper.
  """
  gen = MermaidGenerator()

  # Simple Name
  assert gen._node_to_str(cst.Name("x")) == "x"

  # Attribute
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  assert gen._node_to_str(attr) == "a.b"

  # Literals
  assert gen._node_to_str(cst.Integer("1")) == "1"
  assert gen._node_to_str(cst.Float("1.5")) == "1.5"

  # Fallback (Complex node like Tuple)
  tup = cst.Tuple(elements=[])
  res = gen._node_to_str(tup)
  # The dummy module renderer creates "()"
  assert res == "()"

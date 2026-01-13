"""
Tests for Recursive CST Literal Generation in FrameworkInjector.
"""

import pytest
import libcst as cst
from ml_switcheroo.tools.injector_fw import convert_to_cst_literal


def render_node(node: cst.CSTNode) -> str:
  """Helper to render standalone node source."""
  module = cst.parse_module("")
  return module.code_for_node(node)


def test_primitive_recursion():
  # Int/Float/Bool/Str/None
  assert render_node(convert_to_cst_literal(1)) == "1"
  assert render_node(convert_to_cst_literal(1.5)) == "1.5"
  assert render_node(convert_to_cst_literal(True)) == "True"
  assert render_node(convert_to_cst_literal(None)) == "None"
  assert render_node(convert_to_cst_literal("foo")) == '"foo"'


def test_list_recursion():
  # Simple list
  val = [1, 2, "a"]
  node = convert_to_cst_literal(val)
  code = render_node(node)
  assert code == '[1, 2, "a"]'


def test_tuple_recursion():
  # Nested tuple
  val = (1, (2, 3))
  node = convert_to_cst_literal(val)
  code = render_node(node)
  # Note: LibCST rendering might vary slighty on spacing, but structural match is key
  # We clean whitespace for assertion
  clean = code.replace(" ", "")
  assert clean == "(1,(2,3))"


def test_dict_recursion():
  # Dictionary with mixed types
  val = {"alpha": 0.5, "dims": (1, 2), "flag": True}
  node = convert_to_cst_literal(val)
  code = render_node(node)

  clean = code.replace(" ", "")
  assert '"alpha":0.5' in clean
  assert '"dims":(1,2)' in clean
  assert '"flag":True' in clean


def test_deep_nesting():
  # Test depth
  val = [{"a": [1, 2]}, (None,)]
  node = convert_to_cst_literal(val)
  code = render_node(node)
  clean = code.replace(" ", "")
  assert '[{"a":[1,2]},(None,)]' == clean

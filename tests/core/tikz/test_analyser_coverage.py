"""Tests for TikZ analyser coverage."""

import typing

import libcst as cst

from ml_switcheroo.core.compiler.ir import LogicalNode
from ml_switcheroo.core.tikz.analyser import GraphExtractor


def test_analyse_layer_def_not_self() -> None:
  """Docstring."""
  analyser = GraphExtractor()
  assign = typing.cast(
    cst.Assign, typing.cast(cst.SimpleStatementLine, cst.parse_statement("other.conv = nn.Conv2d()")).body[0]
  )
  analyser._analyze_layer_def(assign)
  assert "conv" not in analyser.layer_registry


def test_analyse_layer_def_not_call() -> None:
  """Docstring."""
  analyser = GraphExtractor()
  assign = typing.cast(cst.Assign, typing.cast(cst.SimpleStatementLine, cst.parse_statement("self.conv = 42")).body[0])
  analyser._analyze_layer_def(assign)
  assert "conv" not in analyser.layer_registry


def test_analyze_data_flow_not_call() -> None:
  """Docstring."""
  analyser = GraphExtractor()
  assign = typing.cast(cst.Assign, typing.cast(cst.SimpleStatementLine, cst.parse_statement("x = 42")).body[0])
  analyser._analyze_data_flow(assign)
  assert len(analyser.graph.edges) == 0


def test_resolve_layer_or_func_name_none() -> None:
  """Docstring."""
  analyser = GraphExtractor()
  # A lambda call or complex expression where get_full_name returns None
  expr = typing.cast(cst.BaseExpression, cst.parse_expression("(lambda x: x)"))
  assert analyser._resolve_layer_or_func_name(expr) is None


def test_analyze_call_expression_none() -> None:
  """Docstring."""
  analyser = GraphExtractor()
  call = typing.cast(cst.Call, cst.parse_expression("(lambda x: x)()"))
  analyser._analyze_call_expression(call, ["out"])
  assert len(analyser.graph.edges) == 0


def test_analyser_all_missing_branches() -> None:
  """Test remaining branches in GraphExtractor for 100% coverage."""
  code = """
class Model(nn.Module):
    CLASS_VAR = 1

    def helper(self):
        return

    def forward(self, x):
        y = F.relu(x)
        z = F.relu(y)
        return (lambda: None)()
"""
  analyser = GraphExtractor()
  module = cst.parse_module(code)
  module.visit(analyser)

  # Test return when output is already in layer_registry for both call and var
  ret_call = typing.cast(cst.Return, cst.parse_statement("return self.conv(x)").body[0])  # type: ignore
  analyser._in_forward = True
  analyser.layer_registry["conv"] = LogicalNode("conv", "Conv2d", {})
  analyser.visit_Return(ret_call)
  # Second time when 'output' is already in registry
  analyser.visit_Return(ret_call)

  ret_var = typing.cast(cst.Return, cst.parse_statement("return x").body[0])  # type: ignore
  analyser.provenance["x"] = "input_x"
  analyser.visit_Return(ret_var)
  analyser.visit_Return(ret_var)

  # Test _finalize_graph with empty registry
  empty_analyser = GraphExtractor()
  empty_analyser._finalize_graph()
  assert len(empty_analyser.graph.nodes) == 0

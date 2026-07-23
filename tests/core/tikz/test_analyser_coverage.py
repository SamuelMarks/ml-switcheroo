"""Tests for TikZ analyser coverage."""

import libcst as cst
from ml_switcheroo.core.tikz.analyser import GraphExtractor


def test_analyse_layer_def_not_self():
  """Test analyse layer def not self."""
  analyser = GraphExtractor()
  assign = cst.parse_statement("other.conv = nn.Conv2d()").body[0]
  analyser._analyze_layer_def(assign)
  assert "conv" not in analyser.layer_registry


def test_analyse_layer_def_not_call():
  """Test analyse layer def not call."""
  analyser = GraphExtractor()
  assign = cst.parse_statement("self.conv = 42").body[0]
  analyser._analyze_layer_def(assign)
  assert "conv" not in analyser.layer_registry


def test_analyze_data_flow_not_call():
  """Test analyze data flow not call."""
  analyser = GraphExtractor()
  assign = cst.parse_statement("x = 42").body[0]
  analyser._analyze_data_flow(assign)
  assert len(analyser.graph.edges) == 0


def test_resolve_layer_or_func_name_none():
  """Test resolve layer or func name none."""
  analyser = GraphExtractor()
  # A lambda call or complex expression where get_full_name returns None
  expr = cst.parse_expression("(lambda x: x)")
  assert analyser._resolve_layer_or_func_name(expr) is None


def test_analyze_call_expression_none():
  """Test analyze call expression none."""
  analyser = GraphExtractor()
  call = cst.parse_expression("(lambda x: x)()")
  analyser._analyze_call_expression(call, ["out"])
  assert len(analyser.graph.edges) == 0

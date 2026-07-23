"""Test suite for the Injector Plugin Missing module."""


def test_injector_plugin_missing():
  """Verifies the behavior of injector plugin missing."""
  import libcst as cst
  from ml_switcheroo.tools.injector_plugin import BodyExtractor, PluginGenerator
  from pathlib import Path

  extractor = BodyExtractor("foo")
  extractor.visit_FunctionDef(cst.FunctionDef(name=cst.Name("bar"), params=cst.Parameters(), body=cst.IndentedBlock([])))
  assert extractor.found is False
  _ = PluginGenerator(Path("."))


def test_injector_plugin_generate_body_logic():
  """Verifies the behavior of injector plugin generate body logic."""
  import libcst as cst
  from ml_switcheroo.tools.injector_plugin import PluginGenerator
  from ml_switcheroo.core.dsl import Rule, LogicOp
  from pathlib import Path

  gen = PluginGenerator(Path("."))
  stmts = gen._generate_cst_body_logic([Rule(if_arg="foo", op=LogicOp.GT, val=5, use_api="bar")])
  mod = cst.Module(body=stmts)
  res = mod.code
  assert "val_0 > 5" in res

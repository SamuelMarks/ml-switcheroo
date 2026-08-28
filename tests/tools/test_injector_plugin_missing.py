"""Test suite for the Injector Plugin Missing module."""

from typing import List


def test_injector_plugin_missing() -> None:
  """Verifies the behavior of injector plugin missing."""
  import libcst as cst
  from ml_switcheroo.tools.injector_plugin import BodyExtractor, PluginGenerator
  from pathlib import Path

  extractor: BodyExtractor = BodyExtractor("foo")
  extractor.visit_FunctionDef(cst.FunctionDef(name=cst.Name("bar"), params=cst.Parameters(), body=cst.IndentedBlock([])))
  assert getattr(extractor, "found") is False
  _: PluginGenerator = PluginGenerator(Path("."))


def test_injector_plugin_generate_body_logic() -> None:
  """Verifies the behavior of injector plugin generate body logic."""
  import libcst as cst
  from ml_switcheroo.tools.injector_plugin import PluginGenerator
  from ml_switcheroo.core.dsl import Rule, LogicOp
  from pathlib import Path

  gen: PluginGenerator = PluginGenerator(Path("."))
  stmts: List[cst.BaseStatement] = gen._generate_cst_body_logic([Rule(if_arg="foo", op=LogicOp.GT, val=5, use_api="bar")])
  mod: cst.Module = cst.Module(body=stmts)
  res: str = mod.code
  assert "val_0 > 5" in res

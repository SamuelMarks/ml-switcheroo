def test_injector_plugin_missing():
  import libcst as cst
  from ml_switcheroo.tools.injector_plugin import BodyExtractor, PluginGenerator
  from pathlib import Path

  # 90: return True
  extractor = BodyExtractor("foo")
  extractor.visit_FunctionDef(cst.FunctionDef(name=cst.Name("bar"), params=cst.Parameters(), body=cst.IndentedBlock([])))
  assert extractor.found is False

  gen = PluginGenerator(Path("."))

  # 180-181: SimpleStatementSuite
  body = cst.SimpleStatementSuite(body=[cst.Expr(value=cst.SimpleString('""')), cst.Pass()])
  res = gen._render_body_without_docstring(body)
  assert "pass" in res

  # 199: Empty stmts
  body2 = cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString('""'))])])
  res2 = gen._render_body_without_docstring(body2)
  assert "return node" in res2


def test_injector_plugin_generate_body_logic():
  from ml_switcheroo.tools.injector_plugin import PluginGenerator
  from ml_switcheroo.core.dsl import Rule, LogicOp
  from pathlib import Path

  gen = PluginGenerator(Path("."))

  # 303: GT
  res = gen._generate_body_logic([Rule(if_arg="foo", op=LogicOp.GT, is_val=5, then_set={}, use_api="bar")])
  assert "val_0 > 5" in res

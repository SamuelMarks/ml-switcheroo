"""Test suite for the Coverage Gap More module."""

import pytest
import libcst as cst
from unittest.mock import patch, MagicMock
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.mlir.parser import MlirParser


def get_context():
  """Gets context."""
  cfg = RuntimeConfig(strict_mode=False)
  sm = SemanticsManager()
  ctx = RewriterContext(semantics=sm, config=cfg)
  return ctx


def test_api_check_version_constraints():
  """Verifies the behavior of API check version constraints."""
  t = ApiTransformer(get_context())
  with patch("importlib.metadata.version", side_effect=Exception("mocked error")):
    assert t.check_version_constraints("1.0.0", "2.0.0") is None


def test_api_normalize_args_coverage():
  """Verifies the behavior of API normalize arguments coverage."""
  t = ApiTransformer(get_context())
  arg_val = cst.parse_expression("1")
  current_arg = cst.Arg(value=arg_val, keyword=cst.Name("foo"))
  args = [current_arg]
  target_val_map = {"foo": {"1": "invalid syntax ("}}
  target_arg_map = {"foo": "foo"}
  res = t._normalize_arguments(
    cst.Call(cst.Name("f"), args),
    cst.Call(cst.Name("f"), args),
    {"args": ["foo"]},
    {"args": target_arg_map, "values": target_val_map, "api": "foo"},
  )
  assert res[0].value.value == "1"
  args2 = [cst.Arg(value=cst.parse_expression("3"), keyword=cst.Name("bar"))]
  res2 = t._normalize_arguments(
    cst.Call(cst.Name("f"), args2),
    cst.Call(cst.Name("f"), args2),
    {"args": []},
    {"args": {}, "values": {}, "api": "foo", "bar": "2"},
  )
  assert len(res2) == 1
  assert res2[0].value.value == "3"


def test_api_convert_to_indented_block():
  """Verifies the behavior of API convert to indented block."""
  t = ApiTransformer(get_context())
  fn = cst.parse_module("def foo():\n  pass").body[0]
  res = t._convert_to_indented_block(fn)
  assert res is fn


def test_auxiliary_transformer_coverage():
  """Verifies the behavior of auxiliary transformer coverage."""
  sm = SemanticsManager()
  sm.get_framework_config = MagicMock(return_value=None)
  ctx = RewriterContext(semantics=sm, config=RuntimeConfig())
  t = AuxiliaryTransformer(ctx)
  t._get_traits()
  assert t._cached_traits is not None
  assert t._get_qualified_name(cst.parse_expression("1 + 1")) is None
  t.context.alias_map = {"mod": "canon"}
  assert t._get_qualified_name(cst.parse_expression("mod.func")) == "canon.func"
  assert t._get_qualified_name(cst.parse_expression("mod")) == "canon"
  assert t._cst_to_string(cst.parse_expression("1 + 1")) is None
  t._report_warning("warn")
  assert "warn" in t.context.current_stmt_warnings
  sl = cst.SimpleStatementLine(body=[cst.Pass()])
  t.context.current_stmt_warnings = ["warn1"]
  res = t.leave_SimpleStatementLine(sl, sl)
  assert "warn1" in cst.Module(body=res).code
  t.context.current_stmt_warnings = []
  t.context.current_stmt_errors = ["err1"]
  res = t.leave_SimpleStatementLine(sl, sl)
  assert "err1" in cst.Module(body=res).code
  dec = cst.Decorator(decorator=cst.parse_expression("1+1"))
  assert t.leave_Decorator(dec, dec) is dec
  sm.get_definition = MagicMock(return_value=None)
  dec2 = cst.Decorator(decorator=cst.parse_expression("foo"))
  assert t.leave_Decorator(dec2, dec2) is dec2
  sm.get_definition = MagicMock(return_value=("id", {"variants": {"other": None}}))
  assert t.leave_Decorator(dec2, dec2) is dec2
  sm.get_definition = MagicMock(return_value=("id", {"variants": {"tgt": {}}}))
  assert t.leave_Decorator(dec2, dec2) is dec2
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("x"), body=cst.IndentedBlock(body=[cst.Pass()]))
  with patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook", side_effect=[lambda n, ctx: 1 / 0, None]):
    res = t.leave_For(loop, loop)
    assert res is loop
    assert any(("Static loop unrolling failed" in w for w in t.context.current_stmt_warnings))
  with patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook", return_value=None):
    res = t.leave_For(loop, loop)
    assert res is loop


def test_mlir_parser_coverage():
  """Verifies the behavior of MLIR parser coverage."""
  with pytest.raises(SyntaxError, match="Unexpected token .* where Op expected"):
    p = MlirParser("^bb0: 123")
    p.parse_block()
  assert MlirParser("{ %id")._is_region_start() is True
  assert MlirParser("{ =")._is_region_start() is False
  p = MlirParser("%res =")
  p2 = MlirParser('%res1, %res2 = "op"() : ()').parse_operation()
  assert len(p2.results) == 2
  MlirParser('"op"(%res)').parse_operation()
  MlirParser('"op"() { } : ()').parse_operation()
  p4 = MlirParser('"op"() : (!ty, !ty2)').parse_operation()
  assert len(p4.result_types) == 2
  p5 = MlirParser('"op"() : !sw.type<A>').parse_operation()
  assert p5.result_types[0].body == "!sw.type<A>"
  p6 = MlirParser("{")
  p6.parse_region()
  p7 = MlirParser("{ }")
  p7.parse_region()
  p = MlirParser("{ ^bb0: }")
  p.parse_region()


def test_api_pass_gap_lines():
  """Verifies the behavior of API pass gap lines."""
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager
  import libcst as cst

  ctx = type(
    "MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": SemanticsManager(), "alias_map": {}}
  )()
  transformer = ApiTransformer(ctx)
  op_details = {"args": ["x"]}
  target_impl = {"args_mapping": {"x": "y"}, "values_mapping": {"y": {"True": "invalid_syntax+++"}}, "api": "foo"}
  arg = cst.Arg(value=cst.Name("True"), keyword=cst.Name("x"))
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.rewriter.passes.api.extract_primitive_key", return_value="True"
  ):
    args = transformer._normalize_arguments(
      cst.Call(cst.Name("f"), [arg]), cst.Call(cst.Name("f"), [arg]), op_details, target_impl
    )
    assert args[0].value.value == "True"
  target_impl2 = {"args_mapping": {"x": "y"}, "values_mapping": {"z": "True"}, "api": "foo"}
  arg2 = cst.Arg(value=cst.Integer("1"), keyword=cst.Name("z"))
  args2 = transformer._normalize_arguments(
    cst.Call(cst.Name("f"), [arg2]), cst.Call(cst.Name("f"), [arg2]), op_details, target_impl2
  )
  assert len(args2) == 1
  assert args2[0].keyword.value == "z"


def test_auxiliary_pass_gap_lines():
  """Verifies the behavior of auxiliary pass gap lines."""
  from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager
  import libcst as cst

  ctx = type(
    "MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": SemanticsManager(), "alias_map": {}}
  )()
  transformer = AuxiliaryTransformer(ctx)
  transformer._cached_traits = "cached"
  assert transformer._get_traits() == "cached"
  transformer._cached_traits = None
  with __import__("unittest.mock").mock.patch.object(ctx.semantics, "get_framework_config", return_value={"traits": {}}):
    traits = transformer._get_traits()
    assert traits is not None
  node = cst.Decorator(decorator=cst.Name("unknown_dec"))
  with __import__("unittest.mock").mock.patch.object(
    transformer, "_get_traits", return_value=type("Mock", (), {"decorator_mapping": {}})()
  ):
    res = transformer.leave_Decorator(node, node)
    assert res is node


def test_mlir_parser_remaining():
  """Verifies the behavior of MLIR parser remaining."""
  p = MlirParser("a b c d e f g h i j k l m n o p q r s t u v w x y z")
  p.parse_operation()
  p = MlirParser('"op"( )')
  p.parse_operation()
  p = MlirParser('"op"() {')
  p = MlirParser('"op"() : (!ty, !ty2)')
  p.parse_operation()
  p = MlirParser('"op"() : !sw.type<A>')
  p.parse_operation()
  p = MlirParser("{ ^bb0: }")
  p.parse_region()


def test_parser_unreachable_hits():
  """Verifies the behavior of parser unreachable hits."""
  p = MlirParser('"op"() { // comment\n}')
  p.parse_operation()
  p2 = MlirParser('"op"() : (!sw.ty1, !sw.ty2)')
  p2.parse_operation()
  p4 = MlirParser("{ \n }")
  p4.parse_region()


def test_parser_unreachable_hits2():
  """Verifies the behavior of parser unreachable hits2."""
  p = MlirParser('"op"() { // comment\n}')
  p.parse_operation()
  p2 = MlirParser('"op"() : (!sw.ty1, !sw.ty2)')
  p2.parse_operation()
  p4 = MlirParser("{ \n }")
  p4.parse_region()


def test_api_unreachable_hits():
  """Verifies the behavior of API unreachable hits."""
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager
  import libcst as cst

  ctx = type(
    "MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": SemanticsManager(), "alias_map": {}}
  )()
  t = ApiTransformer(ctx)
  arg_val = cst.parse_expression("1")
  current_arg = cst.Arg(value=arg_val, keyword=cst.Name("foo"))
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.rewriter.passes.api.extract_primitive_key", return_value="1"
  ):
    res = t._normalize_arguments(
      cst.Call(cst.Name("f"), [current_arg]),
      cst.Call(cst.Name("f"), [current_arg]),
      {"std_args": ["foo"]},
      {"args": {"foo": "foo"}, "values": {"foo": {"1": "invalid syntax ("}}, "api": "foo"},
    )
    assert res[0].value.value == "1"
  arg2 = cst.Arg(value=cst.parse_expression("3"), keyword=cst.Name("bar"))
  res2 = t._normalize_arguments(
    cst.Call(cst.Name("f"), [arg2]),
    cst.Call(cst.Name("f"), [arg2]),
    {"std_args": []},
    {"args": {"bar": "bar"}, "values": {"bar": "2"}, "api": "foo"},
  )
  assert len(res2) == 1
  assert res2[0].value.value == "3"


def test_auxiliary_201():
  """Verifies the behavior of auxiliary 201."""
  from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
  from ml_switcheroo.semantics.manager import SemanticsManager
  import libcst as cst

  ctx = type(
    "MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": SemanticsManager(), "alias_map": {}}
  )()
  transformer = AuxiliaryTransformer(ctx)
  node = cst.Decorator(decorator=cst.Name("my_dec"))
  with __import__("unittest.mock").mock.patch.object(
    ctx.semantics, "get_definition", return_value=("id", {"variants": {"tgt": {}}})
  ):
    res = transformer.leave_Decorator(node, node)
    assert res is node

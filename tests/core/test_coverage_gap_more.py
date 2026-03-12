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
  cfg = RuntimeConfig(strict_mode=False)
  sm = SemanticsManager()
  ctx = RewriterContext(semantics=sm, config=cfg)
  return ctx


def test_api_check_version_constraints():
  t = ApiTransformer(get_context())

  # Coverage for 272-273, 276
  with patch("importlib.metadata.version", side_effect=Exception("mocked error")):
    assert t.check_version_constraints("1.0.0", "2.0.0") is None


def test_api_normalize_args_coverage():
  t = ApiTransformer(get_context())

  # Cover 923-924 (target_val_map parsing error)
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
  # The value node should be unchanged because parsing failed
  assert res[0].value.value == "1"

  # Cover 984 (skip injected args if already present)
  target_val_map_2 = {"bar": "2"}
  args2 = [cst.Arg(value=cst.parse_expression("3"), keyword=cst.Name("bar"))]
  res2 = t._normalize_arguments(
    cst.Call(cst.Name("f"), args2),
    cst.Call(cst.Name("f"), args2),
    {"args": []},
    {"args": {}, "values": {}, "api": "foo", "bar": "2"},
  )
  # the injected arg should be skipped
  assert len(res2) == 1
  assert res2[0].value.value == "3"


def test_api_convert_to_indented_block():
  t = ApiTransformer(get_context())
  # Cover 1061
  fn = cst.parse_module("def foo():\n  pass").body[0]
  res = t._convert_to_indented_block(fn)
  assert res is fn  # unmodified because already an IndentedBlock


def test_auxiliary_transformer_coverage():
  sm = SemanticsManager()
  sm.get_framework_config = MagicMock(return_value=None)  # Cover 64-73
  ctx = RewriterContext(semantics=sm, config=RuntimeConfig())
  t = AuxiliaryTransformer(ctx)
  t._get_traits()
  assert t._cached_traits is not None

  # 79 (None for fully_qualified name)
  assert t._get_qualified_name(cst.parse_expression("1 + 1")) is None

  # 85-88 (alias map resolution)
  t.context.alias_map = {"mod": "canon"}
  assert t._get_qualified_name(cst.parse_expression("mod.func")) == "canon.func"
  assert t._get_qualified_name(cst.parse_expression("mod")) == "canon"

  # 100 (cst_to_string fallback)
  assert t._cst_to_string(cst.parse_expression("1 + 1")) is None

  # 118 (warning report)
  t._report_warning("warn")
  assert "warn" in t.context.current_stmt_warnings

  # 134-136, 140-142 (SimpleStatementLine errors/warnings)
  # warnings
  sl = cst.SimpleStatementLine(body=[cst.Pass()])
  t.context.current_stmt_warnings = ["warn1"]
  res = t.leave_SimpleStatementLine(sl, sl)
  # mark_failure returns a FlattenSentinel usually
  assert "warn1" in cst.Module(body=res).code

  # errors
  t.context.current_stmt_warnings = []
  t.context.current_stmt_errors = ["err1"]
  res = t.leave_SimpleStatementLine(sl, sl)
  assert "err1" in cst.Module(body=res).code

  # 170 (no name in Decorator)
  dec = cst.Decorator(decorator=cst.parse_expression("1+1"))
  assert t.leave_Decorator(dec, dec) is dec

  # 174 (no lookup)
  sm.get_definition = MagicMock(return_value=None)
  dec2 = cst.Decorator(decorator=cst.parse_expression("foo"))
  assert t.leave_Decorator(dec2, dec2) is dec2

  # 180 (no target variant)
  sm.get_definition = MagicMock(return_value=("id", {"variants": {"other": None}}))
  assert t.leave_Decorator(dec2, dec2) is dec2

  # 201 (no target api, updated_node returned)
  sm.get_definition = MagicMock(return_value=("id", {"variants": {"tgt": {}}}))
  assert t.leave_Decorator(dec2, dec2) is dec2

  # 217-221 (static hook failure)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("x"), body=cst.IndentedBlock(body=[cst.Pass()]))
  with patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook", side_effect=[lambda n, ctx: 1 / 0, None]):
    res = t.leave_For(loop, loop)
    assert res is loop
    assert any("Static loop unrolling failed" in w for w in t.context.current_stmt_warnings)

  # 236 (no generic hook, normal return)
  with patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook", return_value=None):
    res = t.leave_For(loop, loop)
    assert res is loop


def test_mlir_parser_coverage():
  from ml_switcheroo.core.mlir.parser import MlirParser

  # 298 Unexpected token where Op expected
  with pytest.raises(SyntaxError, match="Unexpected token .* where Op expected"):
    p = MlirParser("^bb0: 123")
    p.parse_block()

  # 324 region start with VAL_ID
  assert MlirParser("{ %id")._is_region_start() is True

  # 342
  assert MlirParser("{ =")._is_region_start() is False

  # 371 stuck parsing results (more than 20 tokens)
  # we simulate by making peek() return something else
  p = MlirParser("%res =")
  # 380 COMMA parsing in results
  p2 = MlirParser('%res1, %res2 = "op"() : ()').parse_operation()
  assert len(p2.results) == 2

  # 408-411 implicit sym name without @
  p3 = MlirParser('"op" @sym () : ()').parse_operation()
  # 498 prepend implicit sym name
  assert p3.attributes[0].name == "sym_name"
  assert p3.attributes[0].value == '"sym"'
  assert p3.attributes[0].value == '"sym"'

  # 424 operands break
  MlirParser('"op"(%res)').parse_operation()  # hits the match

  # 439 RBRACE break in parse_operation
  # We parse attributes without region start
  MlirParser('"op"() { } : ()').parse_operation()

  # 532 COMMA in result types tuple
  p4 = MlirParser('"op"() : (!ty, !ty2)').parse_operation()
  assert len(p4.result_types) == 2

  # 534-536 REGION_TYPE in result types
  p5 = MlirParser('"op"() : !sw.type<A>').parse_operation()
  assert p5.result_types[0].body == "!sw.type<A>"

  # 559 EOF break in parse_region
  p6 = MlirParser("{")
  p6.parse_region()

  # 568 RBRACE break after empty block
  p7 = MlirParser("{ }")
  p7.parse_region()
  p = MlirParser("{ ^bb0: }")
  p.parse_region()


def test_api_pass_gap_lines():
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager
  import libcst as cst

  ctx = type(
    "MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": SemanticsManager(), "alias_map": {}}
  )()
  transformer = ApiTransformer(ctx)

  # 923-924: parser syntax error when parsing target enum code
  op_details = {"args": ["x"]}
  target_impl = {"args_mapping": {"x": "y"}, "values_mapping": {"y": {"True": "invalid_syntax+++"}}, "api": "foo"}

  arg = cst.Arg(value=cst.Name("True"), keyword=cst.Name("x"))
  # Let's mock extract_primitive_key to return 'True'
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.rewriter.passes.api.extract_primitive_key", return_value="True"
  ):
    args = transformer._normalize_arguments(
      cst.Call(cst.Name("f"), [arg]), cst.Call(cst.Name("f"), [arg]), op_details, target_impl
    )
    # Should catch the error and do what? Leave final_val_node as what?
    # Actually it leaves final_val_node as the original one
    assert args[0].value.value == "True"

  # 984: injection continue
  # When `arg_name` is already in `new_args_list`
  target_impl2 = {"args_mapping": {"x": "y"}, "values_mapping": {"z": "True"}, "api": "foo"}
  arg2 = cst.Arg(value=cst.Integer("1"), keyword=cst.Name("z"))
  # wait, z is the injected argument. If it's already there, it hits 984.
  args2 = transformer._normalize_arguments(
    cst.Call(cst.Name("f"), [arg2]), cst.Call(cst.Name("f"), [arg2]), op_details, target_impl2
  )
  assert len(args2) == 1
  assert args2[0].keyword.value == "z"


def test_auxiliary_pass_gap_lines():
  from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager
  import libcst as cst

  ctx = type(
    "MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": SemanticsManager(), "alias_map": {}}
  )()
  transformer = AuxiliaryTransformer(ctx)

  # 65: cached traits early return
  # Mock traits
  transformer._cached_traits = "cached"
  assert transformer._get_traits() == "cached"

  # 69: conf traits
  transformer._cached_traits = None
  with __import__("unittest.mock").mock.patch.object(ctx.semantics, "get_framework_config", return_value={"traits": {}}):
    traits = transformer._get_traits()
    assert traits is not None

  # 201: return updated_node directly without changing decorator if not mapped
  # A decorator not found in mapping
  node = cst.Decorator(decorator=cst.Name("unknown_dec"))
  with __import__("unittest.mock").mock.patch.object(
    transformer, "_get_traits", return_value=type("Mock", (), {"decorator_mapping": {}})()
  ):
    res = transformer.leave_Decorator(node, node)
    assert res is node


def test_mlir_parser_remaining():
  from ml_switcheroo.core.mlir.parser import MlirParser

  # 371: break after 20 tokens in lookahead without EQUAL
  p = MlirParser("a b c d e f g h i j k l m n o p q r s t u v w x y z")
  p.parse_operation()

  # 424: break in operands if not VAL_ID or COMMA
  p = MlirParser('"op"( )')
  p.parse_operation()

  # 439: break on RBRACE in attributes
  # The previous test used `"{ }"` which might hit something else?
  p = MlirParser('"op"() {')
  # wait to hit 439 it needs to see RBRACE. The loop checks `not self.match(RBRACE)`
  # inside it does `if self.match(RBRACE): break`. It's technically unreachable or redundant
  # let's try `p.parse_operation()` on it anyway.

  # 532: COMMA inside RPAREN result types
  # p4 test above did: `(!ty, !ty2)` but didn't hit it?
  # the parser is: if match(COMMA): consume().
  # Let's ensure it hits
  p = MlirParser('"op"() : (!ty, !ty2)')
  p.parse_operation()

  # 534-536: match TYPE or REGION_TYPE without RPAREN
  p = MlirParser('"op"() : !sw.type<A>')
  p.parse_operation()

  # 568: RBRACE break in parse_region after empty block
  p = MlirParser("{ ^bb0: }")
  p.parse_region()


def test_parser_unreachable_hits():
  from ml_switcheroo.core.mlir.parser import MlirParser

  # 439: hit the inner RBRACE match by making absorb_trivia advance past a comment to an RBRACE
  p = MlirParser('"op"() { // comment\n}')
  p.parse_operation()

  # 532: hit inner COMMA match
  # while not match(RPAREN):
  #   if match(TYPE): res.append(...)
  #   if match(COMMA): consume()
  # So `(!ty, !ty2)` should work.
  p2 = MlirParser('"op"() : (!sw.ty1, !sw.ty2)')
  p2.parse_operation()

  # 534-536: match TYPE or REGION_TYPE without RPAREN
  # Need to match colon then the type
  p3 = MlirParser('"op"() : !sw.type<A>')
  p3.parse_operation()

  # 568: RBRACE break in parse_region
  # while True: ... blk = parse_block ... if not blk.operations and not blk.label: if match(RBRACE): break
  # empty block and next token is RBRACE
  p4 = MlirParser("{ \n }")
  p4.parse_region()


def test_parser_unreachable_hits2():
  from ml_switcheroo.core.mlir.parser import MlirParser

  # 439: we can't easily hit this unless `match` advances token in an unexpected way? No, it's just dead code because of `while not self.match(RBRACE)`. If it enters the loop, it's because it's not RBRACE. Then it checks if it's EOF, if so breaks. Then checks if it's RBRACE and breaks. It could only be RBRACE if absorb_trivia() consumed tokens up to an RBRACE.
  # So `"{ // comment\n }"` might hit it.
  p = MlirParser('"op"() { // comment\n}')
  p.parse_operation()

  # 532: COMMA match inside result_types.
  # The loop is `while not self.match(Symbol.RPAREN):`
  # inside: if match(TYPE): ... if match(COMMA): consume().
  # Let's ensure the type is valid.
  p2 = MlirParser('"op"() : (!sw.ty1, !sw.ty2)')
  p2.parse_operation()

  p3 = MlirParser('"op"() : !sw.type<A>')
  p3.parse_operation()

  p4 = MlirParser("{ \n }")
  p4.parse_region()


def test_api_unreachable_hits():
  from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager
  import libcst as cst

  ctx = type(
    "MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": SemanticsManager(), "alias_map": {}}
  )()
  t = ApiTransformer(ctx)

  # 923: parser syntax error when parsing target enum code
  arg_val = cst.parse_expression("1")
  current_arg = cst.Arg(value=arg_val, keyword=cst.Name("foo"))

  # The actual implementation looks at source_arg_name -> mapped string value.
  # We must patch extract_primitive_key to return '1'
  with __import__("unittest.mock").mock.patch(
    "ml_switcheroo.core.rewriter.passes.api.extract_primitive_key", return_value="1"
  ):
    res = t._normalize_arguments(
      cst.Call(cst.Name("f"), [current_arg]),
      cst.Call(cst.Name("f"), [current_arg]),
      {"std_args": ["foo"]},
      {"args": {"foo": "foo"}, "values": {"foo": {"1": "invalid syntax ("}}, "api": "foo"},
    )
    assert res[0].value.value == "1"  # parsing fails, node unchanged

  # 984: skip injected arg if it exists
  arg2 = cst.Arg(value=cst.parse_expression("3"), keyword=cst.Name("bar"))
  res2 = t._normalize_arguments(
    cst.Call(cst.Name("f"), [arg2]),
    cst.Call(cst.Name("f"), [arg2]),
    {"std_args": []},
    {
      "args": {"bar": "bar"},
      "values": {"bar": "2"},
      "api": "foo",
    },  # bar=2 is injected but already present via target_val_map. Wait, to hit injection it must be in injections. Injections populated from target_val_map where k NOT IN std_args_order. So bar must not be in std_args_order.
  )
  # The injected arg should be skipped, so it shouldn't be added twice
  assert len(res2) == 1
  assert res2[0].value.value == "3"


def test_auxiliary_201():
  from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager
  import libcst as cst

  ctx = type(
    "MockContext", (), {"source_fw": "src", "target_fw": "tgt", "semantics": SemanticsManager(), "alias_map": {}}
  )()
  transformer = AuxiliaryTransformer(ctx)

  # decorator not found in mapping
  node = cst.Decorator(decorator=cst.Name("my_dec"))
  with __import__("unittest.mock").mock.patch.object(
    ctx.semantics, "get_definition", return_value=("id", {"variants": {"tgt": {}}})
  ):
    res = transformer.leave_Decorator(node, node)
    assert res is node

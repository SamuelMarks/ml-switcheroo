"""Test suite for the Api module."""

import typing
from unittest.mock import MagicMock, PropertyMock, patch

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits
from tests.conftest import TestRewriter


class MockSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, typing.Any] = {}
    self._validation_status: dict[str, typing.Any] = {}
    self._inject("abs", ["x"], {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}})
    self._inject("add_", ["x", "y"], {"torch": {"api": "torch.Tensor.add_"}, "jax": {"requires_plugin": "mock_unroll"}})
    self._inject("unsupported", [], {"torch": {"api": "torch.bad"}})
    self.framework_configs["jax"] = {"alias": {"module": "jax.numpy", "name": "jnp"}}

  def _inject(self, name: str, args: list[str], variants: dict[str, typing.Any]) -> None:
    """Mock implementation of  inject."""
    self.data[name] = {"std_args": args, "variants": variants}
    for _, v in variants.items():
      if "api" in v:
        self._reverse_index[v["api"]] = (name, self.data[name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> typing.Any:
    """Mock implementation of resolve variant."""
    return self.data.get(aid, {}).get("variants", {}).get(fw)

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified."""
    return True

  def get_framework_config(self, fw: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(fw, {})


@pytest.fixture
def run_pass() -> typing.Callable[[str], str]:
  """Docstring."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  semantics.framework_configs["torch"] = {"traits": {"functional_execution_method": None}}
  rewriter = TestRewriter(semantics, config)

  def _transform(code: str) -> str:
    """Helper to  transform."""
    tree = cst.parse_module(code)
    return typing.cast(str, rewriter.convert(tree).code)

  return _transform


def test_api_call_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of API call rewrite."""
  code = "y = torch.abs(x)"
  res: str = run_pass(code)
  assert "jnp.abs(x)" in res


def test_missing_mapping_strict_failure(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of missing mapping strict successfully handling failure."""
  code = "torch.bad()"
  res: str = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "No mapping available" in res


def test_assignment_unwrapping_passthrough(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of assignment unwrapping passthrough."""
  code = "res = layer(x)"
  res: str = run_pass(code)
  assert "layer(x)" in res
  code2 = "res = layer.apply(v, x)"
  res2: str = run_pass(code2)
  assert "layer.apply" in res2


def test_arg_normalization_logic(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of argument normalization logic."""
  mgr = MagicMock(spec=SemanticsManager)
  op_def: dict[str, typing.Any] = {
    "std_args": ["x", "axis"],
    "variants": {"torch": {"api": "torch.sum", "args": {"axis": "dim"}}, "jax": {"api": "jnp.sum"}},
  }
  mgr.get_definition.return_value = ("Sum", op_def)
  mgr.resolve_variant.return_value = op_def["variants"]["jax"]
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  conf = RuntimeConfig(source_framework="torch", target_framework="jax")
  rewriter = TestRewriter(mgr, conf)
  code = "s = torch.sum(x, dim=1)"
  tree = cst.parse_module(code)
  res: str = typing.cast(str, rewriter.convert(tree).code)
  assert "jnp.sum(x, axis=1)" in res


# --- Merged from test_api_extra2.py ---


class DummySemantics(SemanticsManager):
  """Docstring."""

  framework_configs: dict[str, typing.Any] = {}

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    self.configs: dict[str, typing.Any] = {}
    self.definitions: dict[str, typing.Any] = {}
    self.variants: dict[tuple[str, str], typing.Any] = {}
    self.verified = True
    self._key_origins: dict[str, str] = {}
    self.framework_configs = self.configs

  def get_framework_config(self, fw: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.configs.get(fw, {})

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self.definitions.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock implementation of resolve variant."""
    return self.variants.get((abstract_id, fw))

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified."""
    return self.verified


def get_transformer() -> tuple[ApiTransformer, DummySemantics, RewriterContext]:
  """Gets transformer."""
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True, source_flavour="torch.nn")
  ctx = RewriterContext(semantics, config)
  ctx.hook_context = type(
    "MockHook", (), {"preamble_stmts_mock": [], "inject_preamble": lambda self, s: self.preamble_stmts_mock.append(s)}
  )()  # type: ignore
  transformer = ApiTransformer(ctx)
  return (transformer, semantics, ctx)


def test_api_traits_fallback() -> None:
  """Verifies the behavior of API traits fallback."""
  (t, s, c) = get_transformer()
  s.configs["jax"] = {}
  traits = t._get_target_traits()
  assert isinstance(traits, StructuralTraits)


def test_cst_to_string_fallback() -> None:
  """Verifies the behavior of cst to string fallback."""
  (t, _, _) = get_transformer()
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert t._cst_to_string(node) == "Add"
  node2 = cst.Integer("1")
  assert t._cst_to_string(node2) is None
  assert t._get_qualified_name(node2) is None


def test_module_bases_object() -> None:
  """Verifies the behavior of module bases object."""
  (t, s, _) = get_transformer()

  class DummyTraits:
    """Docstring."""

    module_base = "MyModule"

  s.configs["torch"] = {"traits": DummyTraits()}
  assert t._is_framework_base("MyModule") is True


def test_module_preamble_exception() -> None:
  """Verifies the behavior of module preamble correctly handling an exception."""
  (t, _, c) = get_transformer()
  c.module_preamble.append("invalid python syntax +++")
  mod = cst.Module(body=[])
  res: typing.Any = t.leave_Module(mod, mod)
  assert len(res.body) == 0


def test_classdef_raw_fallback() -> None:
  """Verifies the behavior of classdef raw fallback."""
  (t, s, c) = get_transformer()
  t._known_module_bases = {"MyModule"}

  def mock_get_qualified_name(node: typing.Any) -> typing.Optional[str]:
    """Docstring."""
    return None

  def mock_cst_to_string(node: typing.Any) -> typing.Optional[str]:
    """Docstring."""
    if isinstance(node, cst.Name) and node.value == "MyModule":
      return "MyModule"
    return "Other"

  t._get_qualified_name = mock_get_qualified_name  # type: ignore
  t._cst_to_string = mock_cst_to_string  # type: ignore
  base_node = cst.Arg(value=cst.Name("MyModule"))
  class_def = cst.ClassDef(name=cst.Name("MyClass"), body=cst.IndentedBlock(body=[]), bases=[base_node])
  t.visit_ClassDef(class_def)
  assert c.in_module_class is True


def test_inject_argument_to_signature_already_present() -> None:
  """Injects argument to signature already present."""
  (t, _, _) = get_transformer()
  func_def = typing.cast(cst.FunctionDef, cst.parse_module("def foo(self, rng):\n  pass").body[0])
  res: typing.Any = t._inject_argument_to_signature(func_def, "rng", "Any")
  assert len(res.params.params) == 2


def test_inject_argument_to_signature_comma() -> None:
  """Injects argument to signature comma."""
  (t, _, _) = get_transformer()
  func_def = typing.cast(cst.FunctionDef, cst.parse_module("def foo(self):\n  pass").body[0])
  func_def = func_def.with_changes(
    params=func_def.params.with_changes(params=[cst.Param(name=cst.Name("self"), comma=cst.MaybeSentinel.DEFAULT)])
  )
  res: typing.Any = t._inject_argument_to_signature(func_def, "rng", "Any")
  assert res.params.params[0].comma != cst.MaybeSentinel.DEFAULT


def test_visit_import_aliases() -> None:
  """Verifies the behavior of visit import aliases."""
  (t, _, c) = get_transformer()
  imp_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("a"))])
  with patch.object(t, "_cst_to_string", return_value=None):
    t.visit_Import(imp_node)
  imp_from1 = cst.ImportFrom(module=cst.Name("a"), names=[cst.ImportAlias(name=cst.Name("b"))], relative=[cst.Dot()])
  t.visit_ImportFrom(imp_from1)
  imp_from2 = cst.ImportFrom(module=cst.Name("b"), names=[cst.ImportAlias(name=cst.Name("c"))])
  with patch.object(t, "_cst_to_string", return_value=None):
    t.visit_ImportFrom(imp_from2)


def test_import_star() -> None:
  """Verifies the behavior of import star."""
  (t, _, _) = get_transformer()
  imp_from = cst.ImportFrom(module=cst.Name("a"), names=cst.ImportStar())
  t.visit_ImportFrom(imp_from)


def test_import_from_non_alias() -> None:
  """Verifies the behavior of import from non alias."""
  (t, _, _) = get_transformer()
  mock_imp = MagicMock(spec=cst.ImportFrom)
  mock_imp.relative = False
  mock_imp.module = cst.Name("a")
  mock_imp.names = ["string"]
  t.visit_ImportFrom(mock_imp)


def test_leave_assign_no_source_traits() -> None:
  """Verifies the behavior of leave assign no source traits."""
  (t, _, _) = get_transformer()
  with patch(
    "ml_switcheroo.core.rewriter.passes.api.ApiTransformer.source_traits", new_callable=PropertyMock
  ) as mock_traits:
    mock_traits.side_effect = AttributeError("No source traits")
    call = cst.Call(func=cst.Name("some_func"))
    assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name("x"))], value=call)
    t.leave_Assign(assign, assign)


def test_leave_attribute_no_name() -> None:
  """Verifies the behavior of leave attribute no name."""
  (t, _, _) = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value=""):
    res: typing.Any = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_requires_plugin() -> None:
  """Verifies the behavior of leave attribute requires plugin."""
  (t, s, _) = get_transformer()
  s.definitions["a.b"] = ("abstract_id", {"variants": {"jax": {"requires_plugin": "yes"}}})
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value="a.b"):
    res: typing.Any = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_macro_exception() -> None:
  """Verifies the behavior of leave attribute macro correctly handling an exception."""
  (t, s, _) = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with (
    patch.object(t, "_get_qualified_name", return_value="a.b"),
    patch.object(t, "_get_mapping", return_value={"macro_template": "invalid"}),
    patch("ml_switcheroo.core.rewriter.calls.transformers.rewrite_as_macro", side_effect=Exception),
  ):
    res: typing.Any = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_call_handled_pre_check() -> None:
  """Verifies the behavior of leave call handled pre check."""
  (t, _, _) = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with patch(
    "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(True, cst.Name("handled"))
  ):
    res: typing.Any = t.leave_Call(call, call)
    assert isinstance(res, cst.Name)
    assert res.value == "handled"


def test_leave_call_implicit_method() -> None:
  """Verifies the behavior of leave call implicit method."""
  (t, s, _) = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value="b"),
    patch.object(t, "_get_mapping", side_effect=lambda x, **kwargs: {"api": "b"} if x == "b" else None),
  ):
    res: typing.Any = t.leave_Call(call, call)
    assert res is call


def test_leave_call_is_super() -> None:
  """Verifies the behavior of leave call is super."""
  (t, _, _) = get_transformer()
  call = cst.Call(func=cst.Name("super"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value=None),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.is_super_call", return_value=True),
  ):
    res: typing.Any = t.leave_Call(call, call)
    assert res is call


def test_leave_call_version_warning() -> None:
  """Verifies the behavior of leave call version warning."""
  (t, s, c) = get_transformer()
  call = cst.Call(func=cst.Name("func"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch.object(t, "_get_qualified_name", return_value="func"),
    patch.object(t, "_get_mapping", return_value={"min_version": "1.0", "max_version": "2.0"}),
    patch.object(t, "check_version_constraints", return_value="Version mismatch!"),
    patch.object(t, "_report_warning") as mock_warn,
  ):
    s.definitions["func"] = ("func_abstract", {"op_type": "function"})
    t.leave_Call(call, call)
    mock_warn.assert_called_with("Version mismatch!")


def test_leave_call_no_lookup() -> None:
  """Verifies the behavior of leave call no lookup."""
  (t, s, _) = get_transformer()
  call = cst.Call(func=cst.Name("func"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch.object(t, "_get_qualified_name", return_value="func"),
    patch.object(t, "_get_mapping", return_value={}),
  ):
    res: typing.Any = t.leave_Call(call, call)
    assert res is call


def test_is_module_alias() -> None:
  """Checks if is module alias."""
  (t, _, c) = get_transformer()
  with patch.object(t, "_cst_to_string", return_value=None):
    assert t._is_module_alias(cst.Name("empty")) is False
  c.alias_map["foo"] = "bar"
  with patch.object(t, "_cst_to_string", return_value="foo"):
    assert t._is_module_alias(cst.Name("foo")) is True
  t.config.source_flavour = "torch.nn"
  with patch.object(t, "_cst_to_string", return_value="torch.nn.Module"):
    assert t._is_module_alias(cst.Name("torch")) is True


def test_normalize_arguments_types() -> None:
  """Verifies the behavior of normalize arguments types."""
  (t, _, _) = get_transformer()
  op_details: dict[str, typing.Any] = {"std_args": [{"name": "a", "default": 1}, ("b", 2)]}
  call = cst.Call(func=cst.Name("func"))
  with patch.object(t, "_is_module_alias", return_value=False):
    try:
      normalize_arguments(call, call, op_details, {}, "torch", t._is_module_alias)
    except Exception:
      pass


# --- Merged from test_api_extra3.py ---


def test_api_ctx_property() -> None:
  """Verifies the behavior of API ctx property."""
  (t, _, ctx) = get_transformer()
  assert t.ctx is ctx.hook_context


def test_normalize_arguments_method_call_arg_provided() -> None:
  """Verifies the behavior of normalize arguments method call argument provided."""
  (t, _, _) = get_transformer()
  t._is_module_alias = lambda x: False  # type: ignore
  original_node = cst.Call(
    func=cst.Attribute(value=cst.Name("tensor"), attr=cst.Name("method")),
    args=[cst.Arg(keyword=cst.Name("x"), value=cst.Name("a"))],
  )
  op_details: dict[str, typing.Any] = {"std_args": ["input", "other"], "variants": {"torch": {"args": {"input": "x"}}}}
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 1
  assert typing.cast(cst.Name, args[0].keyword).value == "input"


def test_normalize_arguments_method_call_no_std_args() -> None:
  """Verifies the behavior of normalize arguments method call no std arguments."""
  (t, _, _) = get_transformer()
  t._is_module_alias = lambda x: False  # type: ignore
  original_node = cst.Call(func=cst.Attribute(value=cst.Name("tensor"), attr=cst.Name("method")), args=[])
  op_details: dict[str, typing.Any] = {}
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 1
  assert isinstance(args[0].value, cst.Name)


def test_normalize_arguments_pack_variadics_no_list_single() -> None:
  """Verifies the behavior of normalize arguments pack variadics no list single."""
  (t, _, _) = get_transformer()
  t._is_module_alias = lambda x: False  # type: ignore
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details: dict[str, typing.Any] = {"std_args": [{"name": "dim", "is_variadic": True}]}
  api_mapping: dict[str, typing.Any] = {"pack_to_tuple": "dims", "pack_as": "Tuple"}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert len(args) == 1


def test_normalize_arguments_reconstruct_defaults_error() -> None:
  """Verifies the behavior of normalize arguments reconstruct defaults correctly handling an error."""
  (t, _, _) = get_transformer()
  t._is_module_alias = lambda x: False  # type: ignore
  original_node = cst.Call(func=cst.Name("func"), args=[])
  op_details: dict[str, typing.Any] = {
    "std_args": [
      {"name": "input", "default": type("RaiseStr", (), {"__str__": lambda self: (_ for _ in ()).throw(ValueError)})()}
    ]
  }
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 0


def test_normalize_arguments_reconstruct_no_alias() -> None:
  """Verifies the behavior of normalize arguments reconstruct no alias."""
  (t, _, _) = get_transformer()
  t._is_module_alias = lambda x: False  # type: ignore
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details: dict[str, typing.Any] = {"std_args": ["input"]}
  api_mapping: dict[str, typing.Any] = {"args": {"input": None}}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert len(args) == 0


def test_normalize_arguments_reconstruct_val_map() -> None:
  """Verifies the behavior of normalize arguments reconstruct value map."""
  (t, _, _) = get_transformer()
  t._is_module_alias = lambda x: False  # type: ignore
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(keyword=cst.Name("x"), value=cst.Name("a"))])
  op_details: dict[str, typing.Any] = {"std_args": ["input"], "variants": {"torch": {"args": {"input": "x"}}}}
  api_mapping1: dict[str, typing.Any] = {"arg_values": {"input": {"a": "b"}}}
  normalize_arguments(original_node, original_node, op_details, api_mapping1, "torch", t._is_module_alias)
  api_mapping2: dict[str, typing.Any] = {"arg_values": {"input": "c + d"}}
  normalize_arguments(original_node, original_node, op_details, api_mapping2, "torch", t._is_module_alias)
  api_mapping3: dict[str, typing.Any] = {"arg_values": {"input": "invalid syntax +++"}}
  normalize_arguments(original_node, original_node, op_details, api_mapping3, "torch", t._is_module_alias)
  api_mapping4: dict[str, typing.Any] = {"arg_values": {"input": 42}}
  normalize_arguments(original_node, original_node, op_details, api_mapping4, "torch", t._is_module_alias)


def test_normalize_arguments_reconstruct_different_val() -> None:
  """Verifies the behavior of normalize arguments reconstruct different value."""
  (t, _, _) = get_transformer()
  t._is_module_alias = lambda x: False  # type: ignore
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details: dict[str, typing.Any] = {"std_args": ["input"]}
  api_mapping: dict[str, typing.Any] = {"arg_values": {"input": 42}}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert isinstance(args[0].value, cst.Integer)


def test_normalize_arguments_inject_args() -> None:
  """Verifies the behavior of normalize arguments inject arguments."""
  (t, _, _) = get_transformer()
  t._is_module_alias = lambda x: False  # type: ignore
  original_node = cst.Call(func=cst.Name("func"), args=[])
  op_details: dict[str, typing.Any] = {"std_args": []}
  api_mapping: dict[str, typing.Any] = {
    "arg_values": {"new_arg1": "a + b"},
    "inject_args": {"new_arg2": 42, "new_arg3": "invalid_syntax()"},
  }
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert len(args) == 3


def test_apply_preamble_exception() -> None:
  """Applies preamble correctly handling an exception."""
  (t, _, _) = get_transformer()
  node = cst.FunctionDef(name=cst.Name("func"), params=cst.Parameters(), body=cst.IndentedBlock(body=[]))
  res: typing.Any = t._apply_preamble(node, ["invalid syntax +++"])
  assert len(res.body.body) == 0


def test_inject_stmts_to_body() -> None:
  """Injects stmts to body."""
  (t, _, _) = get_transformer()
  node = cst.FunctionDef(name=cst.Name("func"), params=cst.Parameters(), body=cst.SimpleStatementSuite(body=[cst.Pass()]))
  res: typing.Any = t._inject_stmts_to_body(node, [cst.SimpleStatementLine(body=[cst.Expr(cst.Integer("1"))])])
  assert isinstance(res.body, cst.IndentedBlock)
  docstring_stmt = cst.SimpleStatementLine(body=[cst.Expr(cst.SimpleString('"""doc"""'))])
  node2 = cst.FunctionDef(
    name=cst.Name("func"),
    params=cst.Parameters(),
    body=cst.IndentedBlock(body=[docstring_stmt, cst.SimpleStatementLine(body=[cst.Pass()])]),
  )
  res2: typing.Any = t._inject_stmts_to_body(node2, [cst.SimpleStatementLine(body=[cst.Expr(cst.Integer("1"))])])
  assert len(res2.body.body) == 3
  assert isinstance(res2.body.body[0].body[0].value, cst.SimpleString)

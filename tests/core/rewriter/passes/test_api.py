"""Test suite for the Api module."""

import typing
from unittest.mock import MagicMock, PropertyMock, patch

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.normalization_utils import normalize_arguments
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits
from tests.conftest import TestRewriter


class MockSemantics(SemanticsManager):
  """Stub `SemanticsManager` subclass implementing `_inject`, `get_definition` for API tests."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    super().__init__(load_defaults=False)
    self.data: dict[str, dict[str, JsonValue]] = {}
    self.framework_configs: dict[str, dict[str, JsonValue]] = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    self._validation_status: dict[str, bool] = {}
    self._inject("abs", ["x"], {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}})
    self._inject("add_", ["x", "y"], {"torch": {"api": "torch.Tensor.add_"}, "jax": {"requires_plugin": "mock_unroll"}})
    self._inject("unsupported", [], {"torch": {"api": "torch.bad"}})
    self.framework_configs["jax"] = {"alias": {"module": "jax.numpy", "name": "jnp"}}

  def _inject(self, name: str, args: list[str], variants: dict[str, JsonValue]) -> None:
    """Execute `variants.values` to support inject in API tests.

    Args:
        name: Identifier or operation name string.
        args: Positional argument specifications (`list[str]`).
        variants: Per-framework operation variant mappings (`dict[str, JsonValue]`).
    """
    std_args: list[JsonValue] = list(args)
    entry: dict[str, JsonValue] = {
      "std_args": std_args,
      "variants": variants,
    }
    self.data[name] = entry
    for v in variants.values():
      if isinstance(v, dict):
        api_name = v.get("api")
        if isinstance(api_name, str):
          self._reverse_index[api_name] = (name, entry)

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    op_entry = self.data.get(aid)
    if op_entry is None:
      return None
    variants = op_entry.get("variants")
    if not isinstance(variants, dict):
      return None
    fw_variant = variants.get(fw)
    return fw_variant if isinstance(fw_variant, dict) else None

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: Input `_id` parameter value (`str`).

    Returns:
        Resulting `bool` from is verified.
    """
    return True

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(fw, {})


@pytest.fixture
def run_pass() -> typing.Callable[[str], str]:
  """Create a configured `typing.Callable[[str], str]` fixture for API tests.

  Returns:
      Configured `run_pass` fixture value (typing.Callable[[str], str]).
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  semantics.framework_configs["torch"] = {"traits": {"functional_execution_method": None}}
  rewriter = TestRewriter(semantics, config)

  def _transform(code: str) -> str:
    """Execute `rewriter.convert` to support transform in API tests.

    Args:
        code: Python or DSL source code string to parse or transpile.

    Returns:
        Resulting `str` from transform.
    """
    tree = cst.parse_module(code)
    return str(rewriter.convert(tree).code)

  return _transform


def test_api_call_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `jnp.abs(x)` for API call rewrite.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code = "y = torch.abs(x)"
  res: str = run_pass(code)
  assert "jnp.abs(x)" in res


def test_missing_mapping_strict_failure(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `No mapping available` for missing mapping strict failure.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code = "torch.bad()"
  res: str = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "No mapping available" in res


def test_assignment_unwrapping_passthrough(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` handles assignment unwrapping passthrough as expected.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code = "res = layer(x)"
  res: str = run_pass(code)
  assert "layer(x)" in res
  code2 = "res = layer.apply(v, x)"
  res2: str = run_pass(code2)
  assert "layer.apply" in res2


def test_arg_normalization_logic(run_pass: typing.Callable[[str], str], monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `RuntimeConfig` emits `jnp.sum(x, axis=1)` for arg normalization logic.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
      monkeypatch: Pytest monkeypatch fixture.
  """
  mgr = MagicMock(spec=SemanticsManager)
  jax_variant: dict[str, JsonValue] = {"api": "jnp.sum"}
  op_def: dict[str, JsonValue] = {
    "std_args": ["x", "axis"],
    "variants": {"torch": {"api": "torch.sum", "args": {"axis": "dim"}}, "jax": jax_variant},
  }
  mgr.get_definition.return_value = ("Sum", op_def)
  monkeypatch.setattr(
    mgr.resolve_variant,
    "return_value",
    jax_variant,
    raising=False,
  )
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  conf = RuntimeConfig(source_framework="torch", target_framework="jax")
  rewriter = TestRewriter(mgr, conf)
  code = "s = torch.sum(x, dim=1)"
  tree = cst.parse_module(code)
  res: str = str(rewriter.convert(tree).code)
  assert "jnp.sum(x, axis=1)" in res


class DummySemantics(SemanticsManager):
  """Test double `DummySemantics` used in API tests."""

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    super().__init__(load_defaults=False)
    self.configs: dict[str, dict[str, JsonValue]] = {}
    self.definitions: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    self.variants: dict[tuple[str, str], dict[str, JsonValue]] = {}
    self.verified = True
    self.framework_configs = self.configs

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    _ret_v: dict[str, JsonValue] = self.configs.get(fw, {})
    return _ret_v

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = self.definitions.get(name)
    return _ret_v

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    _ret_v: dict[str, JsonValue] | None = self.variants.get((abstract_id, fw))
    return _ret_v

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: Input `_id` parameter value (`str`).

    Returns:
        Resulting `bool` from is verified.
    """
    return self.verified


def get_transformer() -> tuple[ApiTransformer, DummySemantics, RewriterContext]:
  """Gets transformer.

  Returns:
      Resulting `tuple[ApiTransformer, DummySemantics, RewriterContext]` from get transformer.
  """
  semantics = DummySemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True, source_flavour="torch.nn")
  ctx = RewriterContext(semantics, config)
  ctx.hook_context = type(
    "MockHook", (), {"preamble_stmts_mock": [], "inject_preamble": lambda self, s: self.preamble_stmts_mock.append(s)}
  )()
  transformer = ApiTransformer(ctx)
  return (transformer, semantics, ctx)


def test_api_traits_fallback() -> None:
  """Verify `get_transformer` handles API traits fallback as expected."""
  (t, s, _c) = get_transformer()
  s.configs["jax"] = {}
  traits = t._get_target_traits()
  assert isinstance(traits, StructuralTraits)
  assert traits.init_method_name == "__init__"


def test_cst_to_string_fallback() -> None:
  """Verify `get_transformer` handles CST to string fallback as expected."""
  (t, _, _) = get_transformer()
  node = cst.BinaryOperation(left=cst.Name("a"), operator=cst.Add(), right=cst.Name("b"))
  assert t._cst_to_string(node) == "Add"
  node2 = cst.Integer("1")
  assert t._cst_to_string(node2) is None
  assert t._get_qualified_name(node2) is None


def test_module_bases_object(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles module bases object as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, s, _) = get_transformer()

  class DummyTraits:
    """Sentinel `DummyTraits` fixture class used in API tests."""

    module_base = "MyModule"

  monkeypatch.setattr(s, "framework_configs", {"torch": {"traits": DummyTraits()}}, raising=False)
  assert t._is_framework_base("MyModule") is True


def test_module_preamble_exception() -> None:
  """Verify `get_transformer` returns `0` for module preamble exception."""
  (t, _, c) = get_transformer()
  c.module_preamble.append("invalid python syntax +++")
  mod = cst.Module(body=[])
  res = t.leave_Module(mod, mod)
  assert len(res.body) == 0


def test_classdef_raw_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles classdef raw fallback as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _s, c) = get_transformer()
  assert t._cached_source_traits is None
  assert t._cached_target_traits is None
  assert t.target_traits is t._get_target_traits()
  assert isinstance(t._create_name_node("a.b"), cst.Attribute)
  t.state.known_module_bases = {"MyModule"}

  def mock_get_qualified_name(node: cst.CSTNode) -> str | None:
    """Execute `mock_get_qualified_name` helper in API tests.

    Args:
        node: CST or AST node under inspection or transformation.

    Returns:
        Resulting `str | None` from mock get qualified name.
    """
    return None

  def mock_cst_to_string(node: cst.CSTNode) -> str | None:
    """Execute `mock_cst_to_string` helper in API tests.

    Args:
        node: CST or AST node under inspection or transformation.

    Returns:
        Resulting `str | None` from mock cst to string.
    """
    if isinstance(node, cst.Name) and node.value == "MyModule":
      return "MyModule"
    return "Other"

  monkeypatch.setattr(t, "_get_qualified_name", mock_get_qualified_name, raising=False)
  monkeypatch.setattr(t, "_cst_to_string", mock_cst_to_string, raising=False)
  base_node = cst.Arg(value=cst.Name("MyModule"))
  class_def = cst.ClassDef(name=cst.Name("MyClass"), body=cst.IndentedBlock(body=[]), bases=[base_node])
  t.visit_ClassDef(class_def)
  assert c.in_module_class is True


def test_inject_argument_to_signature_already_present() -> None:
  """Injects argument to signature already present."""
  (t, _, _) = get_transformer()
  func_def = cst.ensure_type(cst.parse_module("def foo(self, rng):\n  pass").body[0], cst.FunctionDef)
  res = t._inject_argument_to_signature(func_def, "rng", "Any")
  assert len(res.params.params) == 2


def test_inject_argument_to_signature_comma() -> None:
  """Injects argument to signature comma."""
  (t, _, _) = get_transformer()
  func_def = cst.ensure_type(cst.parse_module("def foo(self):\n  pass").body[0], cst.FunctionDef)
  func_def = func_def.with_changes(
    params=func_def.params.with_changes(params=[cst.Param(name=cst.Name("self"), comma=cst.MaybeSentinel.DEFAULT)])
  )
  res = t._inject_argument_to_signature(func_def, "rng", "Any")
  assert res.params.params[0].comma != cst.MaybeSentinel.DEFAULT


def test_visit_import_aliases() -> None:
  """Verify `get_transformer` handles visit import aliases as expected."""
  (t, _, _c) = get_transformer()
  imp_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("a"))])
  with patch.object(t, "_cst_to_string", return_value=None):
    t.visit_Import(imp_node)
  imp_from1 = cst.ImportFrom(module=cst.Name("a"), names=[cst.ImportAlias(name=cst.Name("b"))], relative=[cst.Dot()])
  t.visit_ImportFrom(imp_from1)
  imp_from2 = cst.ImportFrom(module=cst.Name("b"), names=[cst.ImportAlias(name=cst.Name("c"))])
  with patch.object(t, "_cst_to_string", return_value=None):
    t.visit_ImportFrom(imp_from2)
  assert t.context.alias_map == {}


def test_import_star() -> None:
  """Verify `get_transformer` handles import star as expected."""
  (t, _, _) = get_transformer()
  imp_from = cst.ImportFrom(module=cst.Name("a"), names=cst.ImportStar())
  _res = t.visit_ImportFrom(imp_from)
  assert _res is False
  assert t.context.alias_map == {}


def test_import_from_non_alias() -> None:
  """Verify `get_transformer` handles import from non alias as expected."""
  (t, _, _) = get_transformer()
  mock_imp = MagicMock(spec=cst.ImportFrom)
  mock_imp.relative = False
  mock_imp.module = cst.Name("a")
  mock_imp.names = []
  _res = t.visit_ImportFrom(mock_imp)
  assert _res is False
  assert t.context.alias_map == {}


def test_leave_assign_no_source_traits() -> None:
  """Verify `get_transformer` handles leave assign no source traits as expected."""
  (t, _, _) = get_transformer()
  with patch(
    "ml_switcheroo.core.rewriter.passes.api.ApiTransformer.source_traits", new_callable=PropertyMock
  ) as mock_traits:
    mock_traits.side_effect = AttributeError("No source traits")
    call = cst.Call(func=cst.Name("some_func"))
    assign = cst.Assign(targets=[cst.AssignTarget(target=cst.Name("x"))], value=call)
    res = t.leave_Assign(assign, assign)
  assert res is assign


def test_leave_attribute_no_name() -> None:
  """Verify `get_transformer` handles leave attribute no name as expected."""
  (t, _, _) = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value=""):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_requires_plugin() -> None:
  """Verify `get_transformer` handles leave attribute requires plugin as expected."""
  (t, s, _) = get_transformer()
  s.definitions["a.b"] = ("abstract_id", {"variants": {"jax": {"requires_plugin": "yes"}}})
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value="a.b"):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_macro_exception() -> None:
  """Verify `get_transformer` handles leave attribute macro exception as expected."""
  (t, _s, _) = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with (
    patch.object(t, "_get_qualified_name", return_value="a.b"),
    patch.object(t, "_get_mapping", return_value={"macro_template": "invalid"}),
    patch("ml_switcheroo.core.rewriter.calls.transformers.rewrite_as_macro", side_effect=ValueError("invalid macro")),
  ):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_call_handled_pre_check() -> None:
  """Verify `get_transformer` handles leave call handled pre check as expected."""
  (t, _, _) = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with patch(
    "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(True, cst.Name("handled"))
  ):
    res = t.leave_Call(call, call)
    assert isinstance(res, cst.Name)
    assert res.value == "handled"


def test_leave_call_implicit_method() -> None:
  """Verify `get_transformer` handles leave call implicit method as expected."""
  (t, _s, _) = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value="b"),
    patch.object(t, "_get_mapping", side_effect=lambda x, **kwargs: {"api": "b"} if x == "b" else None),
  ):
    res = t.leave_Call(call, call)
    assert res is call


def test_leave_call_is_super() -> None:
  """Verify `get_transformer` handles leave call is super as expected."""
  (t, _, _) = get_transformer()
  call = cst.Call(func=cst.Name("super"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method", return_value=None),
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.is_super_call", return_value=True),
  ):
    res = t.leave_Call(call, call)
    assert res is call


def test_leave_call_version_warning() -> None:
  """Verify `get_transformer` handles leave call version warning as expected."""
  (t, s, _c) = get_transformer()
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
  """Verify `get_transformer` handles leave call no lookup as expected."""
  (t, _s, _) = get_transformer()
  call = cst.Call(func=cst.Name("func"))
  with (
    patch("ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(False, call)),
    patch.object(t, "_get_qualified_name", return_value="func"),
    patch.object(t, "_get_mapping", return_value={}),
  ):
    res = t.leave_Call(call, call)
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
  """Verify `get_transformer` handles normalize arguments types as expected."""
  (t, _, _) = get_transformer()
  op_details = {"std_args": [{"name": "a", "default": 1}, ("b", 2)]}
  call = cst.Call(func=cst.Name("func"))
  with patch.object(t, "_is_module_alias", return_value=False):
    res = normalize_arguments(call, call, op_details, {}, "torch", t._is_module_alias)
  assert isinstance(res, list)
  assert len(res) == 1
  assert isinstance(res[0].value, cst.Integer)
  assert res[0].value.value == "1"


def test_api_ctx_property() -> None:
  """Verify `get_transformer` handles API ctx property as expected."""
  (t, _, ctx) = get_transformer()
  assert t.ctx is ctx.hook_context


def test_normalize_arguments_method_call_arg_provided(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles normalize arguments method call arg provided as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _, _) = get_transformer()
  monkeypatch.setattr(t, "_is_module_alias", lambda x: False, raising=False)
  original_node = cst.Call(
    func=cst.Attribute(value=cst.Name("tensor"), attr=cst.Name("method")),
    args=[cst.Arg(keyword=cst.Name("x"), value=cst.Name("a"))],
  )
  op_details = {"std_args": ["input", "other"], "variants": {"torch": {"args": {"input": "x"}}}}
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 1
  assert cst.ensure_type(args[0].keyword, cst.Name).value == "input"


def test_normalize_arguments_method_call_no_std_args(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles normalize arguments method call no std args as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _, _) = get_transformer()
  monkeypatch.setattr(t, "_is_module_alias", lambda x: False, raising=False)
  original_node = cst.Call(func=cst.Attribute(value=cst.Name("tensor"), attr=cst.Name("method")), args=[])
  op_details: dict[str, RuntimeTensorOrValue] = {}
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 1
  assert isinstance(args[0].value, cst.Name)


def test_normalize_arguments_pack_variadics_no_list_single(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles normalize arguments pack variadics no list single as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _, _) = get_transformer()
  monkeypatch.setattr(t, "_is_module_alias", lambda x: False, raising=False)
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {"std_args": [{"name": "dim", "is_variadic": True}]}
  api_mapping = {"pack_to_tuple": "dims", "pack_as": "Tuple"}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert len(args) == 1


def test_normalize_arguments_reconstruct_defaults_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles normalize arguments reconstruct defaults error as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _, _) = get_transformer()
  monkeypatch.setattr(t, "_is_module_alias", lambda x: False, raising=False)
  original_node = cst.Call(func=cst.Name("func"), args=[])
  op_details = {
    "std_args": [
      {"name": "input", "default": type("RaiseStr", (), {"__str__": lambda self: (_ for _ in ()).throw(ValueError)})()}
    ]
  }
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 0


def test_normalize_arguments_reconstruct_no_alias(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles normalize arguments reconstruct no alias as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _, _) = get_transformer()
  monkeypatch.setattr(t, "_is_module_alias", lambda x: False, raising=False)
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {"std_args": ["input"]}
  api_mapping = {"args": {"input": None}}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert len(args) == 0


def test_normalize_arguments_reconstruct_val_map(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles normalize arguments reconstruct val map as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _, _) = get_transformer()
  monkeypatch.setattr(t, "_is_module_alias", lambda x: False, raising=False)
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(keyword=cst.Name("x"), value=cst.Name("a"))])
  op_details = {"std_args": ["input"], "variants": {"torch": {"args": {"input": "x"}}}}

  api_mapping1 = {"arg_values": {"input": {"a": "b"}}}
  res1 = normalize_arguments(original_node, original_node, op_details, api_mapping1, "torch", t._is_module_alias)
  assert len(res1) == 1
  assert isinstance(res1[0].value, cst.Name)
  assert res1[0].value.value == "b"

  api_mapping2 = {"arg_values": {"input": "c + d"}}
  res2 = normalize_arguments(original_node, original_node, op_details, api_mapping2, "torch", t._is_module_alias)
  assert len(res2) == 1
  assert isinstance(res2[0].value, cst.BinaryOperation)

  api_mapping3 = {"arg_values": {"input": "invalid syntax +++"}}
  res3 = normalize_arguments(original_node, original_node, op_details, api_mapping3, "torch", t._is_module_alias)
  assert len(res3) == 1
  assert isinstance(res3[0].value, cst.SimpleString)  # Fallback to string representation if eval fails, or original node

  api_mapping4 = {"arg_values": {"input": 42}}
  res4 = normalize_arguments(original_node, original_node, op_details, api_mapping4, "torch", t._is_module_alias)
  assert len(res4) == 1
  assert isinstance(res4[0].value, cst.Integer)


def test_normalize_arguments_reconstruct_different_val(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` handles normalize arguments reconstruct different val as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _, _) = get_transformer()
  monkeypatch.setattr(t, "_is_module_alias", lambda x: False, raising=False)
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {"std_args": ["input"]}
  api_mapping = {"arg_values": {"input": 42}}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert len(args) == 1
  assert isinstance(args[0].value, cst.Integer)
  assert args[0].value.value == "42"


def test_normalize_arguments_inject_args(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `get_transformer` returns `3` for normalize arguments inject args.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  (t, _, _) = get_transformer()
  monkeypatch.setattr(t, "_is_module_alias", lambda x: False, raising=False)
  original_node = cst.Call(func=cst.Name("func"), args=[])
  op_details: dict[str, RuntimeTensorOrValue] = {"std_args": []}
  api_mapping = {
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
  res = t._apply_preamble(node, ["invalid syntax +++"])
  assert len(res.body.body) == 0


def test_inject_stmts_to_body() -> None:
  """Injects stmts to body."""
  (t, _, _) = get_transformer()
  node = cst.FunctionDef(name=cst.Name("func"), params=cst.Parameters(), body=cst.SimpleStatementSuite(body=[cst.Pass()]))
  res = t._inject_stmts_to_body(node, [cst.SimpleStatementLine(body=[cst.Expr(cst.Integer("1"))])])
  assert isinstance(res.body, cst.IndentedBlock)
  docstring_stmt = cst.SimpleStatementLine(body=[cst.Expr(cst.SimpleString('"""doc"""'))])
  node2 = cst.FunctionDef(
    name=cst.Name("func"),
    params=cst.Parameters(),
    body=cst.IndentedBlock(body=[docstring_stmt, cst.SimpleStatementLine(body=[cst.Pass()])]),
  )
  res2 = t._inject_stmts_to_body(node2, [cst.SimpleStatementLine(body=[cst.Expr(cst.Integer("1"))])])
  assert len(res2.body.body) == 3
  first_stmt = cst.ensure_type(res2.body.body[0], cst.SimpleStatementLine)
  first_expr = cst.ensure_type(first_stmt.body[0], cst.Expr)
  assert isinstance(first_expr.value, cst.SimpleString)


def test_api_transformer_import_alias_tracking_and_version_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `visit_Import` / `visit_ImportFrom` alias tracking and `check_version_constraints` error resilience.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  import importlib.metadata

  ctx = RewriterContext(SemanticsManager(), RuntimeConfig(source_framework="torch", target_framework="jax"))
  p = ApiTransformer(ctx)

  imp_node = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  assert p.visit_Import(imp_node) is False
  assert p.context.alias_map.get("torch") == "torch"

  imp_from = cst.ImportFrom(
    module=cst.Name("torch"),
    names=[cst.ImportAlias(name=cst.Attribute(value=cst.Name("nn"), attr=cst.Name("functional")))],
  )
  assert p.visit_ImportFrom(imp_from) is False
  assert p.context.alias_map.get("functional") == "torch.functional"

  def _raise_runtime_error(_pkg: str) -> str:
    """Raise `RuntimeError` during package metadata lookup.

    Args:
        _pkg: Package name string.

    Returns:
        Never returns; raises `RuntimeError`.
    """
    raise RuntimeError("metadata lookup failure")

  monkeypatch.setattr(importlib.metadata, "version", _raise_runtime_error)
  monkeypatch.setattr(p.context.semantics, "get_framework_config", lambda _fw: None)
  assert p.check_version_constraints("1.0", "2.0") is None

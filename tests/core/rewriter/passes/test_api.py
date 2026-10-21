"""Test suite for the Api module."""

import typing
from pydantic import JsonValue
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
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    self.framework_configs = {}
    self.import_data: dict[str, typing.Hashable | typing.Sized] = {}
    self._reverse_index = {}
    self._validation_status = {}
    self._inject("abs", ["x"], {"torch": {"api": "torch.abs"}, "jax": {"api": "jnp.abs"}})
    self._inject("add_", ["x", "y"], {"torch": {"api": "torch.Tensor.add_"}, "jax": {"requires_plugin": "mock_unroll"}})
    self._inject("unsupported", [], {"torch": {"api": "torch.bad"}})
    self.framework_configs["jax"] = {"alias": {"module": "jax.numpy", "name": "jnp"}}

  def _inject(self, name: str, args: list[str], variants: dict[str, JsonValue]) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        variants: Test argument or fixture.
    """
    getattr(getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v"), "__setitem__")(
      name,
      getattr(
        {
          "v": {
            "std_args": getattr({"v": args}, "__getitem__")("v"),
            "variants": getattr({"v": variants}, "__getitem__")("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    for _, v in variants.items():
      if getattr(
        {"v": getattr({"v": "api"}, "__getitem__")("v") in getattr({"v": v}, "__getitem__")("v")}, "__getitem__"
      )("v"):
        getattr(
          getattr({"v": getattr({"v": self}, "__getitem__")("v")._reverse_index}, "__getitem__")("v"), "__setitem__"
        )(
          getattr({"v": v}, "__getitem__")("v")["api"],
          getattr(
            {
              "v": (
                getattr({"v": name}, "__getitem__")("v"),
                getattr(
                  {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v")[name]},
                  "__getitem__",
                )("v"),
              )
            },
            "__getitem__",
          )("v"),
        )

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {
                        "v": getattr(
                          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").data}, "__getitem__")("v").get},
                          "__getitem__",
                        )("v")(getattr({"v": aid}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
                      },
                      "__getitem__",
                    )("v").get
                  },
                  "__getitem__",
                )("v")(getattr({"v": "variants"}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
              },
              "__getitem__",
            )("v").get
          },
          "__getitem__",
        )("v")(getattr({"v": fw}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(fw, {})


@pytest.fixture
def run_pass() -> typing.Callable[[str], str]:
  """Execute run pass helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  semantics.framework_configs["torch"] = {"traits": {"functional_execution_method": None}}
  rewriter = TestRewriter(semantics, config)

  def _transform(code: str) -> str:
    """Helper to  transform.

    Args:
        code: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    tree = cst.parse_module(code)
    return str(rewriter.convert(tree).code)

  return _transform


def test_api_call_rewrite(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of API call rewrite.

  Args:
      run_pass: Test argument or fixture.
  """
  code = "y = torch.abs(x)"
  res: str = run_pass(code)
  assert "jnp.abs(x)" in res


def test_missing_mapping_strict_failure(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of missing mapping strict successfully handling failure.

  Args:
      run_pass: Test argument or fixture.
  """
  code = "torch.bad()"
  res: str = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "No mapping available" in res


def test_assignment_unwrapping_passthrough(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of assignment unwrapping passthrough.

  Args:
      run_pass: Test argument or fixture.
  """
  code = "res = layer(x)"
  res: str = run_pass(code)
  assert "layer(x)" in res
  code2 = "res = layer.apply(v, x)"
  res2: str = run_pass(code2)
  assert "layer.apply" in res2


def test_arg_normalization_logic(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of argument normalization logic.

  Args:
      run_pass: Test argument or fixture.
  """
  mgr = MagicMock(spec=SemanticsManager)
  op_def = {
    "std_args": ["x", "axis"],
    "variants": {"torch": {"api": "torch.sum", "args": {"axis": "dim"}}, "jax": {"api": "jnp.sum"}},
  }
  mgr.get_definition.return_value = ("Sum", op_def)
  setattr(
    getattr({"v": getattr({"v": mgr}, "__getitem__")("v").resolve_variant}, "__getitem__")("v"),
    "return_value",
    getattr(
      {"v": getattr({"v": getattr({"v": op_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]},
      "__getitem__",
    )("v"),
  )
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  conf = RuntimeConfig(source_framework="torch", target_framework="jax")
  rewriter = TestRewriter(mgr, conf)
  code = "s = torch.sum(x, dim=1)"
  tree = cst.parse_module(code)
  res: str = str(rewriter.convert(tree).code)
  assert "jnp.sum(x, axis=1)" in res


# --- Merged from test_api_extra2.py ---


class DummySemantics(SemanticsManager):
  """Test helper class DummySemantics."""

  framework_configs = {}

  def __init__(self) -> None:
    """Initializes the DummySemantics instance."""
    self.configs: dict[str, typing.Hashable | typing.Sized] = {}
    self.definitions: dict[str, typing.Hashable | typing.Sized] = {}
    self.variants: dict[str, typing.Hashable | typing.Sized] = {}
    self.verified = True
    self._key_origins: dict[str, str] = {}
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "framework_configs",
      getattr({"v": getattr({"v": self}, "__getitem__")("v").configs}, "__getitem__")("v"),
    )

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").configs}, "__getitem__")("v").get}, "__getitem__"
        )("v")(getattr({"v": fw}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").definitions}, "__getitem__")("v").get},
          "__getitem__",
        )("v")(getattr({"v": name}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").variants}, "__getitem__")("v").get}, "__getitem__"
        )("v")(
          getattr(
            {"v": (getattr({"v": abstract_id}, "__getitem__")("v"), getattr({"v": fw}, "__getitem__")("v"))},
            "__getitem__",
          )("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  def is_verified(self, _id: str) -> bool:
    """Mock implementation of is verified.

    Args:
        _id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.verified


def get_transformer() -> tuple[ApiTransformer, DummySemantics, RewriterContext]:
  """Gets transformer.

  Returns:
      Computed test result or fixture value.
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
    """Test helper class DummyTraits."""

    module_base = "MyModule"

  s.configs["torch"] = {"traits": DummyTraits()}
  assert t._is_framework_base("MyModule") is True


def test_module_preamble_exception() -> None:
  """Verifies the behavior of module preamble correctly handling an exception."""
  (t, _, c) = get_transformer()
  c.module_preamble.append("invalid python syntax +++")
  mod = cst.Module(body=[])
  res = t.leave_Module(mod, mod)
  assert len(res.body) == 0


def test_classdef_raw_fallback() -> None:
  """Verifies the behavior of classdef raw fallback."""
  (t, s, c) = get_transformer()
  t._known_module_bases = {"MyModule"}

  def mock_get_qualified_name(node: cst.CSTNode) -> str | None:
    """Execute mock get qualified name helper logic.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return None

  def mock_cst_to_string(node: cst.CSTNode) -> str | None:
    """Execute mock cst to string helper logic.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if isinstance(node, cst.Name) and node.value == "MyModule":
      return "MyModule"
    return "Other"

  setattr(t, "_get_qualified_name", mock_get_qualified_name)
  setattr(t, "_cst_to_string", mock_cst_to_string)
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
  assert t.context.alias_map == {}


def test_import_star() -> None:
  """Verifies the behavior of import star."""
  (t, _, _) = get_transformer()
  imp_from = cst.ImportFrom(module=cst.Name("a"), names=cst.ImportStar())
  _res = t.visit_ImportFrom(imp_from)
  assert _res is False
  assert t.context.alias_map == {}


def test_import_from_non_alias() -> None:
  """Verifies the behavior of import from non alias."""
  (t, _, _) = get_transformer()
  mock_imp = MagicMock(spec=cst.ImportFrom)
  mock_imp.relative = False
  mock_imp.module = cst.Name("a")
  mock_imp.names = ["string"]
  _res = t.visit_ImportFrom(mock_imp)
  assert _res is False
  assert t.context.alias_map == {}


def test_leave_assign_no_source_traits() -> None:
  """Verifies the behavior of leave assign no source traits."""
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
  """Verifies the behavior of leave attribute no name."""
  (t, _, _) = get_transformer()
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value=""):
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_attribute_requires_plugin() -> None:
  """Verifies the behavior of leave attribute requires plugin."""
  (t, s, _) = get_transformer()
  s.definitions["a.b"] = ("abstract_id", {"variants": {"jax": {"requires_plugin": "yes"}}})
  attr = cst.Attribute(value=cst.Name("a"), attr=cst.Name("b"))
  with patch.object(t, "_get_qualified_name", return_value="a.b"):
    res = t.leave_Attribute(attr, attr)
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
    res = t.leave_Attribute(attr, attr)
    assert res is attr


def test_leave_call_handled_pre_check() -> None:
  """Verifies the behavior of leave call handled pre check."""
  (t, _, _) = get_transformer()
  call = cst.Call(func=cst.Name("a"))
  with patch(
    "ml_switcheroo.core.rewriter.passes.api_call_mixin.handle_pre_checks", return_value=(True, cst.Name("handled"))
  ):
    res = t.leave_Call(call, call)
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
    res = t.leave_Call(call, call)
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
    res = t.leave_Call(call, call)
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
  """Verifies the behavior of normalize arguments types."""
  (t, _, _) = get_transformer()
  op_details = {"std_args": [{"name": "a", "default": 1}, ("b", 2)]}
  call = cst.Call(func=cst.Name("func"))
  with patch.object(t, "_is_module_alias", return_value=False):
    res = normalize_arguments(call, call, op_details, {}, "torch", t._is_module_alias)
  assert isinstance(res, list)


# --- Merged from test_api_extra3.py ---


def test_api_ctx_property() -> None:
  """Verifies the behavior of API ctx property."""
  (t, _, ctx) = get_transformer()
  assert t.ctx is ctx.hook_context


def test_normalize_arguments_method_call_arg_provided() -> None:
  """Verifies the behavior of normalize arguments method call argument provided."""
  (t, _, _) = get_transformer()
  setattr(t, "_is_module_alias", lambda x: False)
  original_node = cst.Call(
    func=cst.Attribute(value=cst.Name("tensor"), attr=cst.Name("method")),
    args=[cst.Arg(keyword=cst.Name("x"), value=cst.Name("a"))],
  )
  op_details = {"std_args": ["input", "other"], "variants": {"torch": {"args": {"input": "x"}}}}
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 1
  assert cst.ensure_type(args[0].keyword, cst.Name).value == "input"


def test_normalize_arguments_method_call_no_std_args() -> None:
  """Verifies the behavior of normalize arguments method call no std arguments."""
  (t, _, _) = get_transformer()
  setattr(t, "_is_module_alias", lambda x: False)
  original_node = cst.Call(func=cst.Attribute(value=cst.Name("tensor"), attr=cst.Name("method")), args=[])
  op_details: dict[str, typing.Hashable | typing.Sized] = {}
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 1
  assert isinstance(args[0].value, cst.Name)


def test_normalize_arguments_pack_variadics_no_list_single() -> None:
  """Verifies the behavior of normalize arguments pack variadics no list single."""
  (t, _, _) = get_transformer()
  setattr(t, "_is_module_alias", lambda x: False)
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {"std_args": [{"name": "dim", "is_variadic": True}]}
  api_mapping = {"pack_to_tuple": "dims", "pack_as": "Tuple"}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert len(args) == 1


def test_normalize_arguments_reconstruct_defaults_error() -> None:
  """Verifies the behavior of normalize arguments reconstruct defaults correctly handling an error."""
  (t, _, _) = get_transformer()
  setattr(t, "_is_module_alias", lambda x: False)
  original_node = cst.Call(func=cst.Name("func"), args=[])
  op_details = {
    "std_args": [
      {"name": "input", "default": type("RaiseStr", (), {"__str__": lambda self: (_ for _ in ()).throw(ValueError)})()}
    ]
  }
  args: list[cst.Arg] = normalize_arguments(original_node, original_node, op_details, {}, "torch", t._is_module_alias)
  assert len(args) == 0


def test_normalize_arguments_reconstruct_no_alias() -> None:
  """Verifies the behavior of normalize arguments reconstruct no alias."""
  (t, _, _) = get_transformer()
  setattr(t, "_is_module_alias", lambda x: False)
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {"std_args": ["input"]}
  api_mapping = {"args": {"input": None}}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert len(args) == 0


def test_normalize_arguments_reconstruct_val_map() -> None:
  """Verifies the behavior of normalize arguments reconstruct value map."""
  (t, _, _) = get_transformer()
  setattr(t, "_is_module_alias", lambda x: False)
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


def test_normalize_arguments_reconstruct_different_val() -> None:
  """Verifies the behavior of normalize arguments reconstruct different value."""
  (t, _, _) = get_transformer()
  setattr(t, "_is_module_alias", lambda x: False)
  original_node = cst.Call(func=cst.Name("func"), args=[cst.Arg(value=cst.Name("a"))])
  op_details = {"std_args": ["input"]}
  api_mapping = {"arg_values": {"input": 42}}
  args: list[cst.Arg] = normalize_arguments(
    original_node, original_node, op_details, api_mapping, "torch", t._is_module_alias
  )
  assert isinstance(args[0].value, cst.Integer)


def test_normalize_arguments_inject_args() -> None:
  """Verifies the behavior of normalize arguments inject arguments."""
  (t, _, _) = get_transformer()
  setattr(t, "_is_module_alias", lambda x: False)
  original_node = cst.Call(func=cst.Name("func"), args=[])
  op_details: dict[str, typing.Hashable | typing.Sized] = {"std_args": []}
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
  assert isinstance(
    getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {
                    "v": getattr(
                      {"v": getattr({"v": getattr({"v": res2}, "__getitem__")("v").body}, "__getitem__")("v").body},
                      "__getitem__",
                    )("v")[0]
                  },
                  "__getitem__",
                )("v").body
              },
              "__getitem__",
            )("v")[0]
          },
          "__getitem__",
        )("v").value
      },
      "__getitem__",
    )("v"),
    cst.SimpleString,
  )

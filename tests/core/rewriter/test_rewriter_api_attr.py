"""Test suite for rewriter api attr."""

import typing

import libcst as _cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.schema import SemanticTier, StructuralTraits

cst = _cst


def test_api_attr_mixin_branches() -> None:
  """Verify `RuntimeConfig` handles API attr mixin branches as expected."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  class DummySemantics:
    """Test double `DummySemantics` used in rewriter API attr tests."""

    framework_configs: typing.ClassVar[dict[str, RuntimeTensorOrValue]] = {}
    _key_origins: typing.ClassVar[dict[str, str]] = {"dummy_call": SemanticTier.NEURAL.value}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute `get_definition` on `DummySemantics` for rewriter API attr tests.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
      """
      if name == "dummy.call":
        return ("dummy_call", {"op_type": "function", "std_args": []})
      if name == "dummy.call2":
        return ("dummy_call2", {"op_type": "function"})  # No std_args
      return None

    def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
      """Execute `get_framework_config` on `DummySemantics` for rewriter API attr tests.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {}

    def resolve_variant(self, *args: str, **kwargs: str) -> dict[str, str]:
      """Execute `resolve_variant` on `DummySemantics` for rewriter API attr tests.

      Args:
          args: Positional argument specifications (`str`).
          kwargs: Keyword argument bindings (`str`).

      Returns:
          Resulting `dict[str, str]` from resolve variant.
      """
      return {"target": "jax.numpy.float32", "api": "jax.numpy.float32"}

    def is_verified(self, name: str) -> bool:
      """Execute `is_verified` on `DummySemantics` for rewriter API attr tests.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

  context = RewriterContext(semantics=DummySemantics(), config=config)
  transformer = ApiTransformer(context)

  from unittest import mock

  fake_traits = StructuralTraits(functional_execution_method="apply")
  with mock.patch.object(ApiTransformer, "source_traits", fake_traits):
    _stmt_line = cst.parse_statement("a, b = dummy.call()")
    assert isinstance(_stmt_line, cst.SimpleStatementLine)
    stmt = _stmt_line.body[0]
    assert isinstance(stmt, cst.Assign)
    transformer.leave_Assign(stmt, stmt)

    # 139->144 branch in leave_Attribute
    # "std_args" in details and details["std_args"] is false
    attr_node = cst.parse_expression("dummy.call")
    assert isinstance(attr_node, cst.Attribute)
    transformer.leave_Attribute(attr_node, attr_node)

    attr_node2 = cst.parse_expression("dummy.call2")
    assert isinstance(attr_node2, cst.Attribute)
    _res = transformer.leave_Attribute(attr_node2, attr_node2)
    assert isinstance(_res, cst.Attribute)
    assert cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(_res)])]).code.strip() == "jax.numpy.float32"


class DummySemantics:
  """Test double `DummySemantics` used in rewriter API attr tests."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.alias_map = {"t": "torch"}
    self._key_origins = {"abs_id": "neural"}
    self.get_definition_called = False
    self.defs: dict[str, tuple[str, dict[str, JsonValue]]] = {}

  def get_definition(self, func_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute `get_definition` on `DummySemantics` for rewriter API attr tests.

    Args:
        func_name: Input `func_name` parameter value (`str`).

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    self.get_definition_called = True
    return ("abs_id", {"variants": {}})

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute `get_framework_config` on `DummySemantics` for rewriter API attr tests.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> str | None:
    """Execute `resolve_op_id` on `DummySemantics` for rewriter API attr tests.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).
        name: Identifier or operation name string.

    Returns:
        Resulting `str | None` from resolve op id.
    """
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Execute `is_verified` on `DummySemantics` for rewriter API attr tests.

    Args:
        abs_id: Input `abs_id` parameter value (`str`).

    Returns:
        Resulting `bool` from is verified.
    """
    return True

  def resolve_variant(self, abs_id: str, target: str) -> dict[str, JsonValue] | None:
    """Execute `resolve_variant` on `DummySemantics` for rewriter API attr tests.

    Args:
        abs_id: Input `abs_id` parameter value (`str`).
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    return {"api": "jax.func"}


def test_api_attr_mixin_branches_extra(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `RuntimeConfig` handles API attr mixin branches extra as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  # 61->78: track variable init
  mod = cst.parse_module("a = t.func()\n")
  stmt_line = cst.ensure_type(mod.body[0], cst.SimpleStatementLine)
  assign = cst.ensure_type(stmt_line.body[0], cst.Assign)
  res = transformer.leave_Assign(assign, assign)
  assert res is assign
  assert semantics.get_definition_called

  # 139->144: leave_Attribute function with empty std_args
  semantics.get_definition_called = False
  monkeypatch.setattr(
    semantics, "get_definition", lambda x: ("abs_id", {"op_type": "function", "std_args": {}}), raising=False
  )
  attr = cst.parse_expression("t.func")
  assert isinstance(attr, cst.Attribute)
  transformer.leave_Attribute(attr, attr)

  monkeypatch.setattr(
    semantics, "get_definition", lambda x: ("abs_id", {"op_type": "function", "std_args": {"a": "b"}}), raising=False
  )
  res_attr2 = transformer.leave_Attribute(attr, attr)
  assert res_attr2 is attr


def test_api_attr_mixin_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test remaining branches in ApiTransformerAttrMixin.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)
  transformer._cached_source_traits = StructuralTraits(functional_execution_method="apply")
  _assign_lambda_line = cst.parse_statement("a = (lambda: 1)()")
  assert isinstance(_assign_lambda_line, cst.SimpleStatementLine)
  assign_lambda = _assign_lambda_line.body[0]
  assert isinstance(assign_lambda, cst.Assign)
  assert transformer.leave_Assign(assign_lambda, assign_lambda) is assign_lambda

  # 101->99: target_name is None (e.g. subscript target a[0])
  monkeypatch.setattr(semantics, "get_definition", lambda x: ("abs_id", {"variants": {}}), raising=False)
  semantics._key_origins = {"abs_id": SemanticTier.NEURAL.value}
  _assign_sub_line = cst.parse_statement("a[0] = t.func()")
  assert isinstance(_assign_sub_line, cst.SimpleStatementLine)
  assign_sub = _assign_sub_line.body[0]
  assert isinstance(assign_sub, cst.Assign)
  assert transformer.leave_Assign(assign_sub, assign_sub) is assign_sub
  _assign_multi_line = cst.parse_statement("a = b = model.apply(p, x)")
  assert isinstance(_assign_multi_line, cst.SimpleStatementLine)
  assign_multi = _assign_multi_line.body[0]
  assert isinstance(assign_multi, cst.Assign)
  assert transformer.leave_Assign(assign_multi, assign_multi) is assign_multi
  _assign_single_line = cst.parse_statement("a = model.apply(p, x)")
  assert isinstance(_assign_single_line, cst.SimpleStatementLine)
  assign_single = _assign_single_line.body[0]
  assert isinstance(assign_single, cst.Assign)
  assert transformer.leave_Assign(assign_single, assign_single) is assign_single
  _assign_empty_line = cst.parse_statement("() = model.apply(p, x)")
  assert isinstance(_assign_empty_line, cst.SimpleStatementLine)
  assign_empty = _assign_empty_line.body[0]
  assert isinstance(assign_empty, cst.Assign)
  assert transformer.leave_Assign(assign_empty, assign_empty) is assign_empty

  # 124->134: primary target not BaseAssignTargetExpression (e.g. constant in tuple)
  expr_stmt = cst.ensure_type(cst.parse_statement("model.apply(p, x)"), cst.SimpleStatementLine)
  expr_item = cst.ensure_type(expr_stmt.body[0], cst.Expr)
  assign_const = cst.Assign(
    targets=[cst.AssignTarget(target=cst.Tuple(elements=[cst.Element(value=cst.Integer("1"))]))],
    value=expr_item.value,
  )
  assert transformer.leave_Assign(assign_const, assign_const) is assign_const
  _assign_unwrapped_line = cst.parse_statement("(a, b) = model.apply(p, x)")
  assert isinstance(_assign_unwrapped_line, cst.SimpleStatementLine)
  assign_unwrapped = _assign_unwrapped_line.body[0]
  assert isinstance(assign_unwrapped, cst.Assign)
  assert transformer.leave_Assign(assign_unwrapped, assign_unwrapped) != assign_unwrapped
  _assign_not_apply_line = cst.parse_statement("a = model.other(p, x)")
  assert isinstance(_assign_not_apply_line, cst.SimpleStatementLine)
  assign_not_apply = _assign_not_apply_line.body[0]
  assert isinstance(assign_not_apply, cst.Assign)
  assert transformer.leave_Assign(assign_not_apply, assign_not_apply) is assign_not_apply

  # 171->176: op_type != "function" (e.g. constant)
  semantics.defs = {
    "t.my_const": (
      "id_const",
      {
        "op_type": "constant",
        "variants": {
          "jax": {"api": "jax.const"},
        },
      },
    )
  }
  monkeypatch.setattr(semantics, "get_definition", lambda x: semantics.defs.get(x), raising=False)
  attr_const = cst.parse_expression("t.my_const")
  assert isinstance(attr_const, cst.Attribute)
  assert transformer.leave_Attribute(attr_const, attr_const) is not None


def test_api_pass_remaining_branches() -> None:
  """Test remaining branches in ApiTransformer / ApiPass."""
  from unittest import mock

  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  # 232->exit: _mark_stateful with empty scope_stack
  transformer.context.scope_stack.clear()
  transformer._mark_stateful("my_var")

  # 310->309: visit_FunctionDef with and without parameters
  param_named = cst.Param(name=cst.Name("p"))
  func_base = cst.ensure_type(cst.parse_module("def foo(): pass").body[0], cst.FunctionDef)
  transformer.visit_FunctionDef(func_base)
  func_with_param = func_base.with_changes(params=cst.Parameters(params=[param_named]))
  transformer.visit_FunctionDef(func_with_param)

  # 406: visit_Import with empty names
  mock_import = mock.MagicMock(spec=cst.Import, names=[])
  assert transformer.visit_Import(mock_import) is False

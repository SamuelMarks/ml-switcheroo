"""Test suite for rewriter api attr."""

import typing

from typing import Any

import libcst as _cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.semantics.schema import SemanticTier, StructuralTraits

cst: typing.Any = _cst


def test_api_attr_mixin_branches() -> None:
  """Verify api attr mixin branches behavior."""
  config: typing.Any = RuntimeConfig(source_fw="torch", target_fw="jax")

  class DummySemantics:
    """Test helper class DummySemantics."""

    framework_configs: typing.Any = {}
    _key_origins: typing.Any = {"dummy_call": SemanticTier.NEURAL.value}

    def get_definition(self, name: typing.Any) -> typing.Any:
      """Execute get definition helper logic.

      Args:
          name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if name == "dummy.call":
        return ("dummy_call", {"op_type": "function", "std_args": []})
      if name == "dummy.call2":
        return ("dummy_call2", {"op_type": "function"})  # No std_args
      return None

    def get_framework_config(self, fw: typing.Any) -> typing.Any:
      """Execute get framework config helper logic.

      Args:
          fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {}

    def resolve_variant(self, *args: Any, **kwargs: Any) -> typing.Any:
      """Execute resolve variant helper logic.

      Args:
          args: Test argument or fixture.
          kwargs: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"target": "jax.numpy.float32", "api": "jax.numpy.float32"}

    def is_verified(self, name: typing.Any) -> typing.Any:
      """Execute is verified helper logic.

      Args:
          name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return True

  context: typing.Any = RewriterContext(semantics=DummySemantics(), config=config)
  transformer: typing.Any = ApiTransformer(context)

  transformer.__dict__["source_traits"] = type("StructuralTraits", (), {"functional_execution_method": "apply"})()
  # 61->78 branch in leave_Assign
  # When target_name is falsy
  stmt: typing.Any = getattr(cst.parse_statement("a, b = dummy.call()"), "body")[0]
  transformer.leave_Assign(stmt, stmt)

  # 139->144 branch in leave_Attribute
  # "std_args" in details and details["std_args"] is false
  attr_node: typing.Any = getattr(getattr(cst.parse_statement("dummy.call"), "body")[0], "value")
  transformer.leave_Attribute(attr_node, attr_node)

  attr_node2: typing.Any = getattr(getattr(cst.parse_statement("dummy.call2"), "body")[0], "value")
  _res: typing.Any = transformer.leave_Attribute(attr_node2, attr_node2)
  assert isinstance(_res, cst.BaseExpression)
  assert cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(_res)])]).code.strip() == "jax.numpy.float32"


# --- Merged from test_rewriter_api_attr_extra.py ---


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.alias_map: typing.Any = {"t": "torch"}
    self._key_origins: typing.Any = {"abs_id": "neural"}
    self.get_definition_called: typing.Any = False

  def get_definition(self, func_name: typing.Any) -> typing.Any:
    """Execute get definition helper logic.

    Args:
        func_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    self.get_definition_called = True
    return ("abs_id", {"variants": {}})

  def get_framework_config(self, fw: typing.Any) -> typing.Any:
    """Execute get framework config helper logic.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {"alias": {}}

  def resolve_op_id(self, fw: typing.Any, name: typing.Any) -> typing.Any:
    """Execute resolve op id helper logic.

    Args:
        fw: Test argument or fixture.
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return None

  def is_verified(self, abs_id: typing.Any) -> typing.Any:
    """Execute is verified helper logic.

    Args:
        abs_id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return True

  def resolve_variant(self, abs_id: typing.Any, target: typing.Any) -> typing.Any:
    """Execute resolve variant helper logic.

    Args:
        abs_id: Test argument or fixture.
        target: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {"api": "jax.func"}


def test_api_attr_mixin_branches_extra() -> None:
  """Verify api attr mixin branches extra behavior."""
  config: typing.Any = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: typing.Any = DummySemantics()
  context: typing.Any = RewriterContext(semantics=semantics, config=config)
  transformer: typing.Any = ApiTransformer(context)

  # 61->78: track variable init
  mod: typing.Any = cst.parse_module("a = t.func()\n")
  assign: typing.Any = getattr(getattr(mod, "body")[0], "body")[0]
  res: typing.Any = transformer.leave_Assign(assign, assign)
  assert res is assign
  assert semantics.get_definition_called

  # 139->144: leave_Attribute function with empty std_args
  semantics.get_definition_called = False
  setattr(semantics, "get_definition", lambda x: ("abs_id", {"op_type": "function", "std_args": {}}))
  attr: typing.Any = cst.parse_expression("t.func")
  transformer.leave_Attribute(attr, attr)

  setattr(semantics, "get_definition", lambda x: ("abs_id", {"op_type": "function", "std_args": {"a": "b"}}))
  res_attr2: typing.Any = transformer.leave_Attribute(attr, attr)
  assert res_attr2 is attr


def test_api_attr_mixin_remaining_branches() -> None:
  """Test remaining branches in ApiTransformerAttrMixin."""
  config: typing.Any = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: typing.Any = DummySemantics()
  context: typing.Any = RewriterContext(semantics=semantics, config=config)
  transformer: typing.Any = ApiTransformer(context)
  transformer._cached_source_traits = StructuralTraits(functional_execution_method="apply")

  # 92->109: func_name is None (e.g. lambda call)
  assign_lambda: typing.Any = getattr(cst.parse_statement("a = (lambda: 1)()"), "body")[0]
  assert transformer.leave_Assign(assign_lambda, assign_lambda) is assign_lambda

  # 101->99: target_name is None (e.g. subscript target a[0])
  setattr(semantics, "get_definition", lambda x: ("abs_id", {"variants": {}}))
  semantics._key_origins = {"abs_id": SemanticTier.NEURAL.value}
  assign_sub: typing.Any = getattr(cst.parse_statement("a[0] = t.func()"), "body")[0]
  assert transformer.leave_Assign(assign_sub, assign_sub) is assign_sub

  # 118->134: multi target assign with is_functional_apply
  assign_multi: typing.Any = getattr(cst.parse_statement("a = b = model.apply(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_multi, assign_multi) is assign_multi

  # 120->134: single target but not Tuple/List
  assign_single: typing.Any = getattr(cst.parse_statement("a = model.apply(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_single, assign_single) is assign_single

  # 122->134: empty Tuple target
  assign_empty: typing.Any = getattr(cst.parse_statement("() = model.apply(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_empty, assign_empty) is assign_empty

  # 124->134: primary target not BaseAssignTargetExpression (e.g. constant in tuple)
  assign_const: typing.Any = cst.Assign(
    targets=[cst.AssignTarget(target=cst.Tuple(elements=[cst.Element(value=cst.Integer("1"))]))],
    value=getattr(cst.parse_statement("model.apply(p, x)"), "body")[0].value,
  )
  assert transformer.leave_Assign(assign_const, assign_const) is assign_const

  # 124->125: unwrapping successful
  assign_unwrapped: typing.Any = getattr(cst.parse_statement("(a, b) = model.apply(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_unwrapped, assign_unwrapped) != assign_unwrapped

  # 117 False branch
  assign_not_apply: typing.Any = getattr(cst.parse_statement("a = model.other(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_not_apply, assign_not_apply) is assign_not_apply

  # 171->176: op_type != "function" (e.g. constant)
  semantics.defs = {"t.my_const": ("id_const", {"op_type": "constant", "variants": {"jax": {"api": "jax.const"}}})}
  setattr(semantics, "get_definition", lambda x: getattr(semantics, "defs", {}).get(x))
  attr_const: typing.Any = cst.parse_expression("t.my_const")
  assert transformer.leave_Attribute(attr_const, attr_const) is not None


def test_api_pass_remaining_branches() -> None:
  """Test remaining branches in ApiTransformer / ApiPass."""
  config: typing.Any = RuntimeConfig(source_fw="torch", target_fw="jax")
  semantics: typing.Any = DummySemantics()
  context: typing.Any = RewriterContext(semantics=semantics, config=config)
  transformer: typing.Any = ApiTransformer(context)

  # 232->exit: _mark_stateful with empty scope_stack
  transformer.context.scope_stack.clear()
  transformer._mark_stateful("my_var")

  # 310->309: visit_FunctionDef with param.name not cst.Name
  param_noname: typing.Any = cst.Param(name=cst.SimpleString("'p'"))
  func_noname: typing.Any = getattr(cst.parse_module("def foo(): pass"), "body")[0]
  func_noname = func_noname.with_changes(params=cst.Parameters(params=[param_noname]))
  transformer.visit_FunctionDef(func_noname)

  # 406: visit_Import with non-ImportAlias item in names
  mock_import: typing.Any = type("MockImport", (), {"names": ["not_an_import_alias"]})()
  assert transformer.visit_Import(mock_import) is False

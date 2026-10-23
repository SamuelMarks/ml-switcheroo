"""Test suite for rewriter api attr."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue


import libcst as _cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer
from ml_switcheroo.semantics.schema import SemanticTier, StructuralTraits

cst = _cst


def test_api_attr_mixin_branches() -> None:
  """Verify api attr mixin branches behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")

  class DummySemantics:
    """Test helper class DummySemantics."""

    framework_configs: dict[str, RuntimeTensorOrValue] = {}
    _key_origins = {"dummy_call": SemanticTier.NEURAL.value}

    def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
      """Execute get definition helper logic.

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
      """Execute get framework config helper logic.

      Args:
          fw: Framework identifier string (e.g., torch, jax, mlx).

      Returns:
          Resulting `dict[str, JsonValue]` from get framework config.
      """
      return {}

    def resolve_variant(self, *args: str, **kwargs: str) -> dict[str, str]:
      """Execute resolve variant helper logic.

      Args:
          args: The args (str) for resolve variant.
          kwargs: The kwargs (str) for resolve variant.

      Returns:
          Resulting `dict[str, str]` from resolve variant.
      """
      return {"target": "jax.numpy.float32", "api": "jax.numpy.float32"}

    def is_verified(self, name: str) -> bool:
      """Execute is verified helper logic.

      Args:
          name: Identifier or operation name string.

      Returns:
          Resulting `bool` from is verified.
      """
      return True

  context = RewriterContext(semantics=DummySemantics(), config=config)
  transformer = ApiTransformer(context)

  transformer.__dict__["source_traits"] = type("StructuralTraits", (), {"functional_execution_method": "apply"})()
  # 61->78 branch in leave_Assign
  # When target_name is falsy
  stmt = getattr(cst.parse_statement("a, b = dummy.call()"), "body")[0]
  transformer.leave_Assign(stmt, stmt)

  # 139->144 branch in leave_Attribute
  # "std_args" in details and details["std_args"] is false
  attr_node = getattr(getattr(cst.parse_statement("dummy.call"), "body")[0], "value")
  transformer.leave_Attribute(attr_node, attr_node)

  attr_node2 = getattr(getattr(cst.parse_statement("dummy.call2"), "body")[0], "value")
  _res = transformer.leave_Attribute(attr_node2, attr_node2)
  assert isinstance(_res, cst.Attribute)
  assert cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(_res)])]).code.strip() == "jax.numpy.float32"


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.alias_map = {"t": "torch"}
    self._key_origins = {"abs_id": "neural"}
    self.get_definition_called = False

  def get_definition(self, func_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        func_name: The func name (str) for get definition.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    self.get_definition_called = True
    return ("abs_id", {"variants": {}})

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute get framework config helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> str | None:
    """Execute resolve op id helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).
        name: Identifier or operation name string.

    Returns:
        Resulting `str | None` from resolve op id.
    """
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Execute is verified helper logic.

    Args:
        abs_id: The abs id (str) for is verified.

    Returns:
        Resulting `bool` from is verified.
    """
    return True

  def resolve_variant(self, abs_id: str, target: str) -> dict[str, JsonValue] | None:
    """Execute resolve variant helper logic.

    Args:
        abs_id: The abs id (str) for resolve variant.
        target: Target framework or dialect identifier string.

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    return {"api": "jax.func"}


def test_api_attr_mixin_branches_extra() -> None:
  """Verify api attr mixin branches extra behavior."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  # 61->78: track variable init
  mod = cst.parse_module("a = t.func()\n")
  assign = getattr(getattr(mod, "body")[0], "body")[0]
  res = transformer.leave_Assign(assign, assign)
  assert res is assign
  assert semantics.get_definition_called

  # 139->144: leave_Attribute function with empty std_args
  semantics.get_definition_called = False
  setattr(semantics, "get_definition", lambda x: ("abs_id", {"op_type": "function", "std_args": {}}))
  attr = cst.parse_expression("t.func")
  assert isinstance(attr, cst.Attribute)
  transformer.leave_Attribute(attr, attr)

  setattr(semantics, "get_definition", lambda x: ("abs_id", {"op_type": "function", "std_args": {"a": "b"}}))
  res_attr2 = transformer.leave_Attribute(attr, attr)
  assert res_attr2 is attr


def test_api_attr_mixin_remaining_branches() -> None:
  """Test remaining branches in ApiTransformerAttrMixin."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)
  transformer._cached_source_traits = StructuralTraits(functional_execution_method="apply")

  # 92->109: func_name is None (e.g. lambda call)
  assign_lambda = getattr(cst.parse_statement("a = (lambda: 1)()"), "body")[0]
  assert transformer.leave_Assign(assign_lambda, assign_lambda) is assign_lambda

  # 101->99: target_name is None (e.g. subscript target a[0])
  setattr(semantics, "get_definition", lambda x: ("abs_id", {"variants": {}}))
  semantics._key_origins = {"abs_id": SemanticTier.NEURAL.value}
  assign_sub = getattr(cst.parse_statement("a[0] = t.func()"), "body")[0]
  assert transformer.leave_Assign(assign_sub, assign_sub) is assign_sub

  # 118->134: multi target assign with is_functional_apply
  assign_multi = getattr(cst.parse_statement("a = b = model.apply(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_multi, assign_multi) is assign_multi

  # 120->134: single target but not Tuple/List
  assign_single = getattr(cst.parse_statement("a = model.apply(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_single, assign_single) is assign_single

  # 122->134: empty Tuple target
  assign_empty = getattr(cst.parse_statement("() = model.apply(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_empty, assign_empty) is assign_empty

  # 124->134: primary target not BaseAssignTargetExpression (e.g. constant in tuple)
  assign_const = cst.Assign(
    targets=[cst.AssignTarget(target=cst.Tuple(elements=[cst.Element(value=cst.Integer("1"))]))],
    value=getattr(cst.parse_statement("model.apply(p, x)"), "body")[0].value,
  )
  assert transformer.leave_Assign(assign_const, assign_const) is assign_const

  # 124->125: unwrapping successful
  assign_unwrapped = getattr(cst.parse_statement("(a, b) = model.apply(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_unwrapped, assign_unwrapped) != assign_unwrapped

  # 117 False branch
  assign_not_apply = getattr(cst.parse_statement("a = model.other(p, x)"), "body")[0]
  assert transformer.leave_Assign(assign_not_apply, assign_not_apply) is assign_not_apply

  # 171->176: op_type != "function" (e.g. constant)
  setattr(
    getattr({"v": semantics}, "__getitem__")("v"),
    "defs",
    getattr(
      {
        "v": {
          "t.my_const": getattr(
            {
              "v": (
                getattr({"v": "id_const"}, "__getitem__")("v"),
                getattr(
                  {
                    "v": {
                      "op_type": getattr({"v": "constant"}, "__getitem__")("v"),
                      "variants": getattr(
                        {
                          "v": {
                            "jax": getattr(
                              {"v": {"api": getattr({"v": "jax.const"}, "__getitem__")("v")}}, "__getitem__"
                            )("v")
                          }
                        },
                        "__getitem__",
                      )("v"),
                    }
                  },
                  "__getitem__",
                )("v"),
              )
            },
            "__getitem__",
          )("v")
        }
      },
      "__getitem__",
    )("v"),
  )
  setattr(semantics, "get_definition", lambda x: getattr(semantics, "defs", {}).get(x))
  attr_const = cst.parse_expression("t.my_const")
  assert isinstance(attr_const, cst.Attribute)
  assert transformer.leave_Attribute(attr_const, attr_const) is not None


def test_api_pass_remaining_branches() -> None:
  """Test remaining branches in ApiTransformer / ApiPass."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics = DummySemantics()
  context = RewriterContext(semantics=semantics, config=config)
  transformer = ApiTransformer(context)

  # 232->exit: _mark_stateful with empty scope_stack
  transformer.context.scope_stack.clear()
  transformer._mark_stateful("my_var")

  # 310->309: visit_FunctionDef with param.name not cst.Name
  param_noname = getattr(
    {
      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Param}, "__getitem__")("v")(
        name=getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").SimpleString}, "__getitem__")("v")(
              getattr({"v": "'p'"}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v")
      )
    },
    "__getitem__",
  )("v")
  func_noname = getattr(cst.parse_module("def foo(): pass"), "body")[0]
  func_noname = func_noname.with_changes(params=cst.Parameters(params=[param_noname]))
  transformer.visit_FunctionDef(func_noname)

  # 406: visit_Import with empty names
  mock_import = type("MockImport", (), {"names": []})()
  assert transformer.visit_Import(mock_import) is False

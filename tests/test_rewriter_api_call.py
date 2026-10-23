"""Test suite for rewriter api call."""

from pydantic import JsonValue
import pytest

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.alias_map: dict[str, str] = {"t": "torch"}
    self._key_origins: dict[str, str] = {"abs_id": "neural"}

  def get_definition(self, func_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        func_name: Function or qualified API name string.

    Returns:
        Tuple of abstract operation ID and definition metadata, or None.
    """
    return None

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute get framework config helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Configuration dictionary for the specified framework.
    """
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> str | None:
    """Execute resolve op id helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).
        name: Identifier or operation name string.

    Returns:
        Resolved canonical operation ID, or None if unmapped.
    """
    if name == "torch.nn.Linear":
      return "Linear"
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Execute is verified helper logic.

    Args:
        abs_id: Abstract operation identifier string.

    Returns:
        True if the operation is verified, False otherwise.
    """
    return True

  def resolve_variant(self, abs_id: str, target: str) -> dict[str, str]:
    """Execute resolve variant helper logic.

    Args:
        abs_id: Abstract operation identifier string.
        target: Target framework or dialect identifier string.

    Returns:
        Target framework variant dictionary, or None if unsupported.
    """
    return {"api": "jax.func"}


def test_api_call_mixin_branches() -> None:
  """Verify api call mixin branches behavior."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # 97->100: is_super_call returns updated_node
  call = cst.ensure_type(cst.parse_expression("super().func()"), cst.Call)
  res: cst.CSTNode = transformer.leave_Call(call, call)
  assert res is call

  # 110->114: abstract_id resolving
  call2 = cst.ensure_type(cst.parse_expression("torch.nn.Linear()"), cst.Call)
  res2: cst.CSTNode = transformer.leave_Call(call2, call2)
  assert res2 is call2  # actually it reports failure since tier is neural

  setattr(semantics, "get_definition", lambda x: ("Linear", {}))
  call3 = cst.ensure_type(cst.parse_expression("torch.nn.Linear()"), cst.Call)
  transformer.leave_Call(call3, call3)


class DummySemanticsExtra:
  """Test helper class DummySemanticsExtra."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.alias_map: dict[str, str] = {"t": "torch"}
    self._key_origins: dict[str, str] = {"abs_id": "neural"}

  def get_definition(self, func_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        func_name: Function or qualified API name string.

    Returns:
        Tuple of abstract operation ID and definition metadata, or None.
    """
    return None

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute get framework config helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Configuration dictionary for the specified framework.
    """
    return {"alias": {}}

  def resolve_op_id(self, fw: str, name: str) -> str | None:
    """Execute resolve op id helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).
        name: Identifier or operation name string.

    Returns:
        Resolved canonical operation ID, or None if unmapped.
    """
    if name == "torch.nn.Linear":
      return "Linear"
    return None

  def is_verified(self, abs_id: str) -> bool:
    """Execute is verified helper logic.

    Args:
        abs_id: Abstract operation identifier string.

    Returns:
        True if the operation is verified, False otherwise.
    """
    return True

  def resolve_variant(self, abs_id: str, target: str) -> dict[str, str]:
    """Execute resolve variant helper logic.

    Args:
        abs_id: Abstract operation identifier string.
        target: Target framework or dialect identifier string.

    Returns:
        Target framework variant dictionary, or None if unsupported.
    """
    return {"api": "jax.func"}


def test_api_call_mixin_branches2() -> None:
  """Verify api call mixin branches2 behavior."""
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # 97->100: is_super_call returns updated_node
  call = cst.ensure_type(cst.parse_expression("super().func()"), cst.Call)
  res: cst.CSTNode = transformer.leave_Call(call, call)
  assert res is call

  # 110->114: abstract_id resolving
  call2: cst.Call = cst.ensure_type(cst.parse_expression("torch.nn.Linear()"), cst.Call)
  # We must patch strict_mode to True via config
  config.strict_mode = True
  transformer.leave_Call(call2, call2)
  setattr(semantics, "get_definition", lambda x: ("Linear", {}))
  call3: cst.Call = cst.ensure_type(cst.parse_expression("torch.nn.Linear()"), cst.Call)
  transformer.leave_Call(call3, call3)


def test_api_call_mixin_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test remaining branches in ApiTransformerCallMixin.

  Args:
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.
  """
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics: DummySemantics = DummySemantics()
  context: RewriterContext = RewriterContext(semantics=semantics, config=config)
  transformer: ApiTransformer = ApiTransformer(context)

  # 109->112: resolve_implicit_method returns guessed name, but _get_mapping returns None
  monkeypatch.setattr(
    "ml_switcheroo.core.rewriter.passes.api_call_mixin.resolve_implicit_method",
    lambda *args, **kwargs: "torch.unmapped_guessed",
  )
  call_guess: cst.Call = cst.ensure_type(cst.parse_expression("x.func()"), cst.Call)
  assert transformer.leave_Call(call_guess, call_guess) is call_guess

  # 116->119: is_builtin returns True
  call_builtin: cst.Call = cst.ensure_type(cst.parse_expression("len([1, 2])"), cst.Call)
  assert transformer.leave_Call(call_builtin, call_builtin) is call_builtin

  # 129->133 and 136: lookup_id from abstract_id is not empty, neural tier to pure math jax
  config.strict_mode = True
  setattr(semantics, "resolve_op_id", lambda fw, name: "nn_linear")
  semantics._key_origins = {"nn_linear": "neural"}
  call_strict: cst.Call = cst.ensure_type(cst.parse_expression("torch.nn_linear()"), cst.Call)
  transformer.leave_Call(call_strict, call_strict)
  assert any("Cannot map neural network abstraction" in err for err in transformer.context.current_stmt_errors)
  transformer.context.current_stmt_errors.clear()

  setattr(semantics, "get_definition", lambda x: ("KnownOp", {}))
  monkeypatch.setattr(transformer, "_get_mapping", lambda name, **kwargs: {"api": "jax.some_func"})
  call_dep: cst.Call = cst.ensure_type(cst.parse_expression("torch.dep()"), cst.Call)
  transformer.leave_Call(call_dep, call_dep)

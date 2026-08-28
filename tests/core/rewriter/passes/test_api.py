"""Test suite for the Api module."""

import pytest
import typing
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

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
  """Provides a mock run pass for testing."""
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

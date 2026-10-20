"""Test suite for the Generator Shape module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockShapeSemantics(SemanticsManager):
  """Test helper class MockShapeSemantics."""

  def __init__(self) -> None:
    """Initializes the MockShapeSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self.framework_configs = {}
    self._key_origins = {}
    self.import_data: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
    self._known_rng_methods: set[str] = set()
    self.data["Conv2d"] = {
      "std_args": [{"name": "input", "rank": 4}, {"name": "weight", "rank": 4}],
      "variants": {
        "torch": {"api": "torch.nn.functional.conv2d", "args": {"input": "input"}},
        "jax": {"api": "jax.lax.conv", "args": {"input": "lhs", "weight": "rhs"}},
      },
    }
    self._reverse_index["torch.nn.functional.conv2d"] = ("Conv2d", self.data["Conv2d"])
    self.data["Linear"] = {
      "std_args": ["x", "w"],
      "variants": {"torch": {"api": "torch.nn.functional.linear"}, "jax": {"api": "jax.nn.linear"}},
    }
    self._reverse_index["torch.nn.functional.linear"] = ("Linear", self.data["Linear"])
    self.data["torch_nn"] = {"variants": {"jax": {"api": "jax.nn"}}}
    self._reverse_index["torch.nn"] = ("torch_nn", self.data["torch_nn"])
    self.data["torch_nn_functional"] = {"variants": {"jax": {"api": "jax.nn"}}}
    self._reverse_index["torch.nn.functional"] = ("torch_nn_functional", self.data["torch_nn_functional"])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name in self._reverse_index:
      return self._reverse_index[name]
    return None

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}


class RewriterFactoryProtocol(typing.Protocol):
  """Protocol for rewriter_factory fixture."""

  def __call__(self, strict: bool = False) -> PivotRewriter:
    """Create PivotRewriter with strict mode.

    Args:
        strict: Whether strict mode is enabled.

    Returns:
        Configured PivotRewriter instance.
    """
    ...


@pytest.fixture
def rewriter_factory() -> RewriterFactoryProtocol:
  """Execute rewriter factory helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockShapeSemantics()

  def create(strict: bool = False) -> PivotRewriter:
    """Creates .

    Args:
        strict: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=strict)
    return PivotRewriter(semantics, config)

  return create


def rewrite(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Test argument or fixture.
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


def test_strict_guard_injection(rewriter_factory: RewriterFactoryProtocol) -> None:
  """Verifies the behavior of strict guard injection.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rewriter = rewriter_factory(strict=True)
  code = "y = torch.nn.functional.conv2d(input=x, weight=w)"
  res = rewrite(rewriter, code)
  assert "def _check_rank(x, rank):" in res
  assert "_check_rank(x, 4)" in res
  assert "_check_rank(w, 4)" in res
  assert "jax.lax.conv(lhs=_check_rank(" in res.replace(" ", "")


def test_lax_mode_no_injection(rewriter_factory: RewriterFactoryProtocol) -> None:
  """Verifies the behavior of lax mode no injection.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rewriter = rewriter_factory(strict=False)
  code = "y = torch.nn.functional.conv2d(input=x, weight=w)"
  res = rewrite(rewriter, code)
  assert "_check_rank" not in res
  assert "jax.lax.conv" in res


def test_guard_ignore_no_constraint(rewriter_factory: RewriterFactoryProtocol) -> None:
  """Verifies the behavior of guard ignore no constraint.

  Args:
      rewriter_factory: Test argument or fixture.
  """
  rewriter = rewriter_factory(strict=True)
  code = "y = torch.nn.functional.linear(x, w)"
  res = rewrite(rewriter, code)
  assert "_check_rank" not in res
  assert "jax.nn.linear" in res

"""Test suite for the Output Normalization module."""

import typing
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockOutputSemantics(SemanticsManager):
  """Test helper class MockOutputSemantics."""

  def __init__(self) -> None:
    """Initializes the MockOutputSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {}
    self.framework_configs = {}
    self._inject("split_vals", "torch.split", "jax.numpy.split", select_index=0)

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Computed test result or fixture value.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str, select_index: typing.Optional[int] = None) -> None:
    """Mock implementation of  inject.

    Args:
        name: Test argument or fixture.
        s_api: Test argument or fixture.
        t_api: Test argument or fixture.
        select_index: Test argument or fixture.
    """
    variants = {"torch": {"api": s_api}, "jax": {"api": t_api}}
    target_var = variants["jax"]
    if select_index is not None:
      target_var["output_select_index"] = select_index  # type: ignore[assignment]
    self.data[name] = {"variants": variants, "std_args": ["x", "y"]}  # type: ignore[dict-item]
    self._reverse_index[s_api] = (name, self.data[name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
    """Mock implementation of resolve variant.

    Args:
        abstract_id: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)  # type: ignore[union-attr, return-value]

  def get_import_map(self, target_fw: str) -> dict[str, tuple[str, typing.Optional[str], typing.Optional[str]]]:
    """Mock implementation of get import map.

    Args:
        target_fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}


@pytest.fixture
def rewriter() -> TestRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics = MockOutputSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites .

  Args:
      rewriter: Test argument or fixture.
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


def test_structured_index_wrapping(rewriter: TestRewriter) -> None:
  """Verifies the behavior of structured index wrapping.

  Args:
      rewriter: Test argument or fixture.
  """
  code: str = "res = torch.split(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.split(x)[0]" in result
  assert "lambda" not in result

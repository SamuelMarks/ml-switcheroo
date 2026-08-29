"""Test suite for the Output Normalization module."""

import typing

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockOutputSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockOutputSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._inject("split_vals", "torch.split", "jax.numpy.split", select_index=0)

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str, select_index: typing.Optional[int] = None) -> None:
    """Mock implementation of  inject."""
    variants: dict[str, typing.Any] = {"torch": {"api": s_api}, "jax": {"api": t_api}}
    target_var: dict[str, typing.Any] = variants["jax"]
    if select_index is not None:
      target_var["output_select_index"] = select_index
    self.data[name] = {"variants": variants, "std_args": ["x", "y"]}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock implementation of resolve variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)

  def get_import_map(self, target_fw: str) -> typing.Any:
    """Mock implementation of get import map."""
    return {}


@pytest.fixture
def rewriter() -> TestRewriter:
  """Docstring."""
  semantics = MockOutputSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites ."""
  tree = cst.parse_module(code)
  new_tree: typing.Any = rewriter.convert(tree)
  return typing.cast(str, new_tree.code)


def test_structured_index_wrapping(rewriter: TestRewriter) -> None:
  """Verifies the behavior of structured index wrapping."""
  code: str = "res = torch.split(x)"
  result: str = rewrite(rewriter, code)
  assert "jax.numpy.split(x)[0]" in result
  assert "lambda" not in result

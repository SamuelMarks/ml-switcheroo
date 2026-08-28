"""Test suite for the Rewriter Alias Resolution module."""

import pytest
import libcst as cst
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from typing import Dict, Any, Tuple


class MockAliasSemantics(SemanticsManager):
  """Mock Alias Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockAliasSemantics instance."""
    self.data: Dict[str, Any] = {}
    self._reverse_index: Dict[str, Tuple[str, Any]] = {}
    self._key_origins: Dict[str, Any] = {}
    self.import_data: Dict[str, Any] = {}
    self.framework_configs: Dict[str, Any] = {}
    self._inject("abs", "torch.abs", "jax.numpy.abs")
    self._inject("Linear", "torch.nn.Linear", "flax.nnx.Linear")
    self._inject("relu", "torch.nn.functional.relu", "jax.nn.relu")

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str) -> None:
    """Mock implementation of  inject."""
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": {"api": t_api}}, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing."""
  semantics: MockAliasSemantics = MockAliasSemantics()
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  tree: cst.Module = cst.parse_module(code)
  new_tree: cst.Module = rewriter.convert(tree)
  return new_tree.code


def test_import_as_alias(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of import as alias."""
  code: str = "\nimport torch as t\ny = t.abs(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_from_import_binding(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of from import binding."""
  code: str = "\nfrom torch import nn\nlayer = nn.Linear(1, 2)\n"
  result: str = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear(1, 2)" in result


def test_from_import_as_binding(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of from import as binding."""
  code: str = "\nfrom torch import nn as n\nlayer = n.Linear(1, 2)\n"
  result: str = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear(1, 2)" in result


def test_deep_import_chains(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of deep import chains."""
  code: str = "\nimport torch.nn.functional as F\ny = F.relu(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "jax.nn.relu(x)" in result


def test_standard_import_no_alias(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of standard import no alias."""
  code: str = "\nimport torch\ny = torch.abs(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_relative_import_ignored(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of relative import ignored."""
  code: str = "\nfrom . import utils\n# utils.abs in this context is likely local, so it shouldn't match torch.abs\ny = utils.abs(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "utils.abs(x)" in result
  assert "jax.numpy.abs" not in result


def test_alias_redefinition(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of alias redefinition."""
  code: str = "\nimport torch as t\ny1 = t.abs(x)\n\nimport numpy as t\ny2 = t.abs(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "y1 = jax.numpy.abs(x)" in result
  assert "y2 = t.abs(x)" in result


def test_alias_shadowing_imported_name(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of alias shadowing imported name."""
  code: str = "\nfrom torch import nn\nl = nn.Linear(1, 2)\n"
  result: str = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear" in result

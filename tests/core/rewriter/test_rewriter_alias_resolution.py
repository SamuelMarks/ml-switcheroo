"""Test suite for the Rewriter Alias Resolution module."""

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockAliasSemantics(SemanticsManager):
  """Test double `MockAliasSemantics` used in rewriter alias resolution tests."""

  def __init__(self) -> None:
    """Initializes the MockAliasSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {}
    self._inject("abs", "torch.abs", "jax.numpy.abs")
    self._inject("Linear", "torch.nn.Linear", "flax.nnx.Linear")
    self._inject("relu", "torch.nn.functional.relu", "jax.nn.relu")

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str) -> None:
    """Execute `_inject` on `MockAliasSemantics` for rewriter alias resolution tests.

    Args:
        name: Identifier or operation name string.
        s_api: Source framework API qualified name.
        t_api: Target framework API qualified name.
    """
    self.data[name] = {"variants": {"torch": {"api": s_api}, "jax": {"api": t_api}}, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for rewriter alias resolution tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  semantics: MockAliasSemantics = MockAliasSemantics()
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  tree: cst.Module = cst.parse_module(code)
  new_tree: cst.Module = rewriter.convert(tree)
  return new_tree.code


def test_import_as_alias(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.abs(x)` for import as alias.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nimport torch as t\ny = t.abs(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_from_import_binding(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `flax.nnx.Linear(1, 2)` for from import binding.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nfrom torch import nn\nlayer = nn.Linear(1, 2)\n"
  result: str = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear(1, 2)" in result


def test_from_import_as_binding(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `flax.nnx.Linear(1, 2)` for from import as binding.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nfrom torch import nn as n\nlayer = n.Linear(1, 2)\n"
  result: str = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear(1, 2)" in result


def test_deep_import_chains(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.nn.relu(x)` for deep import chains.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nimport torch.nn.functional as F\ny = F.relu(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "jax.nn.relu(x)" in result


def test_standard_import_no_alias(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jax.numpy.abs(x)` for standard import no alias.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nimport torch\ny = torch.abs(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "jax.numpy.abs(x)" in result


def test_relative_import_ignored(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `utils.abs(x)` for relative import ignored.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nfrom . import utils\n# utils.abs in this context is likely local, so it shouldn't match torch.abs\ny = utils.abs(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "utils.abs(x)" in result
  assert "jax.numpy.abs" not in result


def test_alias_redefinition(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `y1 = jax.numpy.abs(x)` for alias redefinition.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nimport torch as t\ny1 = t.abs(x)\n\nimport numpy as t\ny2 = t.abs(x)\n"
  result: str = rewrite_code(rewriter, code)
  assert "y1 = jax.numpy.abs(x)" in result
  assert "y2 = t.abs(x)" in result


def test_alias_shadowing_imported_name(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `flax.nnx.Linear` for alias shadowing imported name.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\nfrom torch import nn\nl = nn.Linear(1, 2)\n"
  result: str = rewrite_code(rewriter, code)
  assert "flax.nnx.Linear" in result

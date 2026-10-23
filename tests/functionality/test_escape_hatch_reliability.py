"""Test suite for the Escape Hatch Reliability module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {}
    self._inject("good_op", "torch.good", "jax.good")
    self._inject("bad_op", "torch.bad", None)

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str | None) -> None:
    """Mock implementation of  inject.

    Args:
        name: Identifier or operation name string.
        s_api: Source framework API qualified name.
        t_api: Target framework API qualified name.
    """
    variants = {"torch": {"api": s_api}}
    if t_api:
      variants["jax"] = {"api": t_api}
    else:
      getattr(variants, "__setitem__")("jax", None)
    getattr(self.data, "__setitem__")(
      name,
      {
        "variants": variants,
        "std_args": ["x"],
      },
    )
    self._reverse_index[s_api] = (name, self.data[name])


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


def test_verbatim_preservation_on_partial_failure(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of verbatim preservation on partial successfully handling failure.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = torch.good(x) + torch.bad(y)"
  result: str = rewrite_code(rewriter, code)
  assert EscapeHatch.START_MARKER in result
  assert EscapeHatch.END_MARKER in result
  assert "torch.good(x)" in result
  assert "jax.good(x)" not in result
  assert "torch.bad(y)" in result


def test_end_marker_presence(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of end marker presence.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.bad(x)"
  result: str = rewrite_code(rewriter, code)
  assert EscapeHatch.START_MARKER in result
  assert EscapeHatch.END_MARKER in result
  assert "..." in result


def test_nested_call_failure_bubbling(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of nested call successfully handling failure bubbling.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "y = torch.good(torch.bad(x))"
  result: str = rewrite_code(rewriter, code)
  assert "torch.good" in result
  assert "jax.good" not in result
  assert EscapeHatch.START_MARKER in result


def test_multiple_statements_handled_independently(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of multiple statements handled independently.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\ny = torch.bad(x)\nz = torch.good(x)\n"
  result: str = rewrite_code(rewriter, code)
  print(result)
  assert "torch.bad" in result
  assert result.count(EscapeHatch.START_MARKER) == 1
  assert "jax.good(x)" in result
  assert "torch.good" not in result

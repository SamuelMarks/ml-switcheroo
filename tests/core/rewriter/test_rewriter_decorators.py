"""Test suite for the Rewriter Decorators module."""

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OpType
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter


class MockDecoratorSemantics(SemanticsManager):
  """Test double `MockDecoratorSemantics` used in rewriter decorators tests."""

  def __init__(self) -> None:
    """Initializes the MockDecoratorSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {}
    self._inject("jit", "torch.jit.script", "jax.jit")
    self._inject("inference_mode", "torch.inference_mode", None)
    self._inject("compile", "torch.compile", "jax.jit")

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: str | None) -> None:
    """Execute `_inject` on `MockDecoratorSemantics` for rewriter decorators tests.

    Args:
        name: Identifier or operation name string.
        s_api: Input `s_api` parameter value (`str`).
        t_api: Input `t_api` parameter value (`str | None`).
    """
    variants: dict[str, JsonValue] = {"torch": {"api": s_api}}
    if t_api is None:
      variants["jax"] = None
    else:
      variants["jax"] = {"api": t_api}
    op_def: dict[str, JsonValue] = {
      "op_type": OpType.DECORATOR,
      "variants": variants,
      "std_args": ["fn"],
    }
    self.data[name] = op_def
    self._reverse_index[s_api] = (name, op_def)

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock get_definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> dict[str, JsonValue] | None:
    """Mock resolve_variant.

    Args:
        abstract_id: Canonical abstract operation identifier (`str`).
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    defn = self.data.get(abstract_id)
    if not isinstance(defn, dict):
      return None
    variants = defn.get("variants")
    if not isinstance(variants, dict):
      return None
    target = variants.get(fw)
    return target if isinstance(target, dict) else None


@pytest.fixture
def rewriter() -> TestRewriter:
  """Create a configured `TestRewriter` fixture for rewriter decorators tests.

  Returns:
      Configured `rewriter` fixture value (TestRewriter).
  """
  semantics = MockDecoratorSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Execute `rewriter.convert` to support rewrite in rewriter decorators tests.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `str` from rewrite.
  """
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return str(new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewriter failed: {e}")


def test_decorator_renaming(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `@jax.jit` for decorator renaming.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\n@torch.jit.script\ndef func(x):\n    return x\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.jit.script" not in result


def test_decorator_removal(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `def func(x):` for decorator removal.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\n@torch.inference_mode\ndef func(x):\n    return x\n"
  result: str = rewrite(rewriter, code)
  assert "@torch.inference_mode" not in result
  assert "def func(x):" in result


def test_call_decorator_renaming(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `@jax.jit(fullgraph=True)` for call decorator renaming.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\n@torch.compile(fullgraph=True)\ndef func(x):\n    pass\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit(fullgraph=True)" in result
  assert "torch.compile" not in result


def test_multiple_decorators_mixed(rewriter: TestRewriter) -> None:
  """Verify `rewrite` emits `@jax.jit` for multiple decorators mixed.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "\n@torch.jit.script\n@torch.inference_mode\ndef f():\n    pass\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.inference_mode" not in result
  assert "def f():" in result

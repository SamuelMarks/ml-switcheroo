"""Test suite for the Rewriter Decorators module."""

import pytest
import typing
import libcst as cst
from tests.conftest import TestRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OpType


class MockDecoratorSemantics(SemanticsManager):
  """Mock Decorator Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockDecoratorSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {}
    self._key_origins: dict[str, str] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {}
    self._inject("jit", "torch.jit.script", "jax.jit")
    self._inject("inference_mode", "torch.inference_mode", None)
    self._inject("compile", "torch.compile", "jax.jit")

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _inject(self, name: str, s_api: str, t_api: typing.Optional[str]) -> None:
    """Mock implementation of  inject."""
    variants: dict[str, typing.Any] = {"torch": {"api": s_api}}
    if t_api is None:
      variants["jax"] = None
    else:
      variants["jax"] = {"api": t_api}
    self.data[name] = {"op_type": OpType.DECORATOR, "variants": variants, "std_args": ["fn"]}
    self._reverse_index[s_api] = (name, self.data[name])

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock get_definition."""
    return self._reverse_index.get(name)

  def resolve_variant(self, abstract_id: str, fw: str) -> typing.Any:
    """Mock resolve_variant."""
    return self.data.get(abstract_id, {}).get("variants", {}).get(fw)


@pytest.fixture
def rewriter() -> TestRewriter:
  """Provides a mock rewriter for testing."""
  semantics = MockDecoratorSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  return TestRewriter(semantics, config)


def rewrite(rewriter: TestRewriter, code: str) -> str:
  """Rewrites ."""
  tree = cst.parse_module(code)
  try:
    new_tree: typing.Any = rewriter.convert(tree)
    return typing.cast(str, new_tree.code)
  except Exception as e:
    pytest.fail(f"Rewriter failed: {e}")


def test_decorator_renaming(rewriter: TestRewriter) -> None:
  """Verifies the behavior of decorator renaming."""
  code: str = "\n@torch.jit.script\ndef func(x):\n    return x\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.jit.script" not in result


def test_decorator_removal(rewriter: TestRewriter) -> None:
  """Verifies the behavior of decorator removal."""
  code: str = "\n@torch.inference_mode\ndef func(x):\n    return x\n"
  result: str = rewrite(rewriter, code)
  assert "@torch.inference_mode" not in result
  assert "def func(x):" in result


def test_call_decorator_renaming(rewriter: TestRewriter) -> None:
  """Verifies the behavior of call decorator renaming."""
  code: str = "\n@torch.compile(fullgraph=True)\ndef func(x):\n    pass\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit(fullgraph=True)" in result
  assert "torch.compile" not in result


def test_multiple_decorators_mixed(rewriter: TestRewriter) -> None:
  """Verifies the behavior of multiple decorators mixed."""
  code: str = "\n@torch.jit.script\n@torch.inference_mode\ndef f():\n    pass\n"
  result: str = rewrite(rewriter, code)
  assert "@jax.jit" in result
  assert "@torch.inference_mode" not in result
  assert "def f():" in result

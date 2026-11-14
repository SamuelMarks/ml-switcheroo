"""Test suite for the Trait Rewriting module."""

import typing

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockTraitSemantics(SemanticsManager):
  """Stub `SemanticsManager` subclass implementing `get_framework_config` for trait rewriting tests."""

  def __init__(self) -> None:
    """Initializes the MockTraitSemantics instance."""
    super().__init__(load_defaults=False)
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.framework_configs = {
      "custom_nn": {
        "traits": {
          "module_base": "custom.Layer",
          "forward_method": "predict",
          "requires_super_init": True,
          "init_method_name": "__init__",
          "inject_magic_args": [["ctx", "custom.Context"]],
          "strip_magic_args": ["rngs"],
        }
      },
      "jax": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
      "torch": {"traits": {"module_base": "torch.nn.Module", "requires_super_init": True}},
      "ghost_fw": {"traits": {"module_base": "ghost.Network", "forward_method": "ghost_fwd"}},
    }

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        framework: Framework identifier string.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Create a configured `rewriter_factory` fixture for trait rewriting tests.

  Returns:
      Configured `rewriter_factory` fixture value (typing.Callable[[str], PivotRewriter]).
  """

  class CustomNNAdapter:
    """Test double `CustomNNAdapter` implementing `convert` for trait rewriting tests."""

    def convert(self, x: JsonValue) -> JsonValue:
      """Convert input payload in test stub.

      Args:
          x: Primary input tensor or array operand.

      Returns:
          Transpiled source code string or rewritten CST module.
      """
      return x

  register_framework("custom_nn")(CustomNNAdapter)
  register_framework("vanilla")(CustomNNAdapter)
  register_framework("ghost_fw")(CustomNNAdapter)
  semantics = MockTraitSemantics()

  def create(target_framework: str) -> PivotRewriter:
    """Execute `RuntimeConfig` to support create in trait rewriting tests.

    Args:
        target_framework: Target framework identifier string.

    Returns:
        Resulting `PivotRewriter` from create.
    """
    config = RuntimeConfig(source_framework="torch", target_framework=target_framework, strict_mode=False)
    return PivotRewriter(semantics, config)

  return create


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


def test_trait_module_inheritance_rewrite(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `class Model(custom.Layer):` for trait module inheritance rewrite.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "class Model(torch.nn.Module): pass"
  result: str = rewrite_code(rewriter, code)
  assert "class Model(custom.Layer):" in result


def test_dynamic_base_discovery(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `RuntimeConfig` emits `class MyGhost(custom.Layer):` for dynamic base discovery.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  semantics = MockTraitSemantics()
  config = RuntimeConfig(source_framework="ghost_fw", target_framework="custom_nn", strict_mode=False)
  rewriter = PivotRewriter(semantics, config)
  code: str = "\nclass MyGhost(ghost.Network):\n    def forward(self, x):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "class MyGhost(custom.Layer):" in result
  assert "def predict(self, x):" in result


def test_trait_method_renaming(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `def predict(self, x):` for trait method renaming.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "def predict(self, x):" in result
  assert "def forward" not in result


def test_trait_argument_injection(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` handles trait argument injection as expected.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "class Model(torch.nn.Module): \n    def __init__(self): pass"
  result: str = rewrite_code(rewriter, code)
  assert "def __init__(self, ctx: custom.Context):" in result


def test_trait_super_init_requirement(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `super().__init__()` for trait super init requirement.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def __init__(self):\n        self.x = 1\n"
  result: str = rewrite_code(rewriter, code)
  assert "super().__init__()" in result


def test_trait_arg_stripping(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verify `rewriter_factory` handles trait arg stripping as expected.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def __init__(self, rngs, x):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "def __init__(self, ctx: custom.Context, x):" in result
  assert "rngs" not in result

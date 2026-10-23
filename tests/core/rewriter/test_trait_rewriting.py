"""Test suite for the Trait Rewriting module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

import typing
from pydantic import JsonValue

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockTraitSemantics(SemanticsManager):
  """Test helper class MockTraitSemantics."""

  def __init__(self) -> None:
    """Initializes the MockTraitSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    self.import_data: dict[str, RuntimeTensorOrValue] = {}
    self.test_templates = {}
    setattr(
      self,
      "framework_configs",
      {
        "custom_nn": {
          "traits": {
            "module_base": "custom.Layer",
            "forward_method": "predict",
            "requires_super_init": True,
            "init_method_name": "__init__",
            "inject_magic_args": [
              (
                "ctx",
                "custom.Context",
              )
            ],
            "strip_magic_args": ["rngs"],
          }
        },
        "jax": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
        "torch": {"traits": {"module_base": "torch.nn.Module", "requires_super_init": True}},
        "ghost_fw": {"traits": {"module_base": "ghost.Network", "forward_method": "ghost_fwd"}},
      },
    )

  def get_framework_config(self, framework: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Returns:
        Configuration dictionary for the specified framework.
    """
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Execute rewriter factory helper logic.

  Returns:
      Configured `rewriter_factory` fixture value (typing.Callable[[str], PivotRewriter]).
  """

  class CustomNNAdapter:
    """Test helper class CustomNNAdapter."""

    def convert(self, x: JsonValue) -> JsonValue:
      """Converts .

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
    """Creates .

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
  """Verifies the behavior of trait module inheritance rewrite.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "class Model(torch.nn.Module): pass"
  result: str = rewrite_code(rewriter, code)
  assert "class Model(custom.Layer):" in result


def test_dynamic_base_discovery(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of dynamic base discovery.

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
  """Verifies the behavior of trait method renaming.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "def predict(self, x):" in result
  assert "def forward" not in result


def test_trait_argument_injection(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of trait argument injection.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "class Model(torch.nn.Module): \n    def __init__(self): pass"
  result: str = rewrite_code(rewriter, code)
  assert "def __init__(self, ctx: custom.Context):" in result


def test_trait_super_init_requirement(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of trait super initialization requirement.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def __init__(self):\n        self.x = 1\n"
  result: str = rewrite_code(rewriter, code)
  assert "super().__init__()" in result


def test_trait_arg_stripping(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of trait argument stripping.

  Args:
      rewriter_factory: Factory callable producing configured rewriter instances.
  """
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def __init__(self, rngs, x):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "def __init__(self, ctx: custom.Context, x):" in result
  assert "rngs" not in result

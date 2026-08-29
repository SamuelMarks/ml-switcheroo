"""Test suite for the Trait Rewriting module."""

import typing

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


class MockTraitSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockTraitSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    self._reverse_index: dict[str, typing.Any] = {}
    self.import_data: dict[str, typing.Any] = {}
    self.test_templates: dict[str, typing.Any] = {}
    self.framework_configs: dict[str, typing.Any] = {
      "custom_nn": {
        "traits": {
          "module_base": "custom.Layer",
          "forward_method": "predict",
          "requires_super_init": True,
          "init_method_name": "__init__",
          "inject_magic_args": [("ctx", "custom.Context")],
          "strip_magic_args": ["rngs"],
        }
      },
      "jax": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
      "torch": {"traits": {"module_base": "torch.nn.Module", "requires_super_init": True}},
      "ghost_fw": {"traits": {"module_base": "ghost.Network", "forward_method": "ghost_fwd"}},
    }

  def get_framework_config(self, framework: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})


@pytest.fixture
def rewriter_factory() -> typing.Callable[[str], PivotRewriter]:
  """Docstring."""

  class CustomNNAdapter:
    def convert(self, x: typing.Any) -> typing.Any:
      """Converts ."""
      return x

  register_framework("custom_nn")(CustomNNAdapter)
  register_framework("vanilla")(CustomNNAdapter)
  register_framework("ghost_fw")(CustomNNAdapter)
  semantics = MockTraitSemantics()

  def create(target_fw: str) -> PivotRewriter:
    """Creates ."""
    config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=False)
    return PivotRewriter(semantics, config)

  return create


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  new_tree: typing.Any = rewriter.convert(tree)
  return typing.cast(str, new_tree.code)


def test_trait_module_inheritance_rewrite(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of trait module inheritance rewrite."""
  rewriter = rewriter_factory("custom_nn")
  code: str = "class Model(torch.nn.Module): pass"
  result: str = rewrite_code(rewriter, code)
  assert "class Model(custom.Layer):" in result


def test_dynamic_base_discovery(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of dynamic base discovery."""
  semantics = MockTraitSemantics()
  config = RuntimeConfig(source_framework="ghost_fw", target_framework="custom_nn", strict_mode=False)
  rewriter = PivotRewriter(semantics, config)
  code: str = "\nclass MyGhost(ghost.Network):\n    def forward(self, x):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "class MyGhost(custom.Layer):" in result
  assert "def predict(self, x):" in result


def test_trait_method_renaming(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of trait method renaming."""
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "def predict(self, x):" in result
  assert "def forward" not in result


def test_trait_argument_injection(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of trait argument injection."""
  rewriter = rewriter_factory("custom_nn")
  code: str = "class Model(torch.nn.Module): \n    def __init__(self): pass"
  result: str = rewrite_code(rewriter, code)
  assert "def __init__(self, ctx: custom.Context):" in result


def test_trait_super_init_requirement(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of trait super initialization requirement."""
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def __init__(self):\n        self.x = 1\n"
  result: str = rewrite_code(rewriter, code)
  assert "super().__init__()" in result


def test_trait_arg_stripping(rewriter_factory: typing.Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of trait argument stripping."""
  rewriter = rewriter_factory("custom_nn")
  code: str = "\nclass Model(torch.nn.Module):\n    def __init__(self, rngs, x):\n        pass\n"
  result: str = rewrite_code(rewriter, code)
  assert "def __init__(self, ctx: custom.Context, x):" in result
  assert "rngs" not in result

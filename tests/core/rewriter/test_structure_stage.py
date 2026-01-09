"""
Tests for StructureStage independent execution.

Verifies that the standalone StructureStage correctly applies rewrites
using the shared RewriterContext without relying on PivotRewriter (Legacy).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.structure import StructureStage
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


class MockSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self.framework_configs = {
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
      "jax": {
        "traits": {
          "module_base": "flax.nnx.Module",
          "forward_method": "__call__",
          "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
        }
      },
    }
    self.import_data = {}

  def get_framework_config(self, fw):
    return self.framework_configs.get(fw, {})

  def get_definition(self, name):
    return None  # StructureStage relies on traits or resolve_variant


@pytest.fixture
def structure_stage():
  sem = MockSemantics()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(sem, cfg)
  return StructureStage(ctx)


def test_structure_class_rewrite(structure_stage):
  """Verify class renaming."""
  code = "class Net(torch.nn.Module): pass"
  tree = cst.parse_module(code)

  new_tree = tree.visit(structure_stage)
  res = new_tree.code

  assert "flax.nnx.Module" in res


def test_structure_func_rewrite(structure_stage):
  """Verify function signature injection via context stack."""
  code = """ 
class Net(torch.nn.Module): 
    def __init__(self, features): 
        pass
"""
  tree = cst.parse_module(code)
  new_tree = tree.visit(structure_stage)
  res = new_tree.code

  # Check injection
  assert "rngs: flax.nnx.Rngs" in res


def test_structure_type_rewrite(structure_stage):
  """Verify type annotation rewrite using context lookups."""
  # Mock lookup. We mock get_definition to return the ID,
  # AND we populate the data dict so resolve_variant works.

  # 1. Mock definition return matches (ID, details)
  definition = {"variants": {"jax": {"api": "jax.Array"}}}
  structure_stage.semantics.get_definition = MagicMock(return_value=("Tensor", definition))

  # 2. Populate data so resolve_variant finds it using ID "Tensor"
  structure_stage.semantics.data["Tensor"] = definition

  code = "def f(x: Tensor): pass"
  tree = cst.parse_module(code)
  new_tree = tree.visit(structure_stage)

  assert "jax.Array" in new_tree.code

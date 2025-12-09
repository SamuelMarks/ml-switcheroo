"""
Tests for Class Structure Rewriting: Torch -> PaxML (Praxis).
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import SemanticTier


class MockPaxmlSemantics(SemanticsManager):
  """
  Mock Manager defining mappings for Neural abstractions to PaxML.
  """

  def __init__(self) -> None:
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}

    # Configure Traits
    self.framework_configs = {
      "paxml": {
        "traits": {
          "module_base": "praxis.base_layer.BaseLayer",
          "forward_method": "__call__",
          "init_method_name": "setup",
          "requires_super_init": False,
        }
      },
      # FIX: Add Source Traits
      "torch": {"traits": {"module_base": "torch.nn.Module", "forward_method": "forward"}},
    }

    # Define 'Linear' (Neural Tier)
    self._inject(
      "Linear",
      SemanticTier.NEURAL,
      "torch",
      "torch.nn.Linear",
      "paxml",
      "praxis.layers.Linear",
    )

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(
    self,
    name: str,
    tier: SemanticTier,
    s_fw: str,
    s_api: str,
    t_fw: str,
    t_api: str,
  ) -> None:
    variants = {s_fw: {"api": s_api}, t_fw: {"api": t_api}}
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = tier.value


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Creates a Rewriter configured for Torch -> PaxML translation."""
  semantics = MockPaxmlSemantics()
  # Note: target_framework enum 'paxml'
  config = RuntimeConfig(source_framework="torch", target_framework="paxml", strict_mode=False)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Helper to parse and rewrite code."""
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_inheritance_rewrite_paxml(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(torch.nn.Module): 
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(praxis.base_layer.BaseLayer):" in result


def test_inheritance_rewrite_alias(rewriter: PivotRewriter) -> None:
  code = """ 
import torch.nn as nn
class MyModel(nn.Module): 
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(praxis.base_layer.BaseLayer):" in result


def test_forward_renaming_paxml(rewriter: PivotRewriter) -> None:
  """
  Verifies `forward` to `__call__` renaming for PaxML (Functionally similar to JAX/Flax).
  """
  code = """ 
class MyModel(torch.nn.Module): 
    def forward(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)
  assert "def __call__(self, x):" in result
  assert "def forward" not in result

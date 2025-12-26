"""
Tests for Class Structure Rewriting: Flax NNX -> PyTorch.

This module verifies the "Reverse Pivot" logic required to translate
JAX/Flax NNX models back into PyTorch's stateful, object-oriented pattern.
"""

import pytest
import libcst as cst
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.enums import SemanticTier


class MockFlaxToTorchSemantics(SemanticsManager):
  """
  Mock Manager defining mappings for Neural abstractions.
  Simulates a knowledge base supporting JAX (Source) to Torch (Target).
  """

  def __init__(self) -> None:
    """Initializes empty stores and injects deterministic test data."""
    self.data = {}
    self._reverse_index = {}
    self._key_origins = {}
    self.import_data = {}

    # Configure Traits for Torch (Target) to match legacy expectations
    self.framework_configs = {
      "torch": {
        "traits": {
          "module_base": "torch.nn.Module",
          "forward_method": "forward",
          "strip_magic_args": ["rngs"],
          "requires_super_init": True,
        }
      },
      # FIX: Add Source Traits for JAX so input class is detected
      "jax": {
        "traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"},
        # Add alias config for normalization mixin to detect flax.nnx
        "alias": {"module": "flax.nnx", "name": "nnx"},
      },
      # Add flax_nnx specific config as well since config source might be set to that
      "flax_nnx": {
        "traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"},
        "alias": {"module": "flax.nnx", "name": "nnx"},
      },
    }

    # Define 'Linear' (Neural Tier)
    self._inject(
      "Linear",
      SemanticTier.NEURAL,
      source_fw="jax",
      source_api="flax.nnx.Linear",
      target_fw="torch",
      target_api="torch.nn.Linear",
    )

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _inject(
    self,
    name: str,
    tier: SemanticTier,
    source_fw: str,
    source_api: str,
    target_fw: str,
    target_api: str,
  ) -> None:
    """Helper to inject an operation definition."""
    variants = {source_fw: {"api": source_api}, target_fw: {"api": target_api}}
    self.data[name] = {"variants": variants, "std_args": ["x"]}
    self._reverse_index[source_api] = (name, self.data[name])
    self._key_origins[name] = tier.value


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Creates a Rewriter configured for JAX -> Torch translation."""
  semantics = MockFlaxToTorchSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Helper to parse and rewrite code."""
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


def test_inheritance_rewrite(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(torch.nn.Module): 
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(flax.nnx.Module):" in result


def test_inheritance_alias_rewrite(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(nnx.Module): 
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(flax.nnx.Module):" in result


def test_init_signature_stripping(rewriter: PivotRewriter) -> None:
  # Note: Test seems reversed compared to file purpose? Source is torch, target flax_nnx in fixture.
  # But this test checks for STRIPPING rngs. Stripping happens target=torch.
  # The fixture is configured source=torch, target=flax_nnx.
  # That means we are testing Torch->Flax.
  # Wait, the file is `test_structure_flax_torch.py`.
  # This usually means Flax -> Torch.
  # Let's fix the fixture direction.

  semantics = MockFlaxToTorchSemantics()
  # Correct direction: Source=Flax, Target=Torch
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  rw = PivotRewriter(semantics, config)

  code = """ 
class MyModel(flax.nnx.Module): 
    def __init__(self, rngs: nnx.Rngs, features): 
        pass
"""
  result = rewrite_code(rw, code)

  # 'rngs' and its annotation should be gone
  assert "def __init__(self, features):" in result
  assert "rngs" not in result


def test_init_super_injection(rewriter: PivotRewriter) -> None:
  # Using rw from above fix for consistency
  semantics = MockFlaxToTorchSemantics()
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  rw = PivotRewriter(semantics, config)

  code = """ 
class MyModel(flax.nnx.Module): 
    def __init__(self): 
        self.x = 1
"""
  result = rewrite_code(rw, code)

  assert "super().__init__()" in result
  assert any("super().__init__()" in line for line in result.splitlines())


def test_init_super_injection_docstring_aware(rewriter: PivotRewriter) -> None:
  semantics = MockFlaxToTorchSemantics()
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  rw = PivotRewriter(semantics, config)

  code = """ 
class MyModel(flax.nnx.Module): 
    def __init__(self): 
        "Docstring." 
        self.x = 1
"""
  result = rewrite_code(rw, code)

  assert '"Docstring."' in result
  assert "super().__init__()" in result

  pos_doc = result.find('"Docstring."')
  pos_super = result.find("super().__init__()")
  assert pos_doc < pos_super


def test_init_layer_instantiation_stripping(rewriter: PivotRewriter) -> None:
  semantics = MockFlaxToTorchSemantics()
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  rw = PivotRewriter(semantics, config)

  # Added import so NormalizationMixin knows flax.nnx is a module
  code = """
import flax.nnx    
class MyModel(flax.nnx.Module): 
    def __init__(self, rngs): 
        self.dense = flax.nnx.Linear(10, 20, rngs=rngs) 
"""
  result = rewrite_code(rw, code)
  assert "torch.nn.Linear(10, 20)" in result
  assert "rngs=" not in result
  assert "flax.nnx" not in result.split("torch.nn.Linear")[1]  # Ensure it didn't inject module as arg


def test_call_method_renaming(rewriter: PivotRewriter) -> None:
  semantics = MockFlaxToTorchSemantics()
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch", strict_mode=False)
  rw = PivotRewriter(semantics, config)

  code = """ 
class MyModel(flax.nnx.Module): 
    def __call__(self, x): 
        return x
"""
  result = rewrite_code(rw, code)
  assert "def forward(self, x):" in result
  assert "__call__" not in result

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
      "jax": {"traits": {"module_base": "flax.nnx.Module", "forward_method": "__call__"}},
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
  config = RuntimeConfig(source_framework="jax", target_framework="torch", strict_mode=False)
  return PivotRewriter(semantics, config)


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Helper to parse, visit, and generate code."""
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewriter crashed: {e}")


def test_inheritance_rewrite(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(flax.nnx.Module): 
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(torch.nn.Module):" in result


def test_inheritance_alias_rewrite(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(nnx.Module): 
    pass
"""
  result = rewrite_code(rewriter, code)
  assert "class MyModel(torch.nn.Module):" in result


def test_init_signature_stripping(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(flax.nnx.Module): 
    def __init__(self, rngs: nnx.Rngs, features): 
        pass
"""
  result = rewrite_code(rewriter, code)

  # 'rngs' and its annotation should be gone
  # Due to the robust comma stripper, output should be clean
  assert "def __init__(self, features):" in result
  assert "rngs" not in result


def test_init_super_injection(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(flax.nnx.Module): 
    def __init__(self): 
        self.x = 1
"""
  result = rewrite_code(rewriter, code)

  assert "super().__init__()" in result
  # Flexible checking for indentation within lines
  assert any("super().__init__()" in line for line in result.splitlines())


def test_init_super_injection_docstring_aware(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(flax.nnx.Module): 
    def __init__(self): 
        "Docstring." 
        self.x = 1
"""
  result = rewrite_code(rewriter, code)

  assert '"Docstring."' in result
  assert "super().__init__()" in result

  pos_doc = result.find('"Docstring."')
  pos_super = result.find("super().__init__()")
  assert pos_doc < pos_super


def test_init_layer_instantiation_stripping(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(flax.nnx.Module): 
    def __init__(self, rngs): 
        self.dense = flax.nnx.Linear(10, 20, rngs=rngs) 
"""
  result = rewrite_code(rewriter, code)
  assert "torch.nn.Linear(10, 20)" in result
  assert "rngs=" not in result


def test_call_method_renaming(rewriter: PivotRewriter) -> None:
  code = """ 
class MyModel(flax.nnx.Module): 
    def __call__(self, x): 
        return x
"""
  result = rewrite_code(rewriter, code)
  assert "def forward(self, x):" in result
  assert "__call__" not in result

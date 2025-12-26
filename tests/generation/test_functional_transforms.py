"""
Tests for Functional API Transformations (Feature: vmap/grad/jit).

Verifies:
1.  `vmap` mapping with argument pivots:
    - Torch `in_dims` <-> Spec `in_axes` <-> JAX `in_axes`
    - Torch `func` <-> Spec `func` <-> JAX `fun`
2.  `grad` mapping.
3.  Integration with Alias Resolution.
"""

import pytest
from typing import Dict, Tuple, Optional, Any
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


# --- Mock Semantic Environment ---
# We inject the state that the scripts above would produce
class FunctionalSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()

    # New attributes
    self._providers = {}
    self._source_registry = {}

    # Define 'vmap' Abstract Spec
    self.data["vmap"] = {
      "std_args": ["func", "in_axes", "out_axes"],
      "variants": {
        "torch": {"api": "torch.vmap", "args": {"func": "func", "in_axes": "in_dims", "out_axes": "out_dims"}},
        "jax": {"api": "jax.vmap", "args": {"func": "fun"}},
      },
    }

    # Define 'grad' Abstract Spec
    self.data["grad"] = {
      "std_args": ["func", "argnums"],
      "variants": {"torch": {"api": "torch.func.grad"}, "jax": {"api": "jax.grad", "args": {"func": "fun"}}},
    }

    # Build Indices
    self._reverse_index["torch.vmap"] = ("vmap", self.data["vmap"])
    self._reverse_index["jax.vmap"] = ("vmap", self.data["vmap"])

    self._reverse_index["torch.func.grad"] = ("grad", self.data["grad"])
    self._reverse_index["jax.grad"] = ("grad", self.data["grad"])

  def get_all_rng_methods(self):
    return set()

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    return {}

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    return {}


@pytest.fixture
def engine_factory():
  semantics = FunctionalSemantics()

  def create(source, target):
    cfg = RuntimeConfig(source_framework=source, target_framework=target)
    return ASTEngine(semantics=semantics, config=cfg)

  return create


def test_torch_vmap_to_jax(engine_factory):
  """
  Scenario: Torch uses 'in_dims'. JAX expects 'in_axes'.
  Source: v = torch.vmap(my_f, in_dims=(0, None))
  Target: v = jax.vmap(fun=my_f, in_axes=(0, None))
  """
  source_code = "v = torch.vmap(my_f, in_dims=(0, None))"

  # 1. Normalize Torch 'in_dims' -> Std 'in_axes'
  # 2. Map Std 'in_axes' -> JAX 'in_axes' (Identity)
  # 3. Map Std 'func' (implicit arg 0) -> JAX 'fun'

  engine = engine_factory("torch", "jax")
  result = engine.run(source_code)

  assert result.success
  # Check API Swap
  assert "jax.vmap" in result.code
  # Check Argument Pivot
  assert "in_axes=(0, None)" in result.code
  # Check 'dims' removed
  assert "in_dims" not in result.code


def test_jax_vmap_to_torch(engine_factory):
  """
  Scenario: JAX uses 'fun' keyword and 'in_axes'. Torch expects 'in_dims'.
  Source: v = jax.vmap(fun=f, in_axes=0)
  Target: v = torch.vmap(f, in_dims=0)  # Torch arg0 is 'func'
  """
  source_code = "v = jax.vmap(fun=f, in_axes=0)"

  # 1. Normalize JAX 'fun' -> Std 'func'
  # 2. Map Std 'func' -> Torch 'func' (Identity map in variant, so it passes as kwarg or pos?)
  #    BaseRewriter Normalization converts it to positional/keyword based on spec order.
  #    If defined as keyword in source, it stays keyword in target usually, unless mapped.
  #    Torch vmap signature is vmap(func, ...). Spec is ["func", "in_axes"].

  engine = engine_factory("jax", "torch")
  result = engine.run(source_code)

  assert result.success
  assert "torch.vmap" in result.code
  assert "in_dims=0" in result.code
  # 'fun' should be pivoted to 'func' because JAX map has {"func": "fun"} (reverse map fun->func)
  # and Torch map has default.
  assert "func=f" in result.code


def test_grad_translation(engine_factory):
  """
  Scenario: torch.func.grad(f) -> jax.grad(f)
  """
  source_code = "g = torch.func.grad(predict)(params)"

  engine = engine_factory("torch", "jax")
  result = engine.run(source_code)

  assert "jax.grad(predict)(params)" in result.code

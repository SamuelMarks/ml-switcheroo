"""Test suite for the Functional Transforms module."""

import pytest
from typing import Dict, Tuple, Optional, Any
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


class FunctionalSemantics(SemanticsManager):
  """Test suite for the Functional Semantics component."""

  def __init__(self):
    """Initializes the FunctionalSemantics instance."""
    self.data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()
    self._providers = {}
    self._source_registry = {}
    self.data["vmap"] = {
      "std_args": ["func", "in_axes", "out_axes"],
      "variants": {
        "torch": {"api": "torch.vmap", "args": {"func": "func", "in_axes": "in_dims", "out_axes": "out_dims"}},
        "jax": {"api": "jax.vmap", "args": {"func": "fun"}},
      },
    }
    self.data["grad"] = {
      "std_args": ["func", "argnums"],
      "variants": {"torch": {"api": "torch.func.grad"}, "jax": {"api": "jax.grad", "args": {"func": "fun"}}},
    }
    self._reverse_index["torch.vmap"] = ("vmap", self.data["vmap"])
    self._reverse_index["jax.vmap"] = ("vmap", self.data["vmap"])
    self._reverse_index["torch.func.grad"] = ("grad", self.data["grad"])
    self._reverse_index["jax.grad"] = ("grad", self.data["grad"])

  def get_all_rng_methods(self):
    """Gets all rng methods."""
    return set()

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Gets import map."""
    return {}

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Gets framework configuration."""
    return {}


@pytest.fixture
def engine_factory():
  """Provides a mock engine factory for testing."""
  semantics = FunctionalSemantics()

  def create(source, target):
    """Creates ."""
    cfg = RuntimeConfig(source_framework=source, target_framework=target)
    return ASTEngine(semantics=semantics, config=cfg)

  return create


def test_torch_vmap_to_jax(engine_factory):
  """Verifies the behavior of PyTorch vmap to JAX."""
  source_code = "v = torch.vmap(my_f, in_dims=(0, None))"
  engine = engine_factory("torch", "jax")
  result = engine.run(source_code)
  assert result.success
  assert "jax.vmap" in result.code
  assert "in_axes=(0, None)" in result.code
  assert "in_dims" not in result.code


def test_jax_vmap_to_torch(engine_factory):
  """Verifies the behavior of JAX vmap to PyTorch."""
  source_code = "v = jax.vmap(fun=f, in_axes=0)"
  engine = engine_factory("jax", "torch")
  result = engine.run(source_code)
  assert result.success
  assert "torch.vmap" in result.code
  assert "in_dims=0" in result.code
  assert "func=f" in result.code


def test_grad_translation(engine_factory):
  """Verifies the behavior of grad translation."""
  source_code = "g = torch.func.grad(predict)(params)"
  engine = engine_factory("torch", "jax")
  result = engine.run(source_code)
  assert "jax.grad(predict)(params)" in result.code

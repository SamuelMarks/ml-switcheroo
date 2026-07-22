"""Test suite for the Vmap module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from typing import Dict, Tuple, Optional, Any


class MockVmapSemantics(SemanticsManager):
  """Mock Vmap Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockVmapSemantics instance."""
    self.data = {}
    self._providers = {}
    self._source_registry = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self.test_templates = {}
    self._known_rng_methods = set()
    vmap_def = {
      "std_args": ["func", "in_axes", "out_axes"],
      "variants": {
        "torch": {"api": "torch.vmap", "args": {"func": "func", "in_axes": "in_dims", "out_axes": "out_dims"}},
        "jax": {"api": "jax.vmap", "args": {"func": "fun", "in_axes": "in_axes", "out_axes": "out_axes"}},
      },
    }
    self.data["vmap"] = vmap_def
    self._reverse_index["torch.vmap"] = ("vmap", vmap_def)
    self._reverse_index["jax.vmap"] = ("vmap", vmap_def)

  def get_all_rng_methods(self):
    """Mock implementation of get all rng methods."""
    return self._known_rng_methods

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Mock implementation of get import map."""
    return {}

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Mock implementation of get framework configuration."""
    return {}


@pytest.fixture
def semantics():
  """Provides a mock semantics for testing."""
  return MockVmapSemantics()


def test_vmap_torch_to_jax_basic(semantics):
  """Verifies the behavior of vmap PyTorch to JAX basic."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code = "v = torch.vmap(my_func)"
  result = engine.run(code)
  assert result.success
  assert "jax.vmap(my_func)" in result.code


def test_vmap_torch_to_jax_args(semantics):
  """Verifies the behavior of vmap PyTorch to JAX arguments."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)
  code = "v = torch.vmap(f, in_dims=0, out_dims=1)"
  result = engine.run(code)
  assert result.success
  assert "in_axes=0" in result.code
  assert "out_axes=1" in result.code
  assert "jax.vmap" in result.code


def test_vmap_jax_to_torch_args(semantics):
  """Verifies the behavior of vmap JAX to PyTorch arguments."""
  config = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine = ASTEngine(semantics=semantics, config=config)
  code = "v = jax.vmap(f, in_axes=(0, None), out_axes=0)"
  result = engine.run(code)
  assert result.success
  assert "torch.vmap" in result.code
  assert "in_dims=(0, None)" in result.code
  assert "out_dims=0" in result.code


def test_vmap_jax_keyword_fun(semantics):
  """Verifies the behavior of vmap JAX keyword fun."""
  config = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine = ASTEngine(semantics=semantics, config=config)
  code = "v = jax.vmap(fun=my_f)"
  result = engine.run(code)
  assert "torch.vmap" in result.code
  assert "func=my_f" in result.code

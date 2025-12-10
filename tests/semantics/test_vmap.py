"""
Tests for `vmap` Semantics (Feature 057).

Verifies:
1.  `torch.vmap` <-> `jax.vmap` mapping.
2.  Argument swapping: `in_dims` (Torch) <-> `in_axes` (JAX).
3.  Argument swapping: `out_dims` (Torch) <-> `out_axes` (JAX).
4.  Function argument mapping: `func` (Torch) <-> `fun` (JAX) (via positional or keyword).
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


class MockVmapSemantics(SemanticsManager):
  """Mock semantics manager containing vmap definitions."""

  def __init__(self):
    # Skip file loading to stay isolated
    self.data = {}
    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self._key_origins = {}
    self._validation_status = {}
    self.test_templates = {}
    self._known_rng_methods = set()

    # Inject vmap definition
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


@pytest.fixture
def semantics():
  """Returns a mock semantics manager with vmap data."""
  return MockVmapSemantics()


def test_vmap_torch_to_jax_basic(semantics):
  """
  Scenario: `v = torch.vmap(my_func)`
  Expect: `v = jax.vmap(my_func)`
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)

  code = "v = torch.vmap(my_func)"
  result = engine.run(code)

  assert result.success
  assert "jax.vmap(my_func)" in result.code


def test_vmap_torch_to_jax_args(semantics):
  """
  Scenario: `v = torch.vmap(f, in_dims=0, out_dims=1)`
  Expect: `v = jax.vmap(f, in_axes=0, out_axes=1)`
  (Renaming `dims` -> `axes`).
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)

  code = "v = torch.vmap(f, in_dims=0, out_dims=1)"
  result = engine.run(code)

  assert result.success
  # Check argument renaming
  assert "in_axes=0" in result.code
  assert "out_axes=1" in result.code
  assert "jax.vmap" in result.code


def test_vmap_jax_to_torch_args(semantics):
  """
  Scenario: `v = jax.vmap(f, in_axes=(0, None), out_axes=0)`
  Expect: `v = torch.vmap(f, in_dims=(0, None), out_dims=0)`
  (Renaming `axes` -> `dims`).
  """
  config = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine = ASTEngine(semantics=semantics, config=config)

  code = "v = jax.vmap(f, in_axes=(0, None), out_axes=0)"
  result = engine.run(code)

  assert result.success
  assert "torch.vmap" in result.code
  assert "in_dims=(0, None)" in result.code
  assert "out_dims=0" in result.code


def test_vmap_jax_keyword_fun(semantics):
  """
  Scenario: `v = jax.vmap(fun=my_f)`
  Expect: `v = torch.vmap(func=my_f)`
  (Renaming `fun` -> `func` if keyword used, assuming standard is `func`).
  """
  config = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine = ASTEngine(semantics=semantics, config=config)

  code = "v = jax.vmap(fun=my_f)"
  result = engine.run(code)

  # Standard arg is 'func'.
  # Torch variant args map: None (implicit func->func)
  # JAX variant args map: func->fun.
  # So source 'fun' maps to standard 'func'.
  # Target 'torch' maps standard 'func' to 'func' (implicit default).
  # Thus 'fun' -> 'func'.

  assert "torch.vmap" in result.code
  # Check keyword swap
  assert "func=my_f" in result.code

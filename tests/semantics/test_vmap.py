"""Test suite for the Vmap module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from typing import Dict, Tuple, Optional, Set, Any


class MockVmapSemantics(SemanticsManager):
  """Mock Vmap Semantics class for testing purposes."""

  def __init__(self) -> None:
    """Initializes the MockVmapSemantics instance."""
    self.data: Dict[str, Any] = {}
    self._providers: Dict[str, Any] = {}
    self._source_registry: Dict[str, Any] = {}
    self.framework_configs: Dict[str, Any] = {}
    self._reverse_index: Dict[str, Any] = {}
    self._key_origins: Dict[str, Any] = {}
    self._validation_status: Dict[str, Any] = {}
    self.test_templates: Dict[str, Any] = {}
    self._known_rng_methods: Set[str] = set()
    vmap_def: Dict[str, Any] = {
      "std_args": ["func", "in_axes", "out_axes"],
      "variants": {
        "torch": {"api": "torch.vmap", "args": {"func": "func", "in_axes": "in_dims", "out_axes": "out_dims"}},
        "jax": {"api": "jax.vmap", "args": {"func": "fun", "in_axes": "in_axes", "out_axes": "out_axes"}},
      },
    }
    self.data["vmap"] = vmap_def
    self._reverse_index["torch.vmap"] = ("vmap", vmap_def)
    self._reverse_index["jax.vmap"] = ("vmap", vmap_def)

  def get_all_rng_methods(self) -> Set[str]:
    """Mock implementation of get all rng methods.

    Returns:
        Set[str]: Set of methods.
    """
    return self._known_rng_methods

  def get_import_map(self, target_fw: str) -> Dict[str, Tuple[str, Optional[str], Optional[str]]]:
    """Mock implementation of get import map.

    Args:
        target_fw (str): Target framework string.

    Returns:
        Dict[str, Tuple[str, Optional[str], Optional[str]]]: Empty map.
    """
    return {}

  def get_framework_config(self, framework: str) -> Dict[str, Any]:
    """Mock implementation of get framework configuration.

    Args:
        framework (str): Target framework string.

    Returns:
        Dict[str, Any]: Empty dict.
    """
    return {}


@pytest.fixture
def semantics() -> MockVmapSemantics:
  """Provides a mock semantics for testing.

  Returns:
      MockVmapSemantics: A mock semantics instance.
  """
  return MockVmapSemantics()


def test_vmap_torch_to_jax_basic(semantics: MockVmapSemantics) -> None:
  """Verifies the behavior of vmap PyTorch to JAX basic.

  Args:
      semantics (MockVmapSemantics): The semantics fixture.
  """
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine: ASTEngine = ASTEngine(semantics=semantics, config=config)
  code: str = "v = torch.vmap(my_func)"
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "jax.vmap(my_func)" in result.code


def test_vmap_torch_to_jax_args(semantics: MockVmapSemantics) -> None:
  """Verifies the behavior of vmap PyTorch to JAX arguments.

  Args:
      semantics (MockVmapSemantics): The semantics fixture.
  """
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine: ASTEngine = ASTEngine(semantics=semantics, config=config)
  code: str = "v = torch.vmap(f, in_dims=0, out_dims=1)"
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "in_axes=0" in result.code
  assert "out_axes=1" in result.code
  assert "jax.vmap" in result.code


def test_vmap_jax_to_torch_args(semantics: MockVmapSemantics) -> None:
  """Verifies the behavior of vmap JAX to PyTorch arguments.

  Args:
      semantics (MockVmapSemantics): The semantics fixture.
  """
  config: RuntimeConfig = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine: ASTEngine = ASTEngine(semantics=semantics, config=config)
  code: str = "v = jax.vmap(f, in_axes=(0, None), out_axes=0)"
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "torch.vmap" in result.code
  assert "in_dims=(0, None)" in result.code
  assert "out_dims=0" in result.code


def test_vmap_jax_keyword_fun(semantics: MockVmapSemantics) -> None:
  """Verifies the behavior of vmap JAX keyword fun.

  Args:
      semantics (MockVmapSemantics): The semantics fixture.
  """
  config: RuntimeConfig = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine: ASTEngine = ASTEngine(semantics=semantics, config=config)
  code: str = "v = jax.vmap(fun=my_f)"
  result: ConversionResult = engine.run(code)
  assert "torch.vmap" in result.code
  assert "func=my_f" in result.code

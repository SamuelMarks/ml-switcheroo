"""
Tests for Verification Harness Protocol Adherence.

Verifies that all registered adapters:
1. Implement `harness_imports` returning a list.
2. Implement `get_harness_init_code` returning a string.
3. Specific frameworks (JAX, Flax) provide expected initialization logic.
4. Other frameworks return default/empty values.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.base import get_adapter
from ml_switcheroo.frameworks import available_frameworks
import ml_switcheroo.frameworks.jax
import ml_switcheroo.frameworks.flax_nnx
import ml_switcheroo.frameworks.torch


@pytest.fixture
def mock_all_imports():
  """Simulate all frameworks being importable to avoid Ghost mode fallback logic in constructors."""
  with patch.dict(
    sys.modules,
    {
      "jax": MagicMock(),
      "jax.numpy": MagicMock(),
      "flax.nnx": MagicMock(),
      "torch": MagicMock(),
      "tensorflow": MagicMock(),
      "mlx": MagicMock(),
      "mlx.core": MagicMock(),
    },
  ):
    yield


def test_protocol_implementation_coverage(mock_all_imports):
  """
  Iterates all registered frameworks and verifies types.
  """
  fws = available_frameworks()
  # Ensure our main targets are present in registry
  assert "jax" in fws
  assert "flax_nnx" in fws
  assert "torch" in fws

  for fw in fws:
    adapter = get_adapter(fw)

    # Check imports property
    imports = adapter.harness_imports
    assert isinstance(imports, list), f"{fw} harness_imports should be list"

    # Check code method
    code = adapter.get_harness_init_code()
    assert isinstance(code, str), f"{fw} get_harness_init_code should return str"


def test_jax_implementation_content():
  adapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()

  # Imports
  assert "import jax.random" in adapter.harness_imports

  # Init Code
  code = adapter.get_harness_init_code()
  assert "def _make_jax_key" in code
  assert "jax.random.PRNGKey" in code


def test_flax_implementation_content():
  # Mock flax import check inside init
  with patch.dict(sys.modules, {"flax.nnx": MagicMock()}):
    adapter = ml_switcheroo.frameworks.flax_nnx.FlaxNNXAdapter()

    # Imports
    assert "from flax import nnx" in adapter.harness_imports

    # Init Code
    code = adapter.get_harness_init_code()
    assert "def _make_flax_rngs" in code
    assert "nnx.Rngs" in code


def test_torch_no_op_implementation():
  with patch.dict(sys.modules, {"torch": MagicMock()}):
    adapter = ml_switcheroo.frameworks.torch.TorchAdapter()

    assert adapter.harness_imports == []
    assert adapter.get_harness_init_code() == ""

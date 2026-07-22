"""Test suite for the Harness Protocol module."""

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
  """Provides a mock all imports for testing."""
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
  """Verifies the behavior of protocol implementation coverage."""
  fws = available_frameworks()
  assert "jax" in fws
  assert "flax_nnx" in fws
  assert "torch" in fws
  for fw in fws:
    adapter = get_adapter(fw)
    imports = adapter.harness_imports
    assert isinstance(imports, list), f"{fw} harness_imports should be list"
    code = adapter.get_harness_init_code()
    assert isinstance(code, str), f"{fw} get_harness_init_code should return str"


def test_jax_implementation_content():
  """Verifies the behavior of JAX implementation content."""
  adapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
  assert "import jax.random" in adapter.harness_imports
  code = adapter.get_harness_init_code()
  assert "def _make_jax_key" in code
  assert "jax.random.PRNGKey" in code


def test_flax_implementation_content():
  """Verifies the behavior of Flax implementation content."""
  with patch.dict(sys.modules, {"flax.nnx": MagicMock()}):
    adapter = ml_switcheroo.frameworks.flax_nnx.FlaxNNXAdapter()
    assert "from flax import nnx" in adapter.harness_imports
    code = adapter.get_harness_init_code()
    assert "def _make_flax_rngs" in code
    assert "nnx.Rngs" in code


def test_torch_no_op_implementation():
  """Verifies the behavior of PyTorch no op implementation."""
  with patch.dict(sys.modules, {"torch": MagicMock()}):
    adapter = ml_switcheroo.frameworks.torch.TorchAdapter()
    assert adapter.harness_imports == []
    assert adapter.get_harness_init_code() == ""

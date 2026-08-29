"""Test suite for the Harness Protocol module."""

import sys
import typing
from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo.frameworks.flax_nnx
import ml_switcheroo.frameworks.jax
import ml_switcheroo.frameworks.torch
from ml_switcheroo.frameworks import available_frameworks
from ml_switcheroo.frameworks.base import get_adapter


@pytest.fixture
def mock_all_imports() -> typing.Generator[None, None, None]:
  """Docstring."""
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


def test_protocol_implementation_coverage(mock_all_imports: None) -> None:
  """Verifies the behavior of protocol implementation coverage."""
  fws: list[str] = available_frameworks()
  assert "jax" in fws
  assert "flax_nnx" in fws
  assert "torch" in fws
  for fw in fws:
    adapter: typing.Any = get_adapter(fw)
    imports: typing.Any = adapter.harness_imports
    assert isinstance(imports, list), f"{fw} harness_imports should be list"
    code: str = adapter.get_harness_init_code()
    assert isinstance(code, str), f"{fw} get_harness_init_code should return str"


def test_jax_implementation_content() -> None:
  """Verifies the behavior of JAX implementation content."""
  adapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
  assert "import jax.random" in adapter.harness_imports
  code: str = adapter.get_harness_init_code()
  assert "def _make_jax_key" in code
  assert "jax.random.PRNGKey" in code


def test_flax_implementation_content() -> None:
  """Verifies the behavior of Flax implementation content."""
  with patch.dict(sys.modules, {"flax.nnx": MagicMock()}):
    adapter = ml_switcheroo.frameworks.flax_nnx.FlaxNNXAdapter()
    assert "from flax import nnx" in adapter.harness_imports
    code: str = adapter.get_harness_init_code()
    assert "def _make_flax_rngs" in code
    assert "nnx.Rngs" in code


def test_torch_no_op_implementation() -> None:
  """Verifies the behavior of PyTorch no op implementation."""
  with patch.dict(sys.modules, {"torch": MagicMock()}):
    adapter = ml_switcheroo.frameworks.torch.TorchAdapter()
    assert adapter.harness_imports == []
    assert adapter.get_harness_init_code() == ""

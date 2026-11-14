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
  """Create a configured `typing.Generator[None, None, None]` fixture for harness protocol tests.

  Yields:
      Configured `mock_all_imports` fixture value (typing.Generator[None, None, None]).
  """
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
  """Verify `available_frameworks` emits `jax` for protocol implementation coverage.

  Args:
      mock_all_imports: Input `mock_all_imports` parameter value (`None`).
  """
  fws: list[str] = available_frameworks()
  assert "jax" in fws
  assert "flax_nnx" in fws
  assert "torch" in fws
  for fw in fws:
    adapter = get_adapter(fw)
    assert adapter is not None
    imports = adapter.harness_imports
    assert isinstance(imports, list), f"{fw} harness_imports should be list"
    code: str = adapter.get_harness_init_code()
    assert isinstance(code, str), f"{fw} get_harness_init_code should return str"


def test_jax_implementation_content() -> None:
  """Verify `JaxCoreAdapter` emits `import jax.random` for JAX implementation content."""
  adapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
  assert "import jax.random" in adapter.harness_imports
  code: str = adapter.get_harness_init_code()
  assert "def _make_jax_key" in code
  assert "jax.random.PRNGKey" in code


def test_flax_implementation_content() -> None:
  """Verify `FlaxNNXAdapter` emits `from flax import nnx` for flax implementation content."""
  with patch.dict(sys.modules, {"flax.nnx": MagicMock()}):
    adapter = ml_switcheroo.frameworks.flax_nnx.FlaxNNXAdapter()
    assert "from flax import nnx" in adapter.harness_imports
    code: str = adapter.get_harness_init_code()
    assert "def _make_flax_rngs" in code
    assert "nnx.Rngs" in code


def test_torch_no_op_implementation() -> None:
  """Verify `TorchAdapter` handles torch no operation implementation as expected."""
  with patch.dict(sys.modules, {"torch": MagicMock()}):
    adapter = ml_switcheroo.frameworks.torch.TorchAdapter()
    assert adapter.harness_imports == []
    assert adapter.get_harness_init_code() == ""

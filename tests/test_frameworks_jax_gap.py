"""Test suite for the Frameworks Jax Gap module."""

from unittest import mock
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


def test_jax_import_success():
  """Verifies the behavior of JAX import successfully."""
  with mock.patch.dict("sys.modules", {"jax": mock.MagicMock(), "jax.numpy": mock.MagicMock()}):
    import ml_switcheroo.frameworks.jax as fjax
    import importlib

    importlib.reload(fjax)
    assert fjax.jax is not None
    assert fjax.jnp is not None


def test_jax_convert_fallback():
  """Verifies the behavior of JAX convert fallback."""
  mock_jax = mock.MagicMock()
  mock_jnp = mock.MagicMock()
  with mock.patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jnp}):
    adapter = JaxCoreAdapter()
    res = adapter.convert("not an array")
    assert res == "not an array"

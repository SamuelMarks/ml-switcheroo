"""Test suite for the Frameworks Jax Gap module."""

from unittest import mock

from ml_switcheroo.frameworks.jax import JaxCoreAdapter


def test_jax_import_success() -> None:
  """Verify `importlib.reload` handles JAX import success as expected."""
  with mock.patch.dict("sys.modules", {"jax": mock.MagicMock(), "jax.numpy": mock.MagicMock()}):
    import importlib

    import ml_switcheroo.frameworks.jax as fjax

    importlib.reload(fjax)
    assert fjax.jax is not None
    assert fjax.jnp is not None
    assert fjax.JaxCoreAdapter().import_alias == ("jax.numpy", "jnp")


def test_jax_convert_fallback() -> None:
  """Verify `JaxCoreAdapter` handles JAX convert fallback as expected."""
  mock_jax: mock.MagicMock = mock.MagicMock()
  mock_jnp: mock.MagicMock = mock.MagicMock()
  with mock.patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jnp}):
    adapter: JaxCoreAdapter = JaxCoreAdapter()
    res = adapter.convert("not an array")
    assert res == "not an array"

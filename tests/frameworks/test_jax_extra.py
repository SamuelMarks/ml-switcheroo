"""Module docstring."""

from unittest.mock import patch
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.enums import SemanticTier


def test_jax_adapter_ghost_mode():
  """Docstring."""
  with patch("ml_switcheroo.frameworks.jax.jax", None):
    with patch(
      "ml_switcheroo.frameworks.jax.load_snapshot_for_adapter",
      return_value={"categories": {"loss": [{"name": "foo", "kind": "function", "api_path": "foo"}]}},
    ):
      adapter = JaxCoreAdapter()
      assert adapter._mode.value == "ghost"

      ghosts = adapter._collect_ghost(SemanticTier.LOSS)
      assert len(ghosts) == 1
      assert ghosts[0].name == "foo"


def test_jax_adapter_ghost_mode_no_snapshot():
  """Docstring."""
  with patch("ml_switcheroo.frameworks.jax.jax", None):
    with patch("ml_switcheroo.frameworks.jax.load_snapshot_for_adapter", return_value={}):
      adapter = JaxCoreAdapter()
      assert adapter._mode.value == "ghost"
      ghosts = adapter._collect_ghost(SemanticTier.LOSS)
      assert len(ghosts) == 0


def test_jax_adapter_plugin_traits():
  """Docstring."""
  adapter = JaxCoreAdapter()
  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays


def test_jax_adapter_collect_live():
  """Docstring."""
  adapter = JaxCoreAdapter()
  with patch("ml_switcheroo.frameworks.jax.OptaxScanner.scan_losses", return_value=["loss1"], create=True):
    assert "loss1" in adapter._collect_live(SemanticTier.LOSS)
  with patch("ml_switcheroo.frameworks.jax.OptaxScanner.scan_optimizers", return_value=["opt1"], create=True):
    assert "opt1" in adapter._collect_live(SemanticTier.OPTIMIZER)


def test_jax_adapter_convert_exception():
  """Docstring."""
  adapter = JaxCoreAdapter()
  with patch("builtins.__import__", side_effect=Exception):
    assert adapter.convert([1, 2]) == [1, 2]

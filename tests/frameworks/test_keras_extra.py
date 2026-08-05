"""Test module."""

from unittest.mock import MagicMock, patch
from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier
import sys


def test_keras_collect_live(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.keras as keras_fw

  # Mock keras and submodules
  mock_keras = MagicMock()
  mock_keras.losses = MagicMock()
  mock_keras.optimizers = MagicMock()
  mock_keras.activations = MagicMock()
  mock_keras.layers = MagicMock()
  monkeypatch.setattr(keras_fw, "keras", mock_keras)

  adapter = keras_fw.KerasAdapter()
  adapter._mode = "LIVE"

  # Mock _scan_module on the adapter
  def mock_scan_module(module, prefix, kind, block_list=None):
    """Mocks _scan_module."""
    from ml_switcheroo_ir.schema.ghost import GhostRef

    return [GhostRef(api_path=prefix + ".Test", name="Test", kind=kind, group=kind, params=[])]

  adapter._scan_module = mock_scan_module

  assert adapter._collect_live(SemanticTier.LOSS)[0].api_path == "keras.losses.Test"
  assert adapter._collect_live(SemanticTier.OPTIMIZER)[0].api_path == "keras.optimizers.Test"
  assert adapter._collect_live(SemanticTier.ACTIVATION)[0].api_path == "keras.activations.Test"
  assert adapter._collect_live(SemanticTier.LAYER)[0].api_path == "keras.layers.Test"
  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


def test_keras_convert(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  mock_keras = MagicMock()
  mock_keras.ops.convert_to_tensor.return_value = "tensor"

  # Needs to patch import inside method
  with patch.dict(sys.modules, {"keras": mock_keras}):
    assert adapter.convert([1, 2, 3]) == "tensor"
    mock_keras.ops.convert_to_tensor.assert_called_once_with([1, 2, 3])


def test_keras_convert_fail(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  # Force ImportError
  with patch.dict(sys.modules, {"keras": None}):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]


def test_keras_collect_ghost(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.keras as keras_fw

  monkeypatch.setattr(keras_fw, "keras", None)

  with patch(
    "ml_switcheroo.frameworks.keras.load_snapshot_for_adapter",
    return_value={
      "categories": {
        "extras": [{"name": "fake", "api_path": "keras.fake", "kind": "class", "group": "class", "params": []}]
      }
    },
  ):
    adapter = keras_fw.KerasAdapter()
    ghosts = adapter._collect_ghost(SemanticTier.EXTRAS)
    assert len(ghosts) == 1
    assert ghosts[0].api_path == "keras.fake"

    adapter._snapshot_data = {}
    assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_keras_collect_ghost_no_snapshot():
  """Test function."""
  # Hit line 208
  adapter = KerasAdapter()
  adapter._snapshot_data = None
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_keras_rng_split():
  """Test function."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"


def test_keras_init_missing():
  """Test function."""
  # Hit lines 25-26, 70
  import importlib
  import ml_switcheroo.frameworks.keras as keras_fw

  old_keras = sys.modules.get("keras")
  sys.modules["keras"] = None
  try:
    importlib.reload(keras_fw)
    assert keras_fw.keras is None
    with patch("ml_switcheroo.frameworks.keras.load_snapshot_for_adapter", return_value=None):
      adapter = keras_fw.KerasAdapter()
      assert adapter._mode.name == "GHOST"
  finally:
    if old_keras:
      sys.modules["keras"] = old_keras
    else:
      del sys.modules["keras"]

"""Test module."""

from unittest.mock import patch, MagicMock
import typing
import pytest
from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_keras_test_config() -> None:
  """Test function."""
  adapter = KerasAdapter()
  assert adapter.test_config["import"] == "import keras\nfrom keras import ops"


def test_keras_get_to_numpy_code() -> None:
  """Test function."""
  adapter = KerasAdapter()
  assert adapter.get_to_numpy_code() == "if hasattr(obj, 'numpy'): return obj.numpy()"


def test_keras_rng_seed_methods() -> None:
  """Test function."""
  adapter = KerasAdapter()
  assert adapter.rng_seed_methods == ["utils.set_random_seed"]


def test_keras_device_check_syntax() -> None:
  """Test function."""
  adapter = KerasAdapter()
  assert adapter.get_device_check_syntax() == "len(keras.config.list_logical_devices('GPU')) > 0"


def test_keras_apply_wiring() -> None:
  """Test function."""
  adapter = KerasAdapter()
  adapter.apply_wiring({})


def test_keras_get_doc_url() -> None:
  """Test function."""
  adapter = KerasAdapter()
  assert adapter.get_doc_url("keras.layers.Dense") == "https://keras.io/search.html?q=keras.layers.Dense"


def test_keras_convert_import_error() -> None:
  """Test function."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()
  real_import = __import__

  def mock_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks __import__ to raise ImportError."""
    if name == "keras":
      raise ImportError("Fail")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    assert adapter.convert([1]) == [1]


def test_keras_collect_live_all(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test function."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  mock_keras = MagicMock()
  monkeypatch.setattr(keras_fw, "keras", mock_keras)

  def mock_scan(module: typing.Any, prefix: str, kind: str, block_list: typing.Any = None) -> typing.Any:
    """Mocks _scan_module."""
    from ml_switcheroo_ir.schema.ghost import GhostRef

    return [GhostRef(api=prefix + ".X", api_path=prefix + ".X", name="X", kind=kind, group=kind, params=[])]  # type: ignore

  adapter._scan_module = mock_scan  # type: ignore

  l1: list[typing.Any] = adapter._collect_live(SemanticTier.LOSS)
  l2: list[typing.Any] = adapter._collect_live(SemanticTier.OPTIMIZER)
  l3: list[typing.Any] = adapter._collect_live(SemanticTier.ACTIVATION)
  l4: list[typing.Any] = adapter._collect_live(SemanticTier.LAYER)
  assert len(l1) == 1
  assert len(l2) == 1
  assert len(l3) == 1
  assert len(l4) == 1

  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


def test_keras_structural_traits() -> None:
  """Test function."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "keras.Layer"
  assert traits.impurity_methods == ["fit", "compile"]
  assert traits.auto_strip_magic_args is True


def test_keras_plugin_traits() -> None:
  """Test function."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  traits: typing.Any = adapter.plugin_traits
  assert traits.requires_explicit_rng is False


def test_keras_get_tiered_examples() -> None:
  """Test function."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  with patch("ml_switcheroo.frameworks.keras_examples.get_keras_tiered_examples", return_value={"t": "v"}):
    assert adapter.get_tiered_examples() == {"t": "v"}

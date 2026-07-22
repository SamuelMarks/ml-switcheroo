"""Test suite for the Keras module."""

from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo_ir.schema.ghost import SemanticTier
from unittest.mock import patch


def test_keras_adapter_init():
  """Verifies the behavior of Keras adapter initialization."""
  adapter = KerasAdapter()
  assert adapter.display_name == "Keras"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 25


def test_keras_import_alias():
  """Verifies the behavior of Keras import alias."""
  adapter = KerasAdapter()
  assert adapter.import_alias == ("keras", "keras")


def test_keras_import_namespaces():
  """Verifies the behavior of Keras import namespaces."""
  adapter = KerasAdapter()
  ns = adapter.import_namespaces
  assert "keras" in ns
  assert "keras.ops" in ns
  assert "keras.layers" in ns
  assert "numpy" in ns


def test_keras_test_config():
  """Verifies the behavior of Keras test configuration."""
  adapter = KerasAdapter()
  config = adapter.test_config
  assert "import keras" in config["import"]
  assert "keras.ops.convert_to_tensor" in config["convert_input"]


def test_keras_harness_imports():
  """Verifies the behavior of Keras harness imports."""
  adapter = KerasAdapter()
  assert adapter.harness_imports == []


def test_keras_harness_init_code():
  """Verifies the behavior of Keras harness initialization code."""
  adapter = KerasAdapter()
  assert adapter.get_harness_init_code() == ""


def test_keras_get_to_numpy_code():
  """Verifies the behavior of Keras get to NumPy code."""
  adapter = KerasAdapter()
  assert "hasattr(obj, 'numpy')" in adapter.get_to_numpy_code()


def test_keras_supported_tiers():
  """Verifies the behavior of Keras supported tiers."""
  adapter = KerasAdapter()
  tiers = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers
  assert SemanticTier.NEURAL in tiers


def test_keras_declared_magic_args():
  """Verifies the behavior of Keras declared magic arguments."""
  adapter = KerasAdapter()
  assert adapter.declared_magic_args == []


def test_keras_structural_traits():
  """Verifies the behavior of Keras structural traits."""
  adapter = KerasAdapter()
  traits = adapter.structural_traits
  assert traits.module_base == "keras.Layer"
  assert traits.forward_method == "call"
  assert traits.requires_super_init


def test_keras_rng_seed_methods():
  """Verifies the behavior of Keras rng seed methods."""
  adapter = KerasAdapter()
  assert "utils.set_random_seed" in adapter.rng_seed_methods


def test_keras_definitions(monkeypatch):
  """Verifies the behavior of Keras definitions."""
  adapter = KerasAdapter()
  defs = adapter.definitions
  assert isinstance(defs, dict)


def test_keras_device_syntax():
  """Verifies the behavior of Keras device syntax."""
  adapter = KerasAdapter()
  assert "keras.name_scope('gpu')" == adapter.get_device_syntax("cuda")
  assert "keras.name_scope('cpu')" == adapter.get_device_syntax("cpu")


def test_keras_device_check_syntax():
  """Verifies the behavior of Keras device check syntax."""
  adapter = KerasAdapter()
  assert "keras.config.list_logical_devices" in adapter.get_device_check_syntax()


def test_keras_apply_wiring():
  """Verifies the behavior of Keras apply wiring."""
  adapter = KerasAdapter()
  adapter.apply_wiring({})


def test_keras_doc_url():
  """Verifies the behavior of Keras documentation URL."""
  adapter = KerasAdapter()
  url = adapter.get_doc_url("keras.layers.Dense")
  assert "search.html?q=keras.layers.Dense" in url


@patch("ml_switcheroo.frameworks.keras_examples.get_keras_tiered_examples")
def test_keras_tiered_examples(mock_examples):
  """Verifies the behavior of Keras tiered examples."""
  mock_examples.return_value = {"tier2_neural": "some_code"}
  adapter = KerasAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  mock_examples.assert_called_once()


def test_keras_init_ghost_mode(monkeypatch):
  """Verifies the behavior of Keras initialization ghost mode."""
  monkeypatch.setattr("ml_switcheroo.frameworks.keras.keras", None)
  adapter = KerasAdapter()
  assert adapter._mode == InitMode.GHOST


def test_keras_init_live_mode(monkeypatch):
  """Verifies the behavior of Keras initialization live mode."""
  monkeypatch.setattr("ml_switcheroo.frameworks.keras.keras", True)
  adapter = KerasAdapter()
  assert adapter._mode == InitMode.LIVE

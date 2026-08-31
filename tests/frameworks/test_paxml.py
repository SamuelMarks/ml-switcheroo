"""Test suite for the Paxml module."""

import sys
import typing
from unittest.mock import patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.paxml import PaxmlAdapter


def test_paxml_adapter_init() -> None:
  """Verifies the behavior of Paxml adapter initialization."""
  adapter = PaxmlAdapter()
  assert adapter.display_name == "PaxML / Praxis"
  assert adapter.inherits_from == "jax"
  assert adapter.ui_priority == 60
  assert adapter._mode == InitMode.GHOST


def test_paxml_import_alias() -> None:
  """Verifies the behavior of Paxml import alias."""
  adapter = PaxmlAdapter()
  assert adapter.import_alias == ("praxis.layers", "pl")


def test_paxml_import_namespaces() -> None:
  """Verifies the behavior of Paxml import namespaces."""
  adapter = PaxmlAdapter()
  ns: typing.Any = adapter.import_namespaces
  assert "praxis.layers" in ns
  assert "praxis.base_layer" in ns


def test_paxml_test_config() -> None:
  """Docstring."""
  adapter = PaxmlAdapter()
  config: dict[str, typing.Any] = adapter.test_config
  assert "import praxis.layers as pl" in config["import"]


def test_paxml_harness_imports() -> None:
  """Verifies the behavior of Paxml harness imports."""
  adapter = PaxmlAdapter()
  assert "import jax" in adapter.harness_imports


def test_paxml_harness_init_code() -> None:
  """Verifies the behavior of Paxml harness initialization code."""
  adapter = PaxmlAdapter()
  code: str = adapter.get_harness_init_code()
  assert "def _make_jax_key(seed):" in code


def test_paxml_supported_tiers() -> None:
  """Verifies the behavior of Paxml supported tiers."""
  adapter = PaxmlAdapter()
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert SemanticTier.NEURAL in adapter.supported_tiers


def test_paxml_declared_magic_args() -> None:
  """Verifies the behavior of Paxml declared magic arguments."""
  adapter = PaxmlAdapter()
  assert adapter.declared_magic_args == []


def test_paxml_structural_traits() -> None:
  """Verifies the behavior of Paxml structural traits."""
  adapter = PaxmlAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "praxis.base_layer.BaseLayer"
  assert traits.init_method_name == "setup"
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_paxml_definitions() -> None:
  """Verifies the behavior of Paxml definitions."""
  adapter = PaxmlAdapter()
  defs: typing.Any = adapter.definitions
  assert "Linear" in defs
  assert defs["Linear"].args["in_features"] == "input_dims"
  assert defs["Linear"].args["bias"] == "use_bias"


def test_paxml_rng_seed_methods() -> None:
  """Verifies the behavior of Paxml rng seed methods."""
  adapter = PaxmlAdapter()
  assert adapter.rng_seed_methods == []


def test_paxml_convert() -> None:
  """Verifies the behavior of Paxml convert."""
  adapter = PaxmlAdapter()
  assert adapter.convert("test") == "test"


def test_paxml_apply_wiring() -> None:
  """Verifies the behavior of Paxml apply wiring."""
  adapter = PaxmlAdapter()
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_paxml_doc_url() -> None:
  """Verifies the behavior of Paxml documentation URL."""
  adapter = PaxmlAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("praxis.layers.Linear")
  assert url is not None
  assert "github.com" in url


def test_paxml_tiered_examples() -> None:
  """Verifies the behavior of Paxml tiered examples."""
  adapter = PaxmlAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples
  assert "tier4_qwen3-vl" in examples


def test_paxml_init_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Paxml initialization live mode."""
  monkeypatch.setattr("ml_switcheroo.frameworks.paxml.praxis", True)
  adapter = PaxmlAdapter()
  assert adapter._mode == InitMode.LIVE


# --- Merged from test_paxml_extra.py ---


def test_paxml_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  monkeypatch.setitem(sys.modules, "paxml", None)  # type: ignore
  monkeypatch.setitem(sys.modules, "praxis", None)  # type: ignore
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  real_import = __import__

  def fake_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks __import__."""
    if name == "praxis":
      raise ImportError("Fail praxis")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", fake_import):
    importlib.reload(pax_fw)

  adapter = pax_fw.PaxmlAdapter()
  assert adapter._mode.name == "GHOST"

  # Reload without patch to hit standard imports
  importlib.reload(pax_fw)


def test_paxml_collect_ghost_no_snapshot() -> None:
  """Docstring."""
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  importlib.reload(pax_fw)
  adapter = pax_fw.PaxmlAdapter()

  adapter._snapshot_data = None  # type: ignore
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []

  adapter._snapshot_data = {"categories": {"extras": []}}  # type: ignore
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_paxml_convert_fail(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.paxml as pax_fw

  adapter = pax_fw.PaxmlAdapter()

  import ml_switcheroo.frameworks.jax as jax_fw

  with patch.object(jax_fw.JaxCoreAdapter, "convert", return_value="converted"):
    assert adapter.convert([1, 2, 3]) == "converted"


def test_paxml_properties_device_syntax() -> None:
  """Docstring."""
  adapter = PaxmlAdapter()
  syntax: str = adapter.get_device_syntax("cpu")
  assert "jax" in syntax

  check: str = adapter.get_device_check_syntax()
  assert check == "True" or "len" in check


def test_paxml_doc_url_extra() -> None:
  """Docstring."""
  adapter = PaxmlAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("paxml.BaseModel")
  assert url is not None
  assert "github.com" in url


def test_paxml_apply_wiring_extra() -> None:
  """Docstring."""
  adapter = PaxmlAdapter()
  adapter.apply_wiring({})


def test_paxml_defs_missing() -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.paxml as pax_fw

  with patch.object(pax_fw, "load_definitions") as mock_load:
    mock_load.return_value = {}
    adapter = pax_fw.PaxmlAdapter()
    defs: typing.Any = adapter.definitions
    assert "Linear" in defs
    assert "Sequential" in defs
    assert "ReLU" in defs

    class MockLinear:
      """A mock linear module."""

      args = None

    mock_load.return_value = {"Linear": MockLinear(), "Sequential": "s", "ReLU": "r"}
    defs2: typing.Any = pax_fw.PaxmlAdapter().definitions
    assert defs2["Linear"].args is not None


def test_paxml_plugin_traits() -> None:
  """Docstring."""
  adapter = PaxmlAdapter()
  traits: typing.Any = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True
  assert traits.requires_explicit_rng is True
  assert traits.requires_functional_control_flow is True
  assert traits.enforce_purity_analysis is True


def test_paxml_init_logging(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import sys

  monkeypatch.setitem(sys.modules, "paxml", None)  # type: ignore
  monkeypatch.setitem(sys.modules, "praxis", None)  # type: ignore
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  importlib.reload(pax_fw)

  with patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value=None):
    adapter = pax_fw.PaxmlAdapter()
    assert adapter._mode.name == "GHOST"


def test_paxml_ghost_mode_with_snapshot() -> None:
  """Docstring."""
  from unittest.mock import patch
  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  with patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value={"test": 1}):
    adapter = PaxmlAdapter()
    assert adapter._snapshot_data == {"test": 1}

"""Test suite for the Paxml module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

import sys
from pydantic import JsonValue
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
  ns = adapter.import_namespaces
  assert "praxis.layers" in ns
  assert "praxis.base_layer" in ns


def test_paxml_test_config() -> None:
  """Verify paxml test config behavior."""
  adapter = PaxmlAdapter()
  config = adapter.test_config
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
  traits = adapter.structural_traits
  assert traits.module_base == "praxis.base_layer.BaseLayer"
  assert traits.init_method_name == "setup"
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_paxml_definitions() -> None:
  """Verifies the behavior of Paxml definitions."""
  adapter = PaxmlAdapter()
  defs = adapter.definitions
  assert "Linear" in defs
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": defs}, "__getitem__")("v")["Linear"]}, "__getitem__")("v").args}, "__getitem__"
      )("v")["in_features"]
    },
    "__getitem__",
  )("v") == getattr({"v": "input_dims"}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": defs}, "__getitem__")("v")["Linear"]}, "__getitem__")("v").args}, "__getitem__"
      )("v")["bias"]
    },
    "__getitem__",
  )("v") == getattr({"v": "use_bias"}, "__getitem__")("v")


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
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_paxml_doc_url() -> None:
  """Verifies the behavior of Paxml documentation URL."""
  adapter = PaxmlAdapter()
  url: str | None = adapter.get_doc_url("praxis.layers.Linear")
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
  """Verifies the behavior of Paxml initialization live mode.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  monkeypatch.setattr("ml_switcheroo.frameworks.paxml.praxis", True)
  adapter = PaxmlAdapter()
  assert adapter._mode == InitMode.LIVE


def test_paxml_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify paxml init missing behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  monkeypatch.setitem(sys.modules, "paxml", None)
  monkeypatch.setitem(sys.modules, "praxis", None)
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  import types

  real_import = __import__

  def fake_import(name: str, *args: JsonValue, **kwargs: JsonValue) -> types.ModuleType:
    """Mocks __import__.

    Args:
        name: Identifier or operation name string.
        args: The args (JsonValue) for fake import.
        kwargs: The kwargs (JsonValue) for fake import.

    Returns:
        Resulting `types.ModuleType` from fake import.
    """
    if name == "praxis":
      raise ImportError("Fail praxis")
    _ret_v: types.ModuleType = getattr(
      {
        "v": getattr({"v": real_import}, "__getitem__")("v")(
          getattr({"v": name}, "__getitem__")("v"), *args, **getattr({"v": kwargs}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  with patch("builtins.__import__", fake_import):
    importlib.reload(pax_fw)

  adapter = pax_fw.PaxmlAdapter()
  assert adapter._mode.name == "GHOST"

  # Reload without patch to hit standard imports
  importlib.reload(pax_fw)


def test_paxml_collect_ghost_no_snapshot() -> None:
  """Verify paxml collect ghost no snapshot behavior."""
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  importlib.reload(pax_fw)
  adapter = pax_fw.PaxmlAdapter()

  setattr(getattr({"v": adapter}, "__getitem__")("v"), "_snapshot_data", getattr({"v": None}, "__getitem__")("v"))
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []

  adapter._snapshot_data = {"categories": {"extras": []}}
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_paxml_convert_fail(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify paxml convert fail behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import ml_switcheroo.frameworks.paxml as pax_fw

  adapter = pax_fw.PaxmlAdapter()

  import ml_switcheroo.frameworks.jax as jax_fw

  with patch.object(jax_fw.JaxCoreAdapter, "convert", return_value="converted"):
    assert adapter.convert([1, 2, 3]) == "converted"


def test_paxml_properties_device_syntax() -> None:
  """Verify paxml properties device syntax behavior."""
  adapter = PaxmlAdapter()
  syntax: str = adapter.get_device_syntax("cpu")
  assert "jax" in syntax

  check: str = adapter.get_device_check_syntax()
  assert check == "True" or "len" in check


def test_paxml_doc_url_extra() -> None:
  """Verify paxml doc url extra behavior."""
  adapter = PaxmlAdapter()
  url: str | None = adapter.get_doc_url("paxml.BaseModel")
  assert url is not None
  assert "github.com" in url


def test_paxml_apply_wiring_extra() -> None:
  """Verify paxml apply wiring extra behavior."""
  adapter = PaxmlAdapter()
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_paxml_defs_missing() -> None:
  """Verify paxml defs missing behavior."""
  import ml_switcheroo.frameworks.paxml as pax_fw

  with patch.object(pax_fw, "load_definitions") as mock_load:
    mock_load.return_value = {}
    adapter = pax_fw.PaxmlAdapter()
    defs = adapter.definitions
    assert "Linear" in defs
    assert "Sequential" in defs
    assert "ReLU" in defs

    class MockLinear:
      """A mock linear module."""

      args = None

    mock_load.return_value = {"Linear": MockLinear(), "Sequential": "s", "ReLU": "r"}
    defs2 = pax_fw.PaxmlAdapter().definitions
    assert defs2["Linear"].args is not None


def test_paxml_plugin_traits() -> None:
  """Verify paxml plugin traits behavior."""
  adapter = PaxmlAdapter()
  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True
  assert traits.requires_explicit_rng is True
  assert traits.requires_functional_control_flow is True
  assert traits.enforce_purity_analysis is True


def test_paxml_init_logging(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify paxml init logging behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import sys

  monkeypatch.setitem(sys.modules, "paxml", None)
  monkeypatch.setitem(sys.modules, "praxis", None)
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  importlib.reload(pax_fw)

  with patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value=None):
    adapter = pax_fw.PaxmlAdapter()
    assert adapter._mode.name == "GHOST"


def test_paxml_ghost_mode_with_snapshot() -> None:
  """Verify paxml ghost mode with snapshot behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  with patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value={"test": 1}):
    adapter = PaxmlAdapter()
    assert adapter._snapshot_data == {"test": 1}

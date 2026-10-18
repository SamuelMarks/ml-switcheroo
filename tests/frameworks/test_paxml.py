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
  adapter: typing.Any = PaxmlAdapter()
  assert adapter.display_name == "PaxML / Praxis"
  assert adapter.inherits_from == "jax"
  assert adapter.ui_priority == 60
  assert adapter._mode == InitMode.GHOST


def test_paxml_import_alias() -> None:
  """Verifies the behavior of Paxml import alias."""
  adapter: typing.Any = PaxmlAdapter()
  assert adapter.import_alias == ("praxis.layers", "pl")


def test_paxml_import_namespaces() -> None:
  """Verifies the behavior of Paxml import namespaces."""
  adapter: typing.Any = PaxmlAdapter()
  ns: typing.Any = adapter.import_namespaces
  assert "praxis.layers" in ns
  assert "praxis.base_layer" in ns


def test_paxml_test_config() -> None:
  """Verify paxml test config behavior."""
  adapter: typing.Any = PaxmlAdapter()
  config: dict[str, typing.Any] = adapter.test_config
  assert "import praxis.layers as pl" in config["import"]


def test_paxml_harness_imports() -> None:
  """Verifies the behavior of Paxml harness imports."""
  adapter: typing.Any = PaxmlAdapter()
  assert "import jax" in adapter.harness_imports


def test_paxml_harness_init_code() -> None:
  """Verifies the behavior of Paxml harness initialization code."""
  adapter: typing.Any = PaxmlAdapter()
  code: str = adapter.get_harness_init_code()
  assert "def _make_jax_key(seed):" in code


def test_paxml_supported_tiers() -> None:
  """Verifies the behavior of Paxml supported tiers."""
  adapter: typing.Any = PaxmlAdapter()
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert SemanticTier.NEURAL in adapter.supported_tiers


def test_paxml_declared_magic_args() -> None:
  """Verifies the behavior of Paxml declared magic arguments."""
  adapter: typing.Any = PaxmlAdapter()
  assert adapter.declared_magic_args == []


def test_paxml_structural_traits() -> None:
  """Verifies the behavior of Paxml structural traits."""
  adapter: typing.Any = PaxmlAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "praxis.base_layer.BaseLayer"
  assert traits.init_method_name == "setup"
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_paxml_definitions() -> None:
  """Verifies the behavior of Paxml definitions."""
  adapter: typing.Any = PaxmlAdapter()
  defs: typing.Any = adapter.definitions
  assert "Linear" in defs
  assert defs["Linear"].args["in_features"] == "input_dims"
  assert defs["Linear"].args["bias"] == "use_bias"


def test_paxml_rng_seed_methods() -> None:
  """Verifies the behavior of Paxml rng seed methods."""
  adapter: typing.Any = PaxmlAdapter()
  assert adapter.rng_seed_methods == []


def test_paxml_convert() -> None:
  """Verifies the behavior of Paxml convert."""
  adapter: typing.Any = PaxmlAdapter()
  assert adapter.convert("test") == "test"


def test_paxml_apply_wiring() -> None:
  """Verifies the behavior of Paxml apply wiring."""
  adapter: typing.Any = PaxmlAdapter()
  snapshot: dict[str, typing.Any] = {}
  adapter.apply_wiring(snapshot)
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_paxml_doc_url() -> None:
  """Verifies the behavior of Paxml documentation URL."""
  adapter: typing.Any = PaxmlAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("praxis.layers.Linear")
  assert url is not None
  assert "github.com" in url


def test_paxml_tiered_examples() -> None:
  """Verifies the behavior of Paxml tiered examples."""
  adapter: typing.Any = PaxmlAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples
  assert "tier4_qwen3-vl" in examples


def test_paxml_init_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Paxml initialization live mode.

  Args:
      monkeypatch: Test argument or fixture.
  """
  monkeypatch.setattr("ml_switcheroo.frameworks.paxml.praxis", True)
  adapter: typing.Any = PaxmlAdapter()
  assert adapter._mode == InitMode.LIVE


# --- Merged from test_paxml_extra.py ---


def test_paxml_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify paxml init missing behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  monkeypatch.setitem(sys.modules, "paxml", None)
  monkeypatch.setitem(sys.modules, "praxis", None)
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  real_import: typing.Any = __import__

  def fake_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks __import__.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "praxis":
      raise ImportError("Fail praxis")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", fake_import):
    importlib.reload(pax_fw)

  adapter: typing.Any = pax_fw.PaxmlAdapter()
  assert adapter._mode.name == "GHOST"

  # Reload without patch to hit standard imports
  importlib.reload(pax_fw)


def test_paxml_collect_ghost_no_snapshot() -> None:
  """Verify paxml collect ghost no snapshot behavior."""
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  importlib.reload(pax_fw)
  adapter: typing.Any = pax_fw.PaxmlAdapter()

  adapter._snapshot_data = None
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []

  adapter._snapshot_data = {"categories": {"extras": []}}
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_paxml_convert_fail(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify paxml convert fail behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import ml_switcheroo.frameworks.paxml as pax_fw

  adapter: typing.Any = pax_fw.PaxmlAdapter()

  import ml_switcheroo.frameworks.jax as jax_fw

  with patch.object(jax_fw.JaxCoreAdapter, "convert", return_value="converted"):
    assert adapter.convert([1, 2, 3]) == "converted"


def test_paxml_properties_device_syntax() -> None:
  """Verify paxml properties device syntax behavior."""
  adapter: typing.Any = PaxmlAdapter()
  syntax: str = adapter.get_device_syntax("cpu")
  assert "jax" in syntax

  check: str = adapter.get_device_check_syntax()
  assert check == "True" or "len" in check


def test_paxml_doc_url_extra() -> None:
  """Verify paxml doc url extra behavior."""
  adapter: typing.Any = PaxmlAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("paxml.BaseModel")
  assert url is not None
  assert "github.com" in url


def test_paxml_apply_wiring_extra() -> None:
  """Verify paxml apply wiring extra behavior."""
  adapter: typing.Any = PaxmlAdapter()
  snapshot: dict[str, typing.Any] = {}
  _res: typing.Any = adapter.apply_wiring(snapshot)
  assert _res is None
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_paxml_defs_missing() -> None:
  """Verify paxml defs missing behavior."""
  import ml_switcheroo.frameworks.paxml as pax_fw

  with patch.object(pax_fw, "load_definitions") as mock_load:
    mock_load.return_value = {}
    adapter: typing.Any = pax_fw.PaxmlAdapter()
    defs: typing.Any = adapter.definitions
    assert "Linear" in defs
    assert "Sequential" in defs
    assert "ReLU" in defs

    class MockLinear:
      """A mock linear module."""

      args: typing.Any = None

    mock_load.return_value = {"Linear": MockLinear(), "Sequential": "s", "ReLU": "r"}
    defs2: typing.Any = pax_fw.PaxmlAdapter().definitions
    assert defs2["Linear"].args is not None


def test_paxml_plugin_traits() -> None:
  """Verify paxml plugin traits behavior."""
  adapter: typing.Any = PaxmlAdapter()
  traits: typing.Any = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True
  assert traits.requires_explicit_rng is True
  assert traits.requires_functional_control_flow is True
  assert traits.enforce_purity_analysis is True


def test_paxml_init_logging(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify paxml init logging behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import sys

  monkeypatch.setitem(sys.modules, "paxml", None)
  monkeypatch.setitem(sys.modules, "praxis", None)
  import importlib

  import ml_switcheroo.frameworks.paxml as pax_fw

  importlib.reload(pax_fw)

  with patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value=None):
    adapter: typing.Any = pax_fw.PaxmlAdapter()
    assert adapter._mode.name == "GHOST"


def test_paxml_ghost_mode_with_snapshot() -> None:
  """Verify paxml ghost mode with snapshot behavior."""
  from unittest.mock import patch

  from ml_switcheroo.frameworks.paxml import PaxmlAdapter

  with patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value={"test": 1}):
    adapter: typing.Any = PaxmlAdapter()
    assert adapter._snapshot_data == {"test": 1}

"""Test module."""

import typing
import sys
import pytest
from unittest.mock import patch
from ml_switcheroo.frameworks.paxml import PaxmlAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_paxml_init_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test function."""
  monkeypatch.setitem(sys.modules, "paxml", None)  # type: ignore
  monkeypatch.setitem(sys.modules, "praxis", None)  # type: ignore
  import ml_switcheroo.frameworks.paxml as pax_fw
  import importlib

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
  """Test function."""
  import ml_switcheroo.frameworks.paxml as pax_fw
  import importlib

  importlib.reload(pax_fw)
  adapter = pax_fw.PaxmlAdapter()

  adapter._snapshot_data = None  # type: ignore
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []

  adapter._snapshot_data = {"categories": {"extras": []}}  # type: ignore
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_paxml_convert_fail(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test function."""
  import ml_switcheroo.frameworks.paxml as pax_fw

  adapter = pax_fw.PaxmlAdapter()

  import ml_switcheroo.frameworks.jax as jax_fw

  with patch.object(jax_fw.JaxCoreAdapter, "convert", return_value="converted"):
    assert adapter.convert([1, 2, 3]) == "converted"


def test_paxml_properties_device_syntax() -> None:
  """Test function."""
  adapter = PaxmlAdapter()
  syntax: str = adapter.get_device_syntax("cpu")
  assert "jax" in syntax

  check: str = adapter.get_device_check_syntax()
  assert check == "True" or "len" in check


def test_paxml_doc_url() -> None:
  """Test function."""
  adapter = PaxmlAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("paxml.BaseModel")
  assert url is not None
  assert "github.com" in url


def test_paxml_apply_wiring() -> None:
  """Test function."""
  adapter = PaxmlAdapter()
  adapter.apply_wiring({})


def test_paxml_defs_missing() -> None:
  """Test function."""
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
  """Test function."""
  adapter = PaxmlAdapter()
  traits: typing.Any = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True
  assert traits.requires_explicit_rng is True
  assert traits.requires_functional_control_flow is True
  assert traits.enforce_purity_analysis is True


def test_paxml_init_logging(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test function."""
  import sys

  monkeypatch.setitem(sys.modules, "paxml", None)  # type: ignore
  monkeypatch.setitem(sys.modules, "praxis", None)  # type: ignore
  import ml_switcheroo.frameworks.paxml as pax_fw
  import importlib

  importlib.reload(pax_fw)

  with patch("ml_switcheroo.frameworks.paxml.load_snapshot_for_adapter", return_value=None):
    adapter = pax_fw.PaxmlAdapter()
    assert adapter._mode.name == "GHOST"

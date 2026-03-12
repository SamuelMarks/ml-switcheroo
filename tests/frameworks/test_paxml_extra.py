import pytest
import sys
from unittest.mock import MagicMock, patch
from ml_switcheroo.frameworks.paxml import PaxmlAdapter
from ml_switcheroo.frameworks.base import StandardMap, StandardCategory


def test_paxml_missing_imports(monkeypatch):
  import ml_switcheroo.frameworks.paxml as paxml_module

  monkeypatch.setattr(paxml_module, "praxis", None)
  adapter = PaxmlAdapter()
  res = adapter._scan_praxis_layers()
  assert res == []


def test_paxml_scan_praxis_layers(monkeypatch):
  import ml_switcheroo.frameworks.paxml as paxml_module

  # mock praxis
  praxis_mock = MagicMock()

  class FakeBaseLayer:
    pass

  class ValidLayer(FakeBaseLayer):
    pass

  class HeuristicLayer:
    pass

  # to hit line 146
  praxis_mock.base_layer.BaseLayer = FakeBaseLayer

  # line 140
  import inspect

  def mock_getmembers(mod):
    if getattr(mod, "__name__", "") == "mod1":
      return [("_hidden", None), ("ValidLayer", ValidLayer), ("HeuristicLayer", HeuristicLayer), ("NotClass", 123)]
    return []

  monkeypatch.setattr(inspect, "getmembers", mock_getmembers)
  monkeypatch.setattr(inspect, "isclass", lambda x: isinstance(x, type))

  # line 133, 135
  mod1 = MagicMock()
  mod1.__name__ = "mod1"
  mod2 = MagicMock()
  mod2.__name__ = "mod2"
  mod3 = MagicMock()
  mod3.__name__ = "mod3"

  praxis_mock.layers = mod1
  praxis_mock.layers.activations = mod2
  praxis_mock.layers.normalizations = mod3

  monkeypatch.setattr(paxml_module, "praxis", praxis_mock)

  # To hit 154-159
  class DummyInspector:
    @staticmethod
    def inspect(obj, path):
      if obj == HeuristicLayer:
        raise Exception("boom")
      return MagicMock()

  monkeypatch.setattr(paxml_module, "GhostInspector", DummyInspector)

  adapter = PaxmlAdapter()
  res = adapter._scan_praxis_layers()

  # to hit 136-137: exception when accessing activations
  type(praxis_mock.layers).__getattr__ = MagicMock(side_effect=AttributeError)
  res2 = adapter._scan_praxis_layers()


def test_paxml_definitions_missing():
  adapter = PaxmlAdapter()

  # line 337, 341, 347, 351
  # We can mock load_definitions to return an empty dict, or a dict where Linear has no args
  with patch(
    "ml_switcheroo.frameworks.paxml.load_definitions",
    return_value={"Linear": StandardMap(api="praxis.layers.Linear", args=None)},
  ):
    defs = adapter.definitions
    assert defs["Linear"].args is not None
    assert "Sequential" in defs
    assert "ReLU" in defs

  with patch("ml_switcheroo.frameworks.paxml.load_definitions", return_value={}):
    defs = adapter.definitions
    assert "Linear" in defs
    assert "Sequential" in defs
    assert "ReLU" in defs


def test_paxml_discover_layer_missing_snapshot():
  adapter = PaxmlAdapter()
  adapter._snapshot_data = None
  res = adapter._collect_ghost(StandardCategory.LAYER)
  assert res == []

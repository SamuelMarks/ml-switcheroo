"""Extra tests for the Paxml framework adapter."""

import sys


def test_paxml_import_success(monkeypatch):
  """Test paxml framework imports when praxis is available."""
  import types

  mock_praxis = types.ModuleType("praxis")
  mock_praxis.layers = types.ModuleType("praxis.layers")
  mock_praxis.base_layer = types.ModuleType("praxis.base_layer")
  mock_praxis.layers.activations = types.ModuleType("praxis.layers.activations")
  mock_praxis.layers.normalizations = types.ModuleType("praxis.layers.normalizations")

  monkeypatch.setitem(sys.modules, "praxis", mock_praxis)
  monkeypatch.setitem(sys.modules, "praxis.layers", mock_praxis.layers)
  monkeypatch.setitem(sys.modules, "praxis.base_layer", mock_praxis.base_layer)
  monkeypatch.setitem(sys.modules, "praxis.layers.activations", mock_praxis.layers.activations)
  monkeypatch.setitem(sys.modules, "praxis.layers.normalizations", mock_praxis.layers.normalizations)

  # Force reload of paxml
  if "ml_switcheroo.frameworks.paxml" in sys.modules:
    del sys.modules["ml_switcheroo.frameworks.paxml"]

  import ml_switcheroo.frameworks.paxml as paxml_mod

  assert paxml_mod.praxis is not None

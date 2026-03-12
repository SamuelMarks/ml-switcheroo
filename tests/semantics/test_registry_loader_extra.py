import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.semantics.registry_loader import RegistryLoader
import ml_switcheroo.semantics.registry_loader as registry_loader


def test_registry_loader_exceptions(monkeypatch, capsys):
  # test 48: adapter is None
  def mock_get(fw):
    if fw == "dummy":
      return None
    elif fw == "dummy_traits":

      class BadTraitsAdapter:
        class FakeTraits:
          def model_dump(self, **kwargs):
            raise ValueError("bad traits")

        structural_traits = FakeTraits()

      return BadTraitsAdapter()
    elif fw == "dummy_wiring":

      class BadWiringAdapter:
        def apply_wiring(self, snap):
          raise ValueError("bad wiring")

      return BadWiringAdapter()
    return None

  monkeypatch.setattr(registry_loader, "get_adapter", mock_get)
  monkeypatch.setattr(registry_loader, "available_frameworks", lambda: ["dummy", "dummy_traits", "dummy_wiring"])

  manager = MagicMock()
  manager.framework_configs = {"dummy_traits": {}, "dummy_wiring": {}}
  loader = RegistryLoader(manager)

  loader._hydrate_adapters()

  out, err = capsys.readouterr()
  assert "Failed to load structural traits for dummy_traits" in out
  assert "Failed to apply wiring for dummy_wiring" in out

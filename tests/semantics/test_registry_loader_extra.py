"""Test suite for the Registry Loader Extra module."""

from unittest.mock import MagicMock
from ml_switcheroo.semantics.registry_loader import RegistryLoader
import ml_switcheroo.semantics.registry_loader as registry_loader


def test_registry_loader_exceptions(monkeypatch, capsys):
  """Verifies the behavior of registry loader exceptions."""

  def mock_get(fw):
    """Provides a mock get for testing."""
    if fw == "dummy":
      return None
    elif fw == "dummy_traits":

      class BadTraitsAdapter:
        """Test suite for the Bad Traits Adapter component."""

        class FakeTraits:
          """Fake Traits class for testing purposes."""

          def model_dump(self, **kwargs):
            """Mock implementation of model dump."""
            raise ValueError("bad traits")

        structural_traits = FakeTraits()

      return BadTraitsAdapter()
    elif fw == "dummy_wiring":

      class BadWiringAdapter:
        """Test suite for the Bad Wiring Adapter component."""

        def apply_wiring(self, snap):
          """Applies wiring."""
          raise ValueError("bad wiring")

      return BadWiringAdapter()
    return None

  monkeypatch.setattr(registry_loader, "get_adapter", mock_get)
  monkeypatch.setattr(registry_loader, "available_frameworks", lambda: ["dummy", "dummy_traits", "dummy_wiring"])
  manager = MagicMock()
  manager.framework_configs = {"dummy_traits": {}, "dummy_wiring": {}}
  loader = RegistryLoader(manager)
  loader._hydrate_adapters()
  (out, err) = capsys.readouterr()
  assert "Failed to load structural traits for dummy_traits" in out
  assert "Failed to apply wiring for dummy_wiring" in out

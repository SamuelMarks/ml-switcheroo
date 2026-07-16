"""Auto-generated doc."""

from unittest.mock import MagicMock
from ml_switcheroo.semantics.registry_loader import RegistryLoader
import ml_switcheroo.semantics.registry_loader as registry_loader


def test_registry_loader_exceptions(monkeypatch, capsys):
  """Auto-generated doc."""

  # test 48: adapter is None
  def mock_get(fw):
    """Auto-generated doc."""
    if fw == "dummy":
      return None
    elif fw == "dummy_traits":

      class BadTraitsAdapter:
        """Auto-generated doc."""

        class FakeTraits:
          """Auto-generated doc."""

          def model_dump(self, **kwargs):
            """Auto-generated doc."""
            raise ValueError("bad traits")

        structural_traits = FakeTraits()

      return BadTraitsAdapter()
    elif fw == "dummy_wiring":

      class BadWiringAdapter:
        """Auto-generated doc."""

        def apply_wiring(self, snap):
          """Auto-generated doc."""
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

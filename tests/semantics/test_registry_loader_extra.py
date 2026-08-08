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


def test_registry_loader_prelabel_and_plugin_metadata(monkeypatch):
  """Verifies the pre-labeling of lowercase keys and plugin metadata loading."""
  # test pre-label
  manager = MagicMock()
  manager._key_origins = {}
  loader = RegistryLoader(manager)

  # For pre-labeling to happen in _hydrate_adapters, we need a valid adapter with definitions
  class ValidAdapter:
    @property
    def definitions(self):
      class MockDef:
        def model_dump(self, **kwargs):
          return {}

      return {"lower_case_op": MockDef(), "UpperCaseOp": MockDef()}

  monkeypatch.setattr(registry_loader, "get_adapter", lambda _: ValidAdapter())
  monkeypatch.setattr(registry_loader, "available_frameworks", lambda: ["valid"])
  manager.framework_configs = {"valid": {}}

  loader._hydrate_adapters()

  # Check line 157
  assert manager._key_origins.get("lower_case_op") == "array"
  assert manager._key_origins.get("UpperCaseOp") == "neural"

  # Check lines 247-248
  # mock hooks.get_all_hook_metadata
  class MockSpec:
    ops = {"plugin_op": {"frameworks": {"jax": {}}}}

  monkeypatch.setattr(registry_loader.hooks, "get_all_hook_metadata", lambda: {"my_plugin": MockSpec()})
  manager.data = {}
  loader._hydrate_plugins()

  # verify merge_tier_data was called by inspecting the mocked merge or side effect
  # since we use MagicMock, we just check that manager._key_origins got the new origin
  assert manager._key_origins.get("plugin_op") == "extras"

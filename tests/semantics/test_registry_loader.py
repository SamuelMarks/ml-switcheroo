"""Test suite for the Registry Loader Extra module."""

import typing

from typing import Any
from unittest.mock import MagicMock


import ml_switcheroo.semantics.registry_loader as registry_loader
from ml_switcheroo.semantics.registry_loader import RegistryLoader


def test_registry_loader_exceptions(monkeypatch: typing.Any, caplog: typing.Any) -> None:
  """Verifies the behavior of registry loader exceptions.

  Args:
      monkeypatch (pytest.MonkeyPatch): Pytest fixture.
      caplog (pytest.LogCaptureFixture): Pytest fixture.
  """

  def mock_get(fw: typing.Any) -> typing.Any:
    """Provides a mock get for testing.

    Args:
        fw (str): Framework string.

    Returns:
        Optional[Any]: Adapter or None.
    """
    if fw == "dummy":
      return None
    elif fw == "dummy_traits":

      class BadTraitsAdapter:
        """Test helper class BadTraitsAdapter."""

        class FakeTraits:
          """Test helper class FakeTraits."""

          def model_dump(self, **kwargs: Any) -> Any:
            """Mock implementation of model dump.

            Args:
                **kwargs (Any): Keyword arguments.

            Raises:
                ValueError: Exception.

            Returns:
                Computed test result or fixture value.
            """
            raise ValueError("bad traits")

        structural_traits: typing.Any = FakeTraits()
        plugin_traits: typing.Any = FakeTraits()

      return BadTraitsAdapter()
    elif fw == "dummy_wiring":

      class BadWiringAdapter:
        """Test helper class BadWiringAdapter."""

        def apply_wiring(self, snap: Any) -> None:
          """Applies wiring.

          Args:
              snap (Any): Snapshot argument.

          Raises:
              ValueError: Exception.
          """
          raise ValueError("bad wiring")

      return BadWiringAdapter()
    return None

  monkeypatch.setattr(registry_loader, "get_adapter", mock_get)
  monkeypatch.setattr(registry_loader, "available_frameworks", lambda: ["dummy", "dummy_traits", "dummy_wiring"])
  manager: typing.Any = MagicMock()
  manager.framework_configs = {"dummy_traits": {}, "dummy_wiring": {}}
  loader: typing.Any = RegistryLoader(manager)
  loader._hydrate_adapters()
  assert "Failed to load structural traits for dummy_traits" in caplog.text
  assert "Failed to apply wiring for dummy_wiring" in caplog.text


def test_registry_loader_prelabel_and_plugin_metadata(monkeypatch: typing.Any) -> None:
  """Verifies the pre-labeling of lowercase keys and plugin metadata loading.

  Args:
      monkeypatch (pytest.MonkeyPatch): Pytest fixture.
  """
  # test pre-label
  manager: typing.Any = MagicMock()
  manager._key_origins = {}
  loader: typing.Any = RegistryLoader(manager)

  # For pre-labeling to happen in _hydrate_adapters, we need a valid adapter with definitions
  class ValidAdapter:
    """Valid adapter."""

    @property
    def definitions(self) -> typing.Any:
      """Definitions.

      Returns:
          dict: Dictionary of definitions.
      """

      class MockDef:
        """Mock def."""

        def model_dump(self, **kwargs: Any) -> typing.Any:
          """Model dump.

          Args:
              **kwargs (Any): Keyword arguments.

          Returns:
              dict: Empty dictionary.
          """
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
    """Mock spec."""

    ops: typing.Any = {"plugin_op": {"frameworks": {"jax": {}}}}

  monkeypatch.setattr(registry_loader.hooks, "get_all_hook_metadata", lambda: {"my_plugin": MockSpec()})
  manager.data = {}
  loader._hydrate_plugins()

  # verify merge_tier_data was called by inspecting the mocked merge or side effect
  # since we use MagicMock, we just check that manager._key_origins got the new origin
  assert manager._key_origins.get("plugin_op") == "extras"


def test_index_variants_edge_cases() -> None:
  """Test index_variants when _variant_cache is missing, variants is not dict, or variant is None."""
  # Case 1: no _variant_cache
  mgr: typing.Any = MagicMock()
  del mgr._variant_cache
  loader: typing.Any = RegistryLoader(mgr)
  loader.index_variants()

  # Case 2: variants is not dict, variant is None
  mgr2: typing.Any = MagicMock()
  mgr2._variant_cache = {}
  mgr2.data = {
    "op1": {"variants": "not_a_dict"},
    "op2": {"variants": {"jax": None, "torch": {"api": "torch.op2"}}},
    "op3": "not_a_dict_details",
  }
  loader2: typing.Any = RegistryLoader(mgr2)
  loader2.index_variants()
  assert ("op2", "torch") in mgr2._variant_cache
  assert ("op2", "jax") not in mgr2._variant_cache


def test_registry_loader_remaining_branches(monkeypatch: typing.Any) -> None:
  """Test remaining branches in registry loader.

  Args:
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  manager: typing.Any = MagicMock()
  manager.framework_configs = {"test_fw": {}}
  manager._key_origins = {}
  manager._providers = {}
  manager._source_registry = {}
  manager.data = {}

  class MockSpec:
    """Mock spec."""

    def model_dump(self, **kwargs: Any) -> typing.Any:
      """Model dump.

      Args:
          **kwargs (Any): Keyword arguments.

      Returns:
          dict: Empty dictionary.
      """
      return {}

  class TestAdapter:
    """Adapter with None traits, lowercase spec, and non-ImportConfig namespace."""

    structural_traits: typing.Any = None
    plugin_traits: typing.Any = None
    specifications: typing.Any = {"lower_spec": MockSpec()}
    import_namespaces: typing.Any = {"path.to.module": "not_an_import_config"}

  monkeypatch.setattr(registry_loader, "get_adapter", lambda _: TestAdapter())
  monkeypatch.setattr(registry_loader, "available_frameworks", lambda: ["test_fw"])

  loader: typing.Any = RegistryLoader(manager)
  loader._hydrate_adapters()
  assert "traits" not in manager.framework_configs["test_fw"]

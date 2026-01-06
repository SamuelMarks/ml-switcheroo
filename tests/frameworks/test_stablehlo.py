"""
Tests for StableHLO Framework Adapter.

Verifies:
1.  Registration in `_ADAPTER_REGISTRY`.
2.  Protocol compliance (properties and methods return safe defaults).
3.  `create_emitter` hook correctness.
4.  End-to-End emission via the internal facade.
5.  Documentation URL generation.
"""

import pytest
from unittest.mock import patch, MagicMock

from ml_switcheroo.frameworks.stablehlo import StableHloAdapter
from ml_switcheroo.frameworks.base import (
  _ADAPTER_REGISTRY,
  get_adapter,
  InitMode,
  StandardCategory,
)


def test_registration():
  """Verify adapter is registered under 'stablehlo' key."""
  assert "stablehlo" in _ADAPTER_REGISTRY
  adapter = get_adapter("stablehlo")
  assert isinstance(adapter, StableHloAdapter)
  assert adapter.display_name == "StableHLO (MLIR)"
  assert adapter.ui_priority == 95


def test_initialization_defaults():
  """Verify default ghost mode and empty lists."""
  adapter = StableHloAdapter()
  assert adapter._mode == InitMode.GHOST
  assert adapter.search_modules == []
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("stablehlo", "stablehlo")
  assert adapter.import_namespaces == {}
  assert adapter.discovery_heuristics == {}


def test_emitter_factory():
  """Verify create_emitter returns a usable object."""
  adapter = StableHloAdapter()
  emitter = adapter.create_emitter()
  assert hasattr(emitter, "emit")
  assert callable(emitter.emit)


def test_emitter_integration(tmp_path):
  """
  Verify the facade correctly instantiates logic and returns a string.

  Scenario: Input `x = torch.abs(y)`
  Expectation: `stablehlo.abs` in output.
  """
  adapter = StableHloAdapter()
  emitter = adapter.create_emitter()

  code = "x = torch.abs(y)"

  # Mock SemanticsManager inside the emitter to ensure torch.abs is mapped
  # Patch must target where the class is defined because it's imported locally
  with patch("ml_switcheroo.semantics.manager.SemanticsManager") as MockMgr:
    mgr_instance = MockMgr.return_value
    # Configure get_definition to return stablehlo mapping
    mgr_instance.get_definition.return_value = ("Abs", {"variants": {"stablehlo": {"api": "stablehlo.abs"}}})

    mlir_output = emitter.emit(code)

  assert "stablehlo.abs" in mlir_output
  assert "tensor<*xf32>" in mlir_output


def test_emitter_syntax_error_handling():
  """Verify robust error handling on bad python code."""
  adapter = StableHloAdapter()
  emitter = adapter.create_emitter()

  mlir_output = emitter.emit("INVALID PYTHON CODE >>>")
  assert "// Error parsing Python source" in mlir_output


def test_doc_url_generation():
  """Verify GitHub deep link generation."""
  adapter = StableHloAdapter()
  url = adapter.get_doc_url("stablehlo.abs")
  assert url == "https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs"

  # Fallback
  assert adapter.get_doc_url("unknown.op") is None


def test_stub_methods_safety():
  """Verify all protocol methods return safe values."""
  adapter = StableHloAdapter()

  assert adapter.get_device_syntax("cuda") == "// Target: cuda"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("r", "k") == ""
  assert adapter.get_serialization_syntax("save", "f") == ""
  assert adapter.collect_api(StandardCategory.LAYER) == []
  assert adapter.convert(123) == "123"

  # Weight methods
  assert "# Weights not supported" in adapter.get_weight_load_code("p")
  assert "return str(obj)" == adapter.get_to_numpy_code()


def test_traits_and_definitions():
  """Verify trait accessors."""
  adapter = StableHloAdapter()
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None

  # Verify definitions loading logic (mocks allow seeing if load_definitions is called)
  with patch("ml_switcheroo.frameworks.stablehlo.load_definitions") as mock_load:
    _ = adapter.definitions
    mock_load.assert_called_with("stablehlo")


def test_example_code():
  """Verify example getter."""
  code = StableHloAdapter.get_example_code()
  assert "stablehlo.abs" in code

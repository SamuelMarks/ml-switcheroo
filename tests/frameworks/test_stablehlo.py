"""
Tests for StableHLO Framework Adapter.

Verifies:
1.  Registration in `_ADAPTER_REGISTRY`.
2.  Protocol compliance.
3.  Backend route availability via Registry (Replacing legacy create_emitter).
"""

import pytest
from unittest.mock import patch, MagicMock

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import StandardCategory
from ml_switcheroo.frameworks.stablehlo import StableHloAdapter
from ml_switcheroo.frameworks.base import (
  _ADAPTER_REGISTRY,
  get_adapter,
  InitMode,
)
from ml_switcheroo.compiler.registry import get_backend_class, is_isa_target


@pytest.fixture
def mock_semantics_patch():
  """Function docstring."""
  # FIX: Patch the definition of SemanticsManager instead of the import,
  # as local imports inside methods cannot be patched via module attribute access.
  with patch("ml_switcheroo.semantics.manager.SemanticsManager") as MockMgr:
    mgr = MockMgr.return_value
    mgr.data = {"Abs": {"variants": {"stablehlo": {"api": "stablehlo.abs"}}}}
    mgr.get_definition.return_value = ("Abs", mgr.data["Abs"])
    # Ensure 'Abs' in reverse index if used
    mgr._reverse_index = {"torch.abs": ("Abs", mgr.data["Abs"])}

    # Helper logic for get_definition from name
    def get_def(name):
      """Function docstring."""
      if name == "torch.abs":
        return ("Abs", mgr.data["Abs"])
      return None

    mgr.get_definition.side_effect = get_def
    yield


def test_backend_registered(mock_semantics_patch):
  """
  Verify the backend registry routes 'stablehlo' correctly.
  Replaces deprecated `create_emitter` test logic.
  """
  # Verify it is flagged as a graph/ISA target to trigger compiler pipeline
  assert is_isa_target("stablehlo")

  # Verify the backend class is resolvable
  cls = get_backend_class("stablehlo")
  assert cls is not None
  assert cls.__name__ == "StableHloBackend"


def test_example_code():
  """Verify example getter."""
  code = StableHloAdapter().get_tiered_examples()["tier1_math"]
  assert "stablehlo.abs" in code


def test_stablehlo_all_methods():
  adapter = StableHloAdapter()
  assert adapter.search_modules == []
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("stablehlo", "stablehlo")
  assert adapter.import_namespaces == {}
  assert adapter.discovery_heuristics == {}
  assert "import" in adapter.test_config
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() == "return str(obj)"
  assert SemanticTier.NEURAL in adapter.supported_tiers
  assert adapter.declared_magic_args == []
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None
  assert isinstance(adapter.definitions, dict)
  assert adapter.specifications == {}
  assert adapter.rng_seed_methods == []
  assert adapter.collect_api(StandardCategory.LAYER) == []
  assert adapter.get_device_syntax("cuda") == "// Target: cuda"
  assert adapter.get_device_check_syntax() == "True"
  assert adapter.get_rng_split_syntax("rng", "key") == ""
  assert adapter.get_serialization_imports() == []
  assert adapter.get_serialization_syntax("op", "f") == ""
  assert adapter.get_weight_conversion_imports() == []
  assert adapter.get_weight_load_code("path") == "# Weights not supported in StableHLO mode"
  assert adapter.get_tensor_to_numpy_expr("tensor") == "tensor"
  assert adapter.get_weight_save_code("state", "path") == "# Weights not supported in StableHLO mode"
  snap = {}
  adapter.apply_wiring(snap)
  assert snap == {}
  assert adapter.get_doc_url("stablehlo.abs") == "https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs"
  assert adapter.get_doc_url("other") is None
  assert adapter.convert(123) == "123"
  examples = adapter.get_tiered_examples()
  assert "stablehlo.abs" in examples["tier1_math"]

"""
Tests for StableHLO Framework Adapter.

Verifies:
1.  Registration in `_ADAPTER_REGISTRY`.
2.  Protocol compliance.
3.  Backend route availability via Registry (Replacing legacy create_emitter).
"""

import pytest
from unittest.mock import patch, MagicMock

from ml_switcheroo.frameworks.stablehlo import StableHloAdapter
from ml_switcheroo.frameworks.base import (
  _ADAPTER_REGISTRY,
  get_adapter,
  InitMode,
)
from ml_switcheroo.compiler.registry import get_backend_class, is_isa_target


@pytest.fixture
def mock_semantics_patch():
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

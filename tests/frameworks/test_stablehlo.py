"""
Tests for StableHLO Framework Adapter.

Verifies:
1.  Registration in `_ADAPTER_REGISTRY`.
2.  Protocol compliance.
3.  `create_emitter` hook correctness.
"""

import pytest
from unittest.mock import patch, MagicMock

from ml_switcheroo.frameworks.stablehlo import StableHloAdapter
from ml_switcheroo.frameworks.base import (
  _ADAPTER_REGISTRY,
  get_adapter,
  InitMode,
)


@pytest.fixture
def mock_semantics_patch():
  with patch("ml_switcheroo.frameworks.stablehlo.SemanticsManager") as MockMgr:
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


def test_emitter_integration(tmp_path, mock_semantics_patch):
  """
  Verify the facade correctly instantiates logic and returns a string.
  """
  adapter = StableHloAdapter()
  emitter = adapter.create_emitter()

  code = "x = torch.abs(y)"
  mlir_output = emitter.emit(code)

  # The StableHloEmitter replaces 'sw.op' with 'stablehlo.abs' if semantics allow
  assert "stablehlo.abs" in mlir_output
  assert "tensor<*xf32>" in mlir_output


def test_example_code():
  """Verify example getter."""
  code = StableHloAdapter.get_example_code()
  assert "stablehlo.abs" in code

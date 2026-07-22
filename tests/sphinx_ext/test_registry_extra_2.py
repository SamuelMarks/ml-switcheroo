"""Test suite for the Registry Extra 2 module."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.sphinx_ext.registry import scan_registry
import json


@patch("ml_switcheroo.sphinx_ext.registry.available_frameworks")
@patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.registry.get_adapter")
def test_scan_registry_no_candidates(mock_get_adapter, mock_priority, mock_avail):
  """Scans registry no candidates."""
  mock_avail.return_value = ["torch"]
  mock_priority.return_value = ["torch"]

  def get_adapter_side_effect(name):
    if name == "torch":
      adapter = MagicMock()
      adapter.inherits_from = None
      adapter.display_name = "PyTorch"
      adapter.supported_tiers = None
      adapter.get_tiered_examples.return_value = {"tier3_extras": "torch_extras"}
      return adapter
    return None

  mock_get_adapter.side_effect = get_adapter_side_effect
  (hierarchy, examples_json, tier_metadata_json) = scan_registry()
  examples = json.loads(examples_json)
  assert "torch_tier3_extras" in examples
  assert examples["torch_tier3_extras"]["requiredTier"] == "extras"

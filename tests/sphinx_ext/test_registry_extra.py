"""Test suite for the Registry Extra module."""

from typing import Dict, Any
from unittest.mock import patch, MagicMock
from ml_switcheroo.sphinx_ext.registry import scan_registry
import json


@patch("ml_switcheroo.sphinx_ext.registry.available_frameworks")
@patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.registry.get_adapter")
def test_scan_registry_extra(mock_get_adapter: MagicMock, mock_priority: MagicMock, mock_avail: MagicMock) -> None:
  """Scans registry extra.

  Args:
      mock_get_adapter (MagicMock): Mock argument.
      mock_priority (MagicMock): Mock argument.
      mock_avail (MagicMock): Mock argument.
  """
  mock_avail.return_value = ["unknown"]
  mock_priority.return_value = ["jax"]

  def get_adapter_side_effect(name: str) -> MagicMock:
    """Effect.

    Args:
        name (str): The name argument.

    Returns:
        MagicMock: A mock adapter.
    """
    if name == "unknown":
      adapter: MagicMock = MagicMock()
      adapter.inherits_from = None
      adapter.display_name = "Unknown"
      adapter.supported_tiers = None
      adapter.get_tiered_examples.return_value = {"tier1_math": "unknown_math", "tier2_neural": "unknown_nn"}
      return adapter
    return None

  mock_get_adapter.side_effect = get_adapter_side_effect
  hierarchy, examples_json, tier_metadata_json = scan_registry()
  examples: Dict[str, Any] = json.loads(examples_json)
  assert examples["unknown_tier1_math"]["tgtFw"] == "jax"

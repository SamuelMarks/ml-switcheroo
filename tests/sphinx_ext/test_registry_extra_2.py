"""Test suite for the Registry Extra 2 module."""

from typing import Dict, Any
from unittest.mock import patch, MagicMock
from ml_switcheroo.sphinx_ext.registry import scan_registry
import json


@patch("ml_switcheroo.sphinx_ext.registry.available_frameworks")
@patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.registry.get_adapter")
def test_scan_registry_no_candidates(
  mock_get_adapter: MagicMock, mock_priority: MagicMock, mock_avail: MagicMock
) -> None:
  """Scans registry no candidates.

  Args:
      mock_get_adapter (MagicMock): Mock argument.
      mock_priority (MagicMock): Mock argument.
      mock_avail (MagicMock): Mock argument.
  """
  mock_avail.return_value = ["torch"]
  mock_priority.return_value = ["torch"]

  def get_adapter_side_effect(name: str) -> MagicMock:
    """Effect.

    Args:
        name (str): The name argument.

    Returns:
        MagicMock: A mock adapter.
    """
    if name == "torch":
      adapter: MagicMock = MagicMock()
      adapter.inherits_from = None
      adapter.display_name = "PyTorch"
      adapter.supported_tiers = None
      adapter.get_tiered_examples.return_value = {"tier3_extras": "torch_extras"}
      return adapter
    return None

  mock_get_adapter.side_effect = get_adapter_side_effect
  hierarchy, examples_json, tier_metadata_json = scan_registry()
  examples: Dict[str, Any] = json.loads(examples_json)
  assert "torch_tier3_extras" in examples
  assert examples["torch_tier3_extras"]["requiredTier"] == "extras"


@patch("ml_switcheroo.sphinx_ext.registry.available_frameworks")
@patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.registry.get_adapter")
def test_scan_registry_else_branch(mock_get_adapter: MagicMock, mock_priority: MagicMock, mock_avail: MagicMock) -> None:
  """Hits the else branch when candidates are not in priorities.

  Args:
      mock_get_adapter (MagicMock): Mock argument.
      mock_priority (MagicMock): Mock argument.
      mock_avail (MagicMock): Mock argument.
  """
  mock_avail.return_value = ["torch", "unknown_fw"]
  mock_priority.return_value = ["torch", "other_fw"]

  def get_adapter_side_effect(name: str) -> MagicMock:
    """Get adapter side effect.

    Args:
        name (str): The name argument.

    Returns:
        MagicMock: A mock adapter.
    """
    adapter: MagicMock = MagicMock()
    adapter.inherits_from = None
    adapter.display_name = name
    adapter.supported_tiers = None
    if name == "torch":
      adapter.get_tiered_examples.return_value = {"tier3_extras": "torch_extras"}
    else:
      adapter.get_tiered_examples.return_value = {"tier3_extras": "unknown_extras"}
    return adapter

  mock_get_adapter.side_effect = get_adapter_side_effect
  scan_registry()

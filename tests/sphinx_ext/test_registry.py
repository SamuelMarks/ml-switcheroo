"""Test suite for the Registry module."""

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from ml_switcheroo.sphinx_ext.registry import scan_registry


@patch("ml_switcheroo.sphinx_ext.registry.available_frameworks")
@patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.registry.get_adapter")
def test_scan_registry_else_fallback(
  mock_get_adapter: MagicMock, mock_priority: MagicMock, mock_avail: MagicMock
) -> None:
  """Tests the else fallback when no other candidates are in the priority list.

  Args:
      mock_get_adapter (MagicMock): Mock argument.
      mock_priority (MagicMock): Mock argument.
      mock_avail (MagicMock): Mock argument.
  """
  mock_avail.return_value = ["torch", "custom1", "custom2"]
  mock_priority.return_value = ["torch", "custom1"]

  def get_adapter_side_effect(name: str) -> MagicMock:
    """Effect.

    Args:
        name (str): The name argument.

    Returns:
        MagicMock: A mock adapter.
    """
    adapter: MagicMock = MagicMock()
    adapter.inherits_from = None
    adapter.display_name = name.title()
    adapter.supported_tiers = None
    adapter.get_tiered_examples.return_value = {"example1": f"{name}_code"}
    return adapter

  mock_get_adapter.side_effect = get_adapter_side_effect
  hierarchy, examples_json, tier_metadata_json = scan_registry()
  examples: Dict[str, Any] = json.loads(examples_json)
  # torch will fall back to custom1 (first candidate)
  assert examples["torch_example1"]["tgtFw"] == "custom1"


@patch("ml_switcheroo.sphinx_ext.registry.available_frameworks")
@patch("ml_switcheroo.sphinx_ext.registry.get_framework_priority_order")
@patch("ml_switcheroo.sphinx_ext.registry.get_adapter")
def test_scan_registry(mock_get_adapter: MagicMock, mock_priority: MagicMock, mock_avail: MagicMock) -> None:
  """Scans registry.

  Args:
      mock_get_adapter (MagicMock): Mock argument.
      mock_priority (MagicMock): Mock argument.
      mock_avail (MagicMock): Mock argument.
  """
  mock_avail.return_value = ["torch", "jax", "flax_nnx", "unknown"]
  mock_priority.return_value = ["torch", "jax"]

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
      adapter.get_tiered_examples.return_value = {"tier1_math": "torch_math"}
      return adapter
    elif name == "jax":
      adapter = MagicMock()
      adapter.inherits_from = None
      adapter.display_name = "JAX"
      tier: MagicMock = MagicMock()
      tier.value = "array"
      adapter.supported_tiers = [tier]
      adapter.get_tiered_examples.return_value = {"tier2_neural": "jax_nn"}
      return adapter
    elif name == "flax_nnx":
      adapter = MagicMock()
      adapter.inherits_from = "jax"
      adapter.display_name = "Flax"
      adapter.supported_tiers = None
      adapter.get_tiered_examples.return_value = {"tier1_math": "flax_math"}
      return adapter
    return None

  mock_get_adapter.side_effect = get_adapter_side_effect
  hierarchy, examples_json, tier_metadata_json = scan_registry()
  assert "torch" in hierarchy
  assert "jax" in hierarchy
  assert len(hierarchy["jax"]) == 1
  assert hierarchy["jax"][0]["key"] == "flax_nnx"
  examples: Dict[str, Any] = json.loads(examples_json)
  assert "torch_tier1_math" in examples
  assert "jax_tier2_neural" in examples
  assert "flax_nnx_tier1_math" in examples
  assert examples["torch_tier1_math"]["tgtFw"] == "jax"
  tier_metadata: Dict[str, Any] = json.loads(tier_metadata_json)
  assert "torch" in tier_metadata
  assert "array" in tier_metadata["jax"]


# --- Merged from test_registry_extra.py ---


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


# --- Merged from test_registry_extra_2.py ---


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

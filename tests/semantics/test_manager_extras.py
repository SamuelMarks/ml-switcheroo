"""Test extra corner cases for SemanticsManager."""

from ml_switcheroo.semantics.manager import SemanticsManager


def test_manager_alias_branches() -> None:
  """Test branches in _build_index and get_framework_aliases."""
  mgr = SemanticsManager()
  # Inject framework configs to hit missing branches
  mgr.framework_configs["no_alias_fw"] = {"some_other_key": "val"}
  mgr.framework_configs["missing_alias_fields_fw"] = {"alias": {"module": "only_mod"}}
  mgr.framework_configs["missing_mod_alias_fw"] = {"alias": {"name": "only_name"}}

  # Trigger _build_index which iterates over framework_configs and checks "alias"
  mgr._build_index()

  # Trigger get_framework_aliases which iterates over framework_configs
  aliases = mgr.get_framework_aliases()

  # Ensure they were safely ignored
  assert "no_alias_fw" not in aliases
  assert "missing_alias_fields_fw" not in aliases
  assert "missing_mod_alias_fw" not in aliases


def test_manager_get_variant_case_insensitive_lookup() -> None:
  """Test resolve_variant when candidate_id matches an existing definition and target_fw exists."""
  mgr = SemanticsManager()
  mgr.data["relu"] = {
    "operation": "relu",
    "variants": {"torch": {"api": "torch.relu"}},
  }
  # Also add an entry for RELU without torch variant to pass the first checks
  mgr.data["RELU"] = {
    "operation": "RELU",
    "variants": {"jax": {"api": "jax.nn.relu"}},
  }
  # Clear cache to ensure lookup occurs
  mgr._variant_cache.clear()
  res = mgr.resolve_variant("RELU", "torch")
  assert res == {"api": "torch.relu"}

  # Test when alt_defn exists but target_fw not in alt_vars (hits 311->306 branch)
  mgr.data["OTHER"] = {
    "operation": "OTHER",
    "variants": {},
  }
  mgr.data["other"] = {
    "operation": "other",
    "variants": {"flax": {"api": "flax.other"}},
  }
  mgr._variant_cache.clear()
  res2 = mgr.resolve_variant("OTHER", "torch")
  assert res2 is None

  # Test lowercase abstract_id where candidate_id == abstract_id (hits candidate_id == abstract_id)
  mgr.data["lower_only"] = {
    "operation": "lower_only",
    "variants": {},
  }
  mgr._variant_cache.clear()
  res3 = mgr.resolve_variant("lower_only", "torch")
  assert res3 is None


def test_manager_update_definition_all_fields_provided() -> None:
  """Test update_definition when operation, variants, description, and std_args are all provided."""
  from unittest.mock import mock_open, patch

  mgr = SemanticsManager()
  new_op = {
    "operation": "custom_op",
    "variants": {"torch": {"api": "torch.custom_op"}},
    "description": "Full description",
    "std_args": ["x"],
    "frameworks": {},
  }
  m_open = mock_open()
  with patch("builtins.open", m_open):
    mgr.update_definition("custom_op", new_op)
  assert mgr.data["custom_op"]["operation"] == "custom_op"
  assert mgr.data["custom_op"]["description"] == "Full description"
  assert mgr.data["custom_op"]["std_args"] == ["x"]


def test_manager_update_definition_with_description() -> None:
  """Test update_definition when description is provided to hit 422->424 branch."""
  from unittest.mock import mock_open, patch

  mgr = SemanticsManager()
  new_op = {
    "operation": "test.op",
    "variants": {},
    "description": "Provided description",
    "std_args": [],
    "frameworks": {},
  }
  m_open = mock_open()
  with patch("builtins.open", m_open):
    mgr.update_definition("test.op", new_op)
  assert mgr.data["test.op"]["description"] == "Provided description"

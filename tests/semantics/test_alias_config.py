"""
Tests for Data-Driven Framework Alias Configuration.

Verifies:
1.  SemanticsManager parses alias config from `__frameworks__`.
2.  ImportFixer respects injected aliases.
3.  Default behavior is preserved.
"""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.enums import SemanticTier


def test_manager_parses_json_alias_config():
  """
  Scenario: Load JSON with alias override.
  Expectation: get_framework_aliases returns override.
  """
  mgr = SemanticsManager()
  # Ensure clean state for test instance since defaults are loaded in init
  # Note: _DEFAULT_ALIASES constant in module will populate defaults.
  # We want to verify override logic.

  mock_data = {
    "__frameworks__": {
      "custom_fw": {"alias": {"module": "custom.lib", "name": "cl"}},
      "jax": {
        # Override default jax alias
        "alias": {"module": "jax.custom", "name": "jc"}
      },
    }
  }

  mgr._merge_tier(mock_data, SemanticTier.EXTRAS)

  aliases = mgr.get_framework_aliases()

  # Check new entry
  assert "custom_fw" in aliases
  assert aliases["custom_fw"] == ("custom.lib", "cl")

  # Check override
  assert aliases["jax"] == ("jax.custom", "jc")

  # Check default preservation
  assert "numpy" in aliases
  assert aliases["numpy"] == ("numpy", "np")


def test_import_fixer_uses_injected_aliases():
  """
  Scenario: ImportFixer initialized with custom map.
  Expectation: Code generation uses the map.
  """
  # Custom map: jax -> import jaxoid as jxD
  alias_map = {"jax": ("jaxoid", "jXd")}

  fixer = ImportFixer(
    source_fw="torch",
    target_fw="jax",
    submodule_map={},
    alias_map=alias_map,
  )

  # Input code uses the alias 'jXd' (simulating rewriter output or user usage)
  code = "y = jXd.array([1])"
  tree = cst.parse_module(code)
  new_tree = tree.visit(fixer)
  result = new_tree.code

  assert "import jaxoid as jXd" in result
  assert "import jax.numpy" not in result


def test_engine_integration_mock(monkeypatch):
  """
  Verify that ASTEngine logic fetches aliases (simulated logic).
  Since ASTEngine is tightly coupled to real SemanticsManager, we verify logic flow.
  """
  # This test mimics the new lines added to engine.py
  mgr = MagicMock()
  mgr.get_framework_aliases.return_value = {"jax": ("mock_jax", "mj")}
  mgr.get_import_map.return_value = {}

  # Simulate Engine.run logic segment
  # ... code rewrite ...
  target_fw = "jax"
  code = "y = mj.abs(x)"
  tree = cst.parse_module(code)

  alias_map = mgr.get_framework_aliases()
  fixer = ImportFixer("torch", target_fw, submodule_map={}, alias_map=alias_map)
  new_tree = tree.visit(fixer)
  res = new_tree.code

  # Should use the mocked alias
  assert "import mock_jax as mj" in res

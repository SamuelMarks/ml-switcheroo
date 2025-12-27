"""
Tests for Data-Driven Framework Alias Configuration.

Verifies:
1.  SemanticsManager parses alias config from registry auto-discovery.
2.  SemanticsManager overrides registry defaults with JSON config.
3.  ImportFixer respects resulting aliases.
"""

import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import merge_tier_data
from ml_switcheroo.core.import_fixer import ImportFixer
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks import register_framework, get_adapter


def test_manager_uses_registry_defaults():
  """
  Scenario: No JSON.
  Expect: Aliases come from Adapter class properties.
  """
  # Assuming standard adapters are registered by default in __init__
  mgr = SemanticsManager()
  mgr._reverse_index = {}

  aliases = mgr.get_framework_aliases()

  assert "jax" in aliases
  # JAX Adapter defines ("jax.numpy", "jnp")
  assert aliases["jax"] == ("jax.numpy", "jnp")


def test_manager_picks_up_new_framework():
  """
  Scenario: User adds a 'fastai' adapter at runtime (Zero-Edit extension).
  Expect: SemanticsManager initialized *after* registration includes it.
  """

  # 1. Verify it doesn't exist yet (sanity check, clean env?)
  # Just register a new one to be sure
  class FastAIAdapter:
    import_alias = ("fastai.vision", "fv")

    # dummy required methods
    def convert(self, x):
      return x

  register_framework("fastai_test")(FastAIAdapter)

  # 2. Init Manager
  mgr = SemanticsManager()
  mgr._reverse_index = {}

  aliases = mgr.get_framework_aliases()

  # 3. Verify
  assert "fastai_test" in aliases
  assert aliases["fastai_test"] == ("fastai.vision", "fv")


def test_manager_parses_json_alias_override():
  """
  Scenario: Load JSON with alias override.
  Expectation: JSON override supercedes Adapter default.
  """
  # 1. Setup Manager (loads adapters implicitly)
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  if not hasattr(mgr, "import_data"):
    mgr.import_data = {}

  # 2. Inject JSON override via internal merge method
  mock_data = {
    "__frameworks__": {
      "jax": {
        # Override default jax alias (jnp -> jc)
        "alias": {"module": "jax.custom", "name": "jc"}
      },
    }
  }

  # Fix: Use merge_tier_data utility instead of removed method
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    import_data=mgr.import_data,
    framework_configs=mgr.framework_configs,
    new_content=mock_data,
    tier=SemanticTier.EXTRAS,
  )

  aliases = mgr.get_framework_aliases()

  # Check override
  assert aliases["jax"] == ("jax.custom", "jc")


def test_import_fixer_uses_injected_aliases():
  """
  Scenario: ImportFixer initialized with custom map.
  Expectation: Code generation uses the map.
  """
  # Custom map: jax -> import jaxoid as jXd
  alias_map = {"jax": ("jaxoid", "jXd")}

  # Updated: use source_fws (plural) to match new signature
  fixer = ImportFixer(
    source_fws="torch",
    target_fw="jax",
    submodule_map={},
    alias_map=alias_map,
  )

  # Input code uses the alias 'jXd' (simulating rewriter output or user usage)
  code = "y = jXd.array([1])"
  tree = cst.parse_module(code)
  new_tree = tree.visit(fixer)
  result = new_tree.code

  # FIX: Assertion now matches the alias defined in alias_map ('jXd')
  assert "import jaxoid as jXd" in result
  assert "import jax.numpy" not in result

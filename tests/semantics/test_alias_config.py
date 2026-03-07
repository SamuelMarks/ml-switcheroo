"""
Tests for Data-Driven Framework Alias Configuration.
"""

import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import merge_tier_data
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks import register_framework


def test_manager_uses_registry_defaults():
  """Function docstring."""
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  aliases = mgr.get_framework_aliases()
  assert "jax" in aliases
  assert aliases["jax"] == ("jax.numpy", "jnp")


def test_manager_picks_up_new_framework():
  """Function docstring."""

  class FastAIAdapter:
    """Class docstring."""

    import_alias = ("fastai.vision", "fv")

    def convert(self, x):
      """Function docstring."""
      return x

  register_framework("fastai_test")(FastAIAdapter)
  mgr = SemanticsManager()
  aliases = mgr.get_framework_aliases()
  assert "fastai_test" in aliases
  assert aliases["fastai_test"] == ("fastai.vision", "fv")


def test_manager_parses_json_alias_override():
  """Function docstring."""
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  if not hasattr(mgr, "import_data"):
    mgr.import_data = {}

  mock_data = {
    "__frameworks__": {
      "jax": {"alias": {"module": "jax.custom", "name": "jc"}},
    }
  }

  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=mock_data,
    tier=SemanticTier.EXTRAS,
  )

  aliases = mgr.get_framework_aliases()
  assert aliases["jax"] == ("jax.custom", "jc")


def test_import_fixer_uses_injected_aliases():
  """Function docstring."""
  alias_map = {"jax": ("jaxoid", "jXd")}

  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_framework_aliases.return_value = alias_map
  mgr.get_import_map.return_value = {}

  code = "y = jXd.array([1])"
  tree = cst.parse_module(code)

  resolver = ImportResolver(mgr)
  plan = resolver.resolve(tree, "jax")

  fixer = ImportFixer(plan=plan, source_fws={"torch"})

  new_tree = tree.visit(fixer)
  result = new_tree.code

  assert "import jaxoid as jXd" in result
  assert "import jax.numpy" not in result

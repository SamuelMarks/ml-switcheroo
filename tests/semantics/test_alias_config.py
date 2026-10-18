"""Test suite for the Alias Config module."""

import typing

from typing import Any
from unittest.mock import MagicMock

import libcst as cst
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver
from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.merging import merge_tier_data


def test_manager_uses_registry_defaults() -> None:
  """Verifies the behavior of manager uses registry defaults."""
  mgr: typing.Any = SemanticsManager()
  mgr._reverse_index = {}
  aliases: typing.Any = mgr.get_framework_aliases()
  assert "jax" in aliases
  assert aliases["jax"] == ("jax.numpy", "jnp")


def test_manager_picks_up_new_framework() -> None:
  """Verifies the behavior of manager picks up new framework."""

  class FastAIAdapter:
    """Test helper class FastAIAdapter."""

    import_alias: typing.Any = ("fastai.vision", "fv")

    def convert(self, x: Any) -> Any:
      """Converts .

      Args:
          x (Any): Item.

      Returns:
          Any: Result.
      """
      return x

  register_framework("fastai_test")(FastAIAdapter)
  mgr: typing.Any = SemanticsManager()
  aliases: typing.Any = mgr.get_framework_aliases()
  assert "fastai_test" in aliases
  assert aliases["fastai_test"] == ("fastai.vision", "fv")


def test_manager_parses_json_alias_override() -> None:
  """Verifies the behavior of manager parses JSON alias override."""
  mgr: typing.Any = SemanticsManager()
  mgr._reverse_index = {}
  if not hasattr(mgr, "import_data"):
    mgr.import_data = {}
  mock_data: typing.Any = {"__frameworks__": {"jax": {"alias": {"module": "jax.custom", "name": "jc"}}}}
  merge_tier_data(
    data=mgr.data,
    key_origins=mgr._key_origins,
    framework_configs=mgr.framework_configs,
    new_content=mock_data,
    tier=SemanticTier.EXTRAS,
  )
  aliases: typing.Any = mgr.get_framework_aliases()
  assert aliases["jax"] == ("jax.custom", "jc")


def test_import_fixer_uses_injected_aliases() -> None:
  """Verifies the behavior of import fixer uses injected aliases."""
  alias_map: typing.Any = {"jax": ("jaxoid", "jXd")}
  mgr: typing.Any = MagicMock(spec=SemanticsManager)
  mgr.get_framework_aliases.return_value = alias_map
  mgr.get_import_map.return_value = {}
  code: typing.Any = "y = jXd.array([1])"
  tree: typing.Any = cst.parse_module(code)
  resolver: typing.Any = ImportResolver(mgr)
  plan: Any = resolver.resolve(tree, "jax")
  fixer: typing.Any = ImportFixer(plan=plan, source_fws={"torch"})
  new_tree: typing.Any = tree.visit(fixer)
  result: typing.Any = new_tree.code
  assert "import jaxoid as jXd" in result
  assert "import jax.numpy" not in result

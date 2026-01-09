"""
Tests for PaxML Default Alias Configuration via Adapter.
"""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver


def test_paxml_alias_loaded_from_adapter():
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  aliases = mgr.get_framework_aliases()

  assert "paxml" in aliases
  module_path, alias_name = aliases["paxml"]
  assert module_path == "praxis.layers"
  assert alias_name == "pl"


def test_import_fixer_injects_pl_alias():
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  resolver = ImportResolver(mgr)

  source_code = """ 
def setup(self): 
    self.layer = pl.Linear(10, 20) 
"""
  tree = cst.parse_module(source_code)

  plan = resolver.resolve(tree, "paxml")
  fixer = ImportFixer(plan, source_fws={"torch"})

  new_tree = tree.visit(fixer)
  generated_code = new_tree.code

  assert "import praxis.layers as pl" in generated_code
  assert "import paxml" not in generated_code

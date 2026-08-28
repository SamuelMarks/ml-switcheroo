"""Test suite for the Manager Paxml Defaults module."""

import libcst as cst
from typing import Dict, Tuple, Any
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.import_fixer import ImportFixer, ImportResolver


def test_paxml_alias_loaded_from_adapter() -> None:
  """Verifies the behavior of Paxml alias loaded from adapter."""
  mgr: SemanticsManager = SemanticsManager()
  mgr._reverse_index = {}
  aliases: Dict[str, Tuple[str, str]] = mgr.get_framework_aliases()
  assert "paxml" in aliases
  (module_path, alias_name) = aliases["paxml"]
  assert module_path == "praxis.layers"
  assert alias_name == "pl"


def test_import_fixer_injects_pl_alias() -> None:
  """Verifies the behavior of import fixer injects pl alias."""
  mgr: SemanticsManager = SemanticsManager()
  mgr._reverse_index = {}
  resolver: ImportResolver = ImportResolver(mgr)
  source_code: str = "\ndef setup(self):\n    self.layer = pl.Linear(10, 20)\n"
  tree: cst.Module = cst.parse_module(source_code)
  plan: Any = resolver.resolve(tree, "paxml")
  fixer: ImportFixer = ImportFixer(plan, source_fws={"torch"})
  new_tree: cst.Module = tree.visit(fixer)
  generated_code: str = new_tree.code
  assert "import praxis.layers as pl" in generated_code
  assert "import paxml" not in generated_code

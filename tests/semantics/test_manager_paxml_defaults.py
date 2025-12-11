"""
Tests for PaxML Default Alias Configuration via Adapter.

Verifies that:
1. `PaxmlAdapter` provides the correct alias config.
2. The ImportFixer correctly injects this alias when targeting 'paxml'.
"""

import libcst as cst
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.import_fixer import ImportFixer


def test_paxml_alias_loaded_from_adapter():
  """
  Verify SemanticsManager gets alias from PaxmlAdapter.
  """
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  aliases = mgr.get_framework_aliases()

  assert "paxml" in aliases

  module_path, alias_name = aliases["paxml"]
  assert module_path == "praxis.layers"
  assert alias_name == "pl"


def test_import_fixer_injects_pl_alias():
  """
  Verify that ImportFixer uses the manager's alias configuration to inject
  'import praxis.layers as pl' when 'pl' is used in the target code.
  """

  # 1. Setup Manager & Aliases
  mgr = SemanticsManager()
  mgr._reverse_index = {}
  alias_map = mgr.get_framework_aliases()

  # ensure we have empty map, no submodule overrides
  submodule_map = {}

  # 2. Input Code (Simulating a conversion where 'pl.Linear' was generated)
  source_code = """
def setup(self):
    self.layer = pl.Linear(10, 20)
"""

  # 3. Run Fixer targeting 'paxml'
  tree = cst.parse_module(source_code)
  fixer = ImportFixer(source_fw="torch", target_fw="paxml", submodule_map=submodule_map, alias_map=alias_map)

  new_tree = tree.visit(fixer)
  generated_code = new_tree.code

  # 4. Assert Injection
  # Should see: import praxis.layers as pl
  assert "import praxis.layers as pl" in generated_code

  # Should not see incorrect 'import paxml'
  assert "import paxml" not in generated_code

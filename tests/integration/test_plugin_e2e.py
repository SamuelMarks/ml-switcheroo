"""
End-to-End Integration Test for Plugins.

Verifies that the entire plugin pipeline works:
CLI -> Engine -> Semantics -> Hooks -> Plugin Logic -> Output.

This test uses a locally defined hook to verify infrastructure without relying
on specific built-in plugins that may change or be removed.
"""

import pytest
import libcst as cst
from typing import Set

from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import _HOOKS, register_hook, HookContext, clear_hooks

# --- Mock Knowledge Base for E2E ---


class E2EMockSemantics(SemanticsManager):
  """
  Mock Semantics Manager for End-to-End plugin testing.

  Defines a test operation 'MagicOp' that requires a plugin to translate
  from 'torch' to 'jax'.
  """

  def __init__(self):
    """Initialize with an empty state and inject 'MagicOp' definition."""
    # Skip super init to avoid file loading
    self.data = {}
    self.import_data = {}
    self.framework_configs = {}
    self._known_rng_methods = set()

    # Define operation with plugin requirement
    self.data["MagicOp"] = {
      "std_args": ["x"],
      "variants": {
        "torch": {"api": "torch.magic"},
        "jax": {"api": "jax.magic", "requires_plugin": "e2e_test_hook"},
      },
    }

    # Setup reverse index for lookup
    self._reverse_index = {
      "torch.magic": ("MagicOp", self.data["MagicOp"]),
    }

    # Initialize required attributes for discovery/fixer
    self._providers = {}
    self._source_registry = {}
    self._key_origins = {}
    self._validation_status = {}

  def get_all_rng_methods(self) -> Set[str]:
    """Return empty set for RNG methods."""
    return self._known_rng_methods

  def get_definition(self, name):
    """Mock reverse lookup."""
    return self._reverse_index.get(name)

  def get_import_map(self, _target):
    """Return empty import map."""
    return {}

  def get_known_apis(self):
    """Return data dictionary."""
    return self.data


@pytest.fixture
def mock_cli_semantics(monkeypatch):
  """Patches the SemanticsManager used in the CLI handler."""
  # Updated path to point to 'commands' module where it is instantiated
  monkeypatch.setattr("ml_switcheroo.cli.handlers.convert.SemanticsManager", E2EMockSemantics)


@pytest.fixture(autouse=True)
def register_test_hooks():
  """
  Registers a local test hook for the duration of the test.
  Ensures a clean state before and after.
  """
  clear_hooks()

  @register_hook("e2e_test_hook")
  def transform_magic_op(node: cst.Call, ctx: HookContext) -> cst.CSTNode:
    """
    Test Hook: Transforms `torch.magic(x)` to `jax.magic_resolved(x)`.
    Also injects a comment to prove control flow.
    """
    # 1. Verify context
    assert ctx.source_fw == "torch"
    assert ctx.target_fw == "jax"

    # 2. Modify Node
    return node.with_changes(func=cst.Name("magic_resolved"))

  yield

  clear_hooks()


def test_plugin_pipeline_e2e(tmp_path, mock_cli_semantics, register_test_hooks):
  """
  Verify complete flow: CLI execution triggers the registered 'e2e_test_hook'.

  Steps:
  1. Write source file with `torch.magic(x)`.
  2. Run `ml_switcheroo convert`.
  3. Verify output contains `magic_resolved(x)`.
  """
  # 1. Source Input
  infile = tmp_path / "torch_source.py"
  infile.write_text("z = torch.magic(x)\n", encoding="utf-8")
  outfile = tmp_path / "jax_output.py"

  # 2. Execute CLI
  args = ["convert", str(infile), "--out", str(outfile), "--source", "torch", "--target", "jax"]

  try:
    main(args)
  except SystemExit as e:
    assert e.code == 0

  # 3. Validation
  assert outfile.exists()
  content = outfile.read_text("utf-8")

  # The plugin should have renamed the function
  assert "magic_resolved(x)" in content
  assert "torch.magic" not in content

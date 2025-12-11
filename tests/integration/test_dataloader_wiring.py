"""
Integration Tests for DataLoader Plugin Wiring via Generation.

Verifies that:
1. The Scaffolder logic now includes static defaults for DataLoaders.
2. Running `scaffold` produces a `k_framework_extras.json` containing the correct
   `requires_plugin: convert_dataloader` directive and the required Namespace definitions.
3. The ASTEngine correctly picks up this configuration and executes the transformation
   without tripping strict mode safety checks on `torch.utils` attributes.
"""

import sys
import json
from unittest.mock import patch, MagicMock

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.discovery.scaffolder import Scaffolder
from ml_switcheroo.core.hooks import load_plugins
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch


def test_generation_and_execution_flow(tmp_path):
  """
  Scenario:
     1. Run Scaffolder in a clean temp directory.
     2. Verify valid JSON generation.
     3. Initialize Engine using that generated JSON.
     4. Convert code to verify the plugin fires (End-to-End).
  """
  # --- Phase 1: Generation ---

  # We use an empty semantics manager to start clean (no pre-loaded files)
  empty_mgr = SemanticsManager()
  # Force clean slate
  empty_mgr.data = {}
  empty_mgr._key_origins = {}

  # Initialize Scaffolder
  scaffolder = Scaffolder(semantics=empty_mgr)
  # Mock the inspector to avoid needing real torch installed for this test
  # We return empty dicts because we are testing the STATIC injection logic of scaffold()
  scaffolder.inspector.inspect = MagicMock(return_value={})

  # Run scaffold targeting the temp path
  # We pass a dummy framework list, but the static injection should happen regardless
  scaffolder.scaffold(["torch"], tmp_path)

  # Check File Creation
  extras_path = tmp_path / "k_framework_extras.json"
  assert extras_path.exists(), "Scaffolder failed to create extras file"

  # Verify Content
  data = json.loads(extras_path.read_text(encoding="utf-8"))
  assert "DataLoader" in data, "DataLoader definition was not injected into JSON"
  assert "torch.utils.data" in data, "Namespace definition for torch.utils.data was not injected"

  dataloader_def = data["DataLoader"]
  assert dataloader_def["variants"]["jax"]["requires_plugin"] == "convert_dataloader"
  assert dataloader_def["variants"]["jax"]["api"] == "GenericDataLoader"

  # --- Phase 2: Execution ---

  # Ensure plugins are loaded
  if "ml_switcheroo.plugins" not in sys.modules:
    load_plugins()

  # Initialize Manager pointed at the generated files
  # We patch resolve_semantics_dir to make the manager load from our temp path
  with patch("ml_switcheroo.semantics.manager.resolve_semantics_dir", return_value=tmp_path):
    # Use real manager logic to load the file we just made
    mgr = SemanticsManager()

    # Verify Manager loaded it
    loaded_def = mgr.get_definition_by_id("DataLoader")
    assert loaded_def is not None

    # Configure Engine
    config = RuntimeConfig(
      source_framework="torch",
      # We target JAX to trigger the plugin path
      target_framework="jax",
      strict_mode=True,
    )

    engine = ASTEngine(semantics=mgr, config=config)

    # Run Conversion
    source_code = """ 
import torch
ds = [1, 2, 3] 
# This call should trigger the plugin
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, drop_last=False) 
"""
    result = engine.run(source_code)

    # Debug info if fails
    if EscapeHatch.START_MARKER in result.code:
      print("\n=== GENERATED CODE WITH ERRORS ===")
      print(result.code)

    assert result.success, f"Conversion failed: {result.errors}"
    code = result.code

    # Assert Shim Class is Injected (Preamble)
    assert "class GenericDataLoader" in code

    # Assert Call is Rewritten
    assert "dl = GenericDataLoader(ds" in code

    # Assert Arguments preserved - robustly check regardless of spacing
    assert "batch_size" in code
    assert "32" in code

    # Assert No Failure Markers
    assert EscapeHatch.START_MARKER not in code

"""
Tests for FrameworkSyncer Import and Linking Flow.

Verifies that:
1. The Syncer correctly utilizes the expanded SEARCH_PATHS (TensorFlow, MLX).
2. Modules are imported via importlib.
3. Matching functions are linked into the semantics dictionary.
4. Frameworks that fail to import don't crash the syncer.
"""

import types
from unittest.mock import patch
import pytest

from ml_switcheroo.discovery.syncer import FrameworkSyncer, SEARCH_PATHS


def mock_module(name: str, functions: dict) -> types.ModuleType:
  """Helper to create a dummy module with specific callable attributes."""
  mod = types.ModuleType(name)
  for func_name, func_obj in functions.items():
    setattr(mod, func_name, func_obj)
  return mod


@pytest.fixture
def syncer():
  """Returns a fresh instance of FrameworkSyncer."""
  return FrameworkSyncer()


def test_search_paths_completeness():
  """
  Verify `SEARCH_PATHS` contains entries for new engines (TF, MLX).
  """
  assert "tensorflow" in SEARCH_PATHS
  assert "tensorflow.math" in SEARCH_PATHS["tensorflow"]
  assert "mlx" in SEARCH_PATHS
  assert "mlx.core" in SEARCH_PATHS["mlx"]


def test_tensorflow_linking_flow(syncer):
  """
  Scenario: User syncs 'tensorflow'.
  Action: Syncer should check `tensorflow` and `tensorflow.math`.
  Result: `tf.math.abs` should be linked to `abs`.
  """
  # 1. Setup Data
  semantics = {
    "abs": {"std_args": ["x"], "variants": {}},
    "unknown_op": {"std_args": ["x"], "variants": {}},
  }

  # 2. Mock TensorFlow Modules
  # Define a compatible function
  def tf_abs(_x):
    pass

  # Create Mocks
  mock_tf_mod = mock_module("tensorflow", {})
  mock_tf_math = mock_module("tensorflow.math", {"abs": tf_abs})

  # 3. Patch importlib to return our mocks
  with patch("importlib.import_module") as mock_import:

    def side_effect(name):
      if name == "tensorflow":
        return mock_tf_mod
      if name == "tensorflow.math":
        return mock_tf_math
      raise ImportError(f"No module named {name}")

    mock_import.side_effect = side_effect

    # 4. Run Sync
    syncer.sync(semantics, "tensorflow")

  # 5. Verify Results
  # 'abs' should be found in tensorflow.math
  assert "tensorflow" in semantics["abs"]["variants"]
  # We expect the module name from the mocked module object
  assert semantics["abs"]["variants"]["tensorflow"]["api"] == "tensorflow.math.abs"

  # 'unknown_op' should remain unmapped
  assert "tensorflow" not in semantics["unknown_op"]["variants"]


def test_mlx_linking_flow(syncer):
  """
  Scenario: User syncs 'mlx'.
  Action: Syncer should check `mlx.core`.
  Result: `mlx.core.add` should match match standard `add`.
  """
  semantics = {"add": {"std_args": ["x", "y"], "variants": {}}}

  def mlx_add(_a, _b):  # Signature matches (2 args)
    pass

  mock_mlx_core = mock_module("mlx.core", {"add": mlx_add})

  with patch("importlib.import_module") as mock_import:

    def side_effect(name):
      # FrameworkSyncer looks for mlx.core, mlx.nn, etc.
      if name == "mlx.core":
        return mock_mlx_core
      # We can allow others to fail import
      raise ImportError(f"No module named {name}")

    mock_import.side_effect = side_effect

    syncer.sync(semantics, "mlx")

  assert "mlx" in semantics["add"]["variants"]
  assert semantics["add"]["variants"]["mlx"]["api"] == "mlx.core.add"


def test_sync_skips_incompatible_signatures(syncer):
  """
  Scenario: Framework has a function with match name but wrong signature.
  Result: It is NOT linked.
  """
  semantics = {"matmul": {"std_args": ["x", "y"], "variants": {}}}

  # Incompatible: matmul taking only 1 arg vs standard 2
  def bad_matmul(_x):
    pass

  mock_mod = mock_module("my_lib", {"matmul": bad_matmul})

  # Hook the SEARCH_PATHS temporarily to test generic logic
  with patch.dict(SEARCH_PATHS, {"my_lib": ["my_lib"]}):
    with patch("importlib.import_module", return_value=mock_mod):
      syncer.sync(semantics, "my_lib")

  # Should NOT have linked
  assert "my_lib" not in semantics["matmul"]["variants"]


def test_fails_gracefully_on_import_error(syncer, capsys):
  """
  Scenario: User asks to sync a framework that is not installed.
  Result: Error message logged, no crash.
  """
  semantics = {"abs": {"std_args": ["x"], "variants": {}}}

  with patch("importlib.import_module", side_effect=ImportError("Not installed")):
    syncer.sync(semantics, "ghost_framework")

  # Check that console printed warning
  # Note: Rich output capture is tricky with capsys unless console is injected.
  # FrameworkSyncer uses the global console.
  # We rely on the fact it didn't raise exception.
  assert "ghost_framework" not in semantics["abs"]["variants"]


def test_sync_preserves_existing_mappings(syncer):
  """
  Scenario: 'variants' already has an entry for this framework (manual override).
  Result: The existing entry is preserved, scan is skipped for that op.
  """
  semantics = {"abs": {"std_args": ["x"], "variants": {"torch": {"api": "manual.override.abs"}}}}

  # Even if we provide a "real" torch.abs in the mock
  def real_abs(_x):
    pass

  mock_torch = mock_module("torch", {"abs": real_abs})

  with patch("importlib.import_module", return_value=mock_torch):
    syncer.sync(semantics, "torch")

  # Should remain the manual override
  assert semantics["abs"]["variants"]["torch"]["api"] == "manual.override.abs"

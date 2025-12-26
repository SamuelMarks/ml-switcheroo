"""
Tests for Framework Flavour Configuration.

Verifies:
1. RuntimeConfig flavour resolution logic.
2. ASTEngine initialization with effective targets.
3. Rewriter behavior change based on flavour context.
"""

import pytest
from unittest.mock import MagicMock, patch
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.rewriter import PivotRewriter


def test_effective_framework_resolution():
  """Verify prioritization of Flavour over Framework settings with fallback."""
  # Case 1: Simple
  c1 = RuntimeConfig(source_framework="torch", target_framework="jax")
  assert c1.effective_source == "torch"
  assert c1.effective_target == "jax"

  # Case 2: Target Flavour Overrides
  c2 = RuntimeConfig(target_framework="jax", target_flavour="flax_nnx")
  assert c2.effective_target == "flax_nnx"
  # Source stays default (dynamically resolved)
  # We don't assert source is 'torch' here because it depends on registry state
  # Instead we check that effective_source matches source_framework
  assert c2.effective_source == c2.source_framework

  # Case 3: Source Flavour Overrides
  c3 = RuntimeConfig(source_framework="jax", source_flavour="paxml", target_framework="torch")
  assert c3.effective_source == "paxml"


def test_engine_adopts_flavour():
  """Verify ASTEngine creates internal state using effective keys."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", target_flavour="paxml")

  engine = ASTEngine(config=config)

  # Engine.target should match flavour
  assert engine.target == "paxml"
  assert engine.source == "torch"


def test_rewriter_integration_mock():
  """
  Verify PivotRewriter (via Engine) requests the specific Flavour config
  from the SemanticsManager.
  """
  # 1. Setup Mock Semantics
  mgr = MagicMock(spec=SemanticsManager)

  # Essential mocks to pass init sequences
  mgr.get_known_apis.return_value = {}
  mgr.get_all_rng_methods.return_value = set()
  mgr.get_definition.return_value = None

  # 2. Setup Flavour Config
  config = RuntimeConfig(source_framework="torch", target_framework="jax", target_flavour="flax_nnx")

  # 3. Initialize Engine (which in turn inits Rewriter in run())
  engine = ASTEngine(semantics=mgr, config=config)

  # 4. Trigger Run
  # NOTE: Must use input that triggers trait lookups (e.g. FunctionDef or ClassDef)
  # Simple assignment 'x=1' skips structure mixins that query target config.
  engine.run("def model_fn(): pass")

  # 5. Verify Semantics Logic
  # The Rewrite cycle should request config for 'flax_nnx' (the effective target)
  calls = [args[0] for args, _ in mgr.get_framework_config.call_args_list]

  assert "flax_nnx" in calls, "Engine should query semantics for the specific flavour"
  # 'torch' is queried by PurityScanner (source), 'flax_nnx' by Rewriter (target)

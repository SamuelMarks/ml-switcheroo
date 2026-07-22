"""Test suite for the Config Flavour module."""

from unittest.mock import MagicMock
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from tests.conftest import TestRewriter as PivotRewriter


def test_effective_framework_resolution():
  """Verifies the behavior of effective framework resolution."""
  c1 = RuntimeConfig(source_framework="torch", target_framework="jax")
  assert c1.effective_source == "torch"
  assert c1.effective_target == "jax"
  c2 = RuntimeConfig(target_framework="jax", target_flavour="flax_nnx")
  assert c2.effective_target == "flax_nnx"
  assert c2.effective_source == c2.source_framework
  c3 = RuntimeConfig(source_framework="jax", source_flavour="paxml", target_framework="torch")
  assert c3.effective_source == "paxml"


def test_engine_adopts_flavour():
  """Verifies the behavior of engine adopts flavour."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax", target_flavour="paxml")
  engine = ASTEngine(config=config)
  assert engine.target == "paxml"
  assert engine.source == "torch"


def test_rewriter_integration_mock():
  """Verifies the behavior of rewriter integration mock."""
  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_known_apis.return_value = {}
  mgr.get_all_rng_methods.return_value = set()
  mgr.get_definition.return_value = None
  config = RuntimeConfig(source_framework="torch", target_framework="jax", target_flavour="flax_nnx")
  rewriter = PivotRewriter(semantics=mgr, config=config)
  rewriter.convert(MagicMock())
  assert rewriter.context.target_fw == "flax_nnx"

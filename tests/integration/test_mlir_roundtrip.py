"""Test suite for the Mlir Roundtrip module."""

import pytest
from unittest.mock import patch
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.tracer import TraceEventType
from ml_switcheroo_ir.schema.ghost import SemanticTier

SOURCE_CODE = "\nclass MyModel:\n    def forward(self, x):\n        return x\n"
CONVNET_SOURCE = "\nimport torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n"


class MockFlaxSemantics(SemanticsManager):
  """Mock Flax Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockFlaxSemantics instance."""
    self.data = {}
    self.import_data = {}
    self.framework_configs = {}
    self._reverse_index = {}
    self.test_templates = {}
    self._key_origins = {}
    self._validation_status = {}
    self._known_rng_methods = set()
    self._providers = {}
    self._source_registry = {}
    self.framework_configs["torch"] = {
      "traits": {"module_base": "torch.nn.Module", "forward_method": "forward", "requires_super_init": True}
    }
    self.framework_configs["flax_nnx"] = {
      "traits": {
        "module_base": "flax.nnx.Module",
        "forward_method": "__call__",
        "inject_magic_args": [("rngs", "flax.nnx.Rngs")],
        "requires_super_init": False,
      },
      "alias": {"module": "flax.nnx", "name": "nnx"},
    }
    self._add_op("Conv2d", ["in", "out", "k"], "torch.nn.Conv2d", "flax.nnx.Conv")
    self._add_op("Linear", ["in", "out"], "torch.nn.Linear", "flax.nnx.Linear")
    self._add_op("Flatten", ["x", "dim"], "torch.flatten", "flax.nnx.Flatten")
    self._add_op("Module", [], "torch.nn.Module", "flax.nnx.Module")
    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
    self._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}

  def get_all_rng_methods(self):
    """Mock implementation of get all rng methods."""
    return set()

  def get_import_map(self, target_fw):
    """Mock implementation of get import map."""
    if target_fw == "flax_nnx":
      return {"torch.nn": ("flax", "nnx", "nnx")}
    return {}

  def get_framework_config(self, framework):
    """Mock implementation of get framework configuration."""
    return self.framework_configs.get(framework, {})

  def _add_op(self, name, args, s_api, t_api):
    """Mock implementation of  add op."""
    variants = {"torch": {"api": s_api}, "flax_nnx": {"api": t_api}}
    self.data[name] = {"std_args": args, "variants": variants}
    self._reverse_index[s_api] = (name, self.data[name])
    self._key_origins[name] = SemanticTier.NEURAL.value


@pytest.fixture
def engine_mlir():
  """Provides a mock engine MLIR for testing."""
  config = RuntimeConfig(source_framework="mlir", target_framework="jax", strict_mode=False)
  with patch("ml_switcheroo.semantics.manager.SemanticsManager") as mock_mgr:
    mgr = mock_mgr.return_value
    mgr.get_framework_config.return_value = {}
    return ASTEngine(semantics=mgr, config=config, intermediate="mlir")


def test_mlir_bridge_activation(engine_mlir):
  """Verifies the behavior of MLIR bridge activation."""
  result = engine_mlir.run('%0 = "sw.noop"()')
  assert result.success
  phase_names = [e["description"] for e in result.trace_events if e["type"] == TraceEventType.PHASE_START]
  assert "MLIR Ingest" in phase_names or "MLIR Bridge" in phase_names


def test_convnet_full_flow_mlir():
  """Verifies the behavior of convnet full flow MLIR."""
  semantics = MockFlaxSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config, intermediate="mlir")
  result = engine.run(CONVNET_SOURCE)
  assert result.success, f"Errors: {result.errors}"
  code = result.code
  assert "class ConvNet(nnx.Module):" in code
  assert "def __init__(self, rngs: nnx.Rngs):" in code
  assert "nnx.Conv" in code
  assert "nnx.Linear" in code
  assert "rngs=rngs" in code
  assert "def __call__(" in code
  assert "nnx.Flatten" in code
  assert "flax.nnx" in code

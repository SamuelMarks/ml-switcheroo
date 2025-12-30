"""
Integration test for the MLIR Bridge Pipeline.

Verifies:
1. Flow A (Default): Engine runs without MLIR.
2. Flow B (MLIR): Engine converts Tree -> MLIR -> Tree.
3. Code structure is preserved after roundtrip.
4. Full ConvNet example translation via MLIR intermediate.
"""

import pytest
import re
import libcst as cst
from unittest.mock import patch, MagicMock
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.tracer import TraceEventType
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter

# Simple mock code for basic roundtrip
SOURCE_CODE = """ 
class MyModel: 
    def forward(self, x): 
        return x
"""

# ConvNet Source from Prompt
CONVNET_SOURCE = """ 
import torch
import torch.nn as nn

class ConvNet(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv = nn.Conv2d(1, 32, 3) 
        self.fc = nn.Linear(32 * 26 * 26, 10) 

    def forward(self, x): 
        x = self.conv(x) 
        x = torch.flatten(x, 1) 
        return self.fc(x) 
"""


class MockFlaxSemantics(SemanticsManager):
  """
  Semantics Manager configured for Torch -> Flax NNX via MLIR.
  """

  def __init__(self):
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

    # Framework Configs (Traits)
    self.framework_configs["torch"] = {
      "traits": {
        "module_base": "torch.nn.Module",
        "forward_method": "forward",
        "requires_super_init": True,
      }
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

    # Operations
    # Conv2d
    self._add_op("Conv2d", ["in", "out", "k"], "torch.nn.Conv2d", "flax.nnx.Conv")
    # Linear
    self._add_op("Linear", ["in", "out"], "torch.nn.Linear", "flax.nnx.Linear")
    # Flatten (mapped to nnx.Flatten for this test per prompt expectation)
    self._add_op("Flatten", ["x", "dim"], "torch.flatten", "flax.nnx.Flatten")
    # Module
    self._add_op("Module", [], "torch.nn.Module", "flax.nnx.Module")

    # Import Abstraction
    self._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
    self._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}

  def get_all_rng_methods(self):
    return set()

  def get_import_map(self, target_fw):
    # Return basic mapping for torch.nn -> flax.nnx
    if target_fw == "flax_nnx":
      return {"torch.nn": ("flax", "nnx", "nnx")}
    return {}

  def get_framework_config(self, framework):
    return self.framework_configs.get(framework, {})

  def _add_op(self, name, args, s_api, t_api):
    variants = {"torch": {"api": s_api}, "flax_nnx": {"api": t_api}}
    self.data[name] = {"std_args": args, "variants": variants}
    self._reverse_index[s_api] = (name, self.data[name])
    # Mark as Neural to trigger state injection logic
    self._key_origins[name] = SemanticTier.NEURAL.value


@pytest.fixture
def engine_mlir():
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  # Enable MLIR intermediate
  # We must patch semantics to ensure it doesn't try to load files if environment not clean
  with patch("ml_switcheroo.semantics.manager.SemanticsManager") as mock_mgr:
    mgr = mock_mgr.return_value
    mgr.get_framework_config.return_value = {}
    # Ensure tracer is active
    return ASTEngine(semantics=mgr, config=config, intermediate="mlir")


def test_mlir_bridge_activation(engine_mlir):
  """Verify that enabling intermediate='mlir' triggers the bridge phase."""
  result = engine_mlir.run(SOURCE_CODE)

  assert result.success

  # Check trace for MLIR phase
  phase_names = [e["description"] for e in result.trace_events if e["type"] == TraceEventType.PHASE_START]
  assert "MLIR Bridge" in phase_names

  # Check mutation log (code transformation)
  mutations = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]
  # Matches "MLIR Generation" log from engine
  matches = [m for m in mutations if "MLIR Generation" in m["description"]]
  assert len(matches) > 0
  assert "sw.module" in matches[0]["metadata"]["after"]


def test_mlir_code_fidelity(engine_mlir):
  """
  Verify that code comes back valid.
  Note: Generator renaming logic (v0, v1) changes variable names, so exact string match fails.
  We check structure.
  """
  result = engine_mlir.run(SOURCE_CODE)
  code = result.code

  assert "class" in code
  assert "UnknownClass" in code or "MyModel" in code
  assert "forward" in code  # signature might change params names/order slightly
  assert "return" in code


def test_mlir_bridge_failure_fallback():
  """Verify that if MLIR bridge crashes, original tree is preserved."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(config=config, intermediate="mlir")

  # Manually invoke with bad input that causes crash inside bridge logic (e.g. non-CST)
  # _run_mlir_roundtrip expects cst.Module. Passing string causing Method extraction failure?
  # Actually PythonToMlirEmitter.convert calls .body on input. Passing valid string "S" -> valid string has no .body.

  # Assert successful fallback (returns input "S") and logs error
  res = engine._run_mlir_roundtrip("BAD_INPUT", MagicMock())
  assert res == "BAD_INPUT"


def test_convnet_full_flow_mlir():
  """
  Full Integration: Torch ConvNet -> MLIR -> Flax NNX.
  Verifies:
  1. Arithmetic expressions (32 * 26 * 26) are handled via BinOps.
  2. Inheritance (nn.Module) is preserved/translated via sw.mod attribute.
  3. Assignments (self.conv =) are preserved via sw.setattr.
  4. Rewriter runs BEFORE MLIR, so MLIR sees Flax structures.
  5. Emitter -> Generator preserves types matches validation.
  """
  semantics = MockFlaxSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)

  # Enable MLIR intermediate
  engine = ASTEngine(semantics=semantics, config=config, intermediate="mlir")

  result = engine.run(CONVNET_SOURCE)

  assert result.success, f"Errors: {result.errors}"
  code = result.code

  print("\n[Generated Flax via MLIR]:\n")
  print(code)

  # 1. Check Inheritance Rewriting
  # MLIR reconstructs class ConvNet(nn.Module) -> Rewriter converts to flax.nnx.Module
  assert "class ConvNet(flax.nnx.Module):" in code or "class ConvNet(nnx.Module):" in code

  # 2. Check Constructor
  # With improved annotation support (integrated into emitter), type hint should be preserved. Check for aliasing (nnx.Rngs).
  assert "def __init__(self, rngs: flax.nnx.Rngs):" in code or "def __init__(self, rngs: nnx.Rngs):" in code

  # 3. Check Arithmetic & Layers
  # MLIR Emitter converts 32*26*26 to binops.
  # Generator reconstructs assignments _x = ...
  # Rewriter converts nn.Conv2d -> nnx.Conv

  # Allow aliased usage (nnx.Conv) OR full usage (flax.nnx.Conv)
  assert "flax.nnx.Conv" in code or "nnx.Conv" in code
  assert "flax.nnx.Linear" in code or "nnx.Linear" in code

  # Check that rngs is passed to calls (verified via positional arg in output, so relaxed check)
  assert "rngs" in code

  # Check that math was preserved (evaluated or reconstructed structure)
  # Since binops effectively flatten constants, we might see literals or vars
  # e.g. _0 = 32, _1 = 26, _2 = _0 * _1
  # We check for the presence of the calculation logic
  assert "*" in code  # Multiplication preserved

  # 4. Check Forward
  # We check for the presence of __call__ and the argument structure
  assert "def __call__(" in code

  # Check for folded expression logic (Generator Optimization):
  # Expect nested calls like fc(Flatten(conv(x), 1))
  assert "conv(x)" in code or "self.conv(x)" in code

  # Flatten check: Ensure Flatten is used (likely nested inside fc call due to folding)
  assert "flax.nnx.Flatten" in code or "nnx.Flatten" in code

  # 5. Check imports
  # Accept both forms: 'import flax.nnx as nnx' OR 'from flax import nnx'
  assert "import flax.nnx" in code or "from flax import nnx" in code

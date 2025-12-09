"""End-to-End Integration Tests for PaxML (Praxis) Conversion.

These tests focus on the structural rewrite from PyTorch to PaxML/Praxis,
verifying 3-Tier Layer setup, `setup` method renaming, and functional forward pass.
"""

import pytest
from pathlib import Path
import libcst as cst
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _read_code(filename: str) -> str:
  path = EXAMPLES_DIR / filename
  return path.read_text(encoding="utf-8")


class PaxE2ESemantics(SemanticsManager):
  """
  Mock Manager with PaxML structural traits.
  """

  def __init__(self):
    self.data = {}
    self.import_data = {}
    self._reverse_index = {}
    self._key_origins = {}

    # Configure Traits for PaxML
    self.framework_configs = {
      "paxml": {
        "traits": {
          "module_base": "praxis.base_layer.BaseLayer",
          "forward_method": "__call__",
          "init_method_name": "setup",
          "requires_super_init": False,
        }
      },
      # FIX: Define Source Framework Traits so torch.nn.Module is recognized
      "torch": {
        "traits": {
          "module_base": "torch.nn.Module",
          "forward_method": "forward",
          "strip_magic_args": ["rngs"],
          "requires_super_init": True,
        }
      },
    }

    # --- Neural Network Maps ---
    # Remap Module
    self._add_op("Module", [], torch="torch.nn.Module", pax="praxis.base_layer.BaseLayer", tier=SemanticTier.NEURAL)

    # Remap Linear
    self._add_op(
      "Linear",
      ["in_features", "out_features"],
      torch="torch.nn.Linear",
      pax="praxis.layers.Linear",
      tier=SemanticTier.NEURAL,
    )

    # Remap ReLU
    self._add_op("ReLU", [], torch="torch.nn.ReLU", pax="praxis.layers.ReLU", tier=SemanticTier.NEURAL)

    # Imports
    self.import_data["torch.nn"] = {"variants": {"paxml": {"root": "praxis", "sub": "layers", "alias": "nn"}}}

    # Abstract mapping for alias 'nn.Module'
    self._alias("nn.Module", "Module")
    self._alias("nn.Linear", "Linear")
    self._alias("nn.ReLU", "ReLU")

  def get_framework_config(self, framework: str):
    return self.framework_configs.get(framework, {})

  def _add_op(self, name, args, torch, pax, tier=None):
    self.data[name] = {
      "std_args": args,
      "variants": {"torch": {"api": torch}, "paxml": {"api": pax}},
    }
    if torch:
      self._reverse_index[torch] = (name, self.data[name])
    if pax:
      self._reverse_index[pax] = (name, self.data[name])

    if tier:
      self._key_origins[name] = tier.value
    else:
      self._key_origins[name] = SemanticTier.ARRAY_API.value

  def _alias(self, api_str, abstract_name):
    if abstract_name in self.data:
      self._reverse_index[api_str] = (abstract_name, self.data[abstract_name])


@pytest.fixture
def pax_engine():
  semantics = PaxE2ESemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="paxml", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


def test_ex06_paxml_full_conversion(pax_engine):
  """
  Runs the E2E conversion of ex06_paxml.torch.py.
  """
  code = _read_code("ex06_paxml.torch.py")
  result = pax_engine.run(code)

  assert result.success, f"Conversion failed: {result.errors}"
  generated = result.code

  # 1. Imports
  assert "import praxis" in generated or "from praxis" in generated

  # 2. Class Structure (Inheritance Rewrite)
  assert "class SimpleMLP(praxis.base_layer.BaseLayer):" in generated

  # 3. Setup Method (Renamed from __init__ via Plugin/Traits)
  # This requires _in_module_class=True during processing
  assert "def setup(self, input_size, hidden_size, num_classes):" in generated
  assert "def __init__" not in generated

  # 4. Super Strip
  # Requires FuncStructureMixin to trigger off _in_module_class=True
  assert "super().__init__()" not in generated

  # 5. Forward Renaming
  # With traits loaded correctly, forward should become __call__
  assert "def __call__(self, x):" in generated
  assert "def forward" not in generated

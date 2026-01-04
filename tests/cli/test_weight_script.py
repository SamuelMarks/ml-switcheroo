"""
Tests for the Weight Migration Script Generator.

Verifies:
1.  AST Layer Extraction via GraphExtractor.
2.  Mapping Rule Construction (Argument renaming, permuting).
3.  Script Generation (Imports, Load/Save logic via Adapters).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

# Mock Source Code
MODEL_CODE = """
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.fc = nn.Linear(128, 10)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        return self.fc(self.conv(x))
"""


@pytest.fixture
def mock_semantics() -> MagicMock:
  """
  Mocks the SemanticsManager to return definition dictionaries for standard ops.

  returns:
      MagicMock: Configured semantics mock.
  """
  mgr = MagicMock(spec=SemanticsManager)

  # 1. Conv2d: Maps weight->kernel and permutations
  conv_def = (
    "Conv2d",
    {
      "variants": {
        "torch": {"api": "torch.nn.Conv2d", "args": {"weight": "weight"}},
        "jax": {
          "api": "flax.nnx.Conv",
          "args": {"weight": "kernel"},
          "layout_map": {"weight": "OIHW->HWIO"},
        },
      }
    },
  )

  # 2. Linear: Maps weight->kernel, no perm
  linear_def = (
    "Linear",
    {
      "variants": {
        "torch": {"api": "torch.nn.Linear"},
        "jax": {"api": "flax.nnx.Linear", "args": {"weight": "kernel"}},
      }
    },
  )

  # 3. BatchNorm: Maps attributes
  bn_def = (
    "BatchNorm",
    {
      "variants": {
        "torch": {"api": "torch.nn.BatchNorm"},
        "jax": {
          "api": "flax.nnx.BatchNorm",
          "args": {"weight": "scale", "bias": "bias"},
        },
      }
    },
  )

  # Hook up lookup
  def get_def(name: str):
    if "Conv2d" in name:
      return conv_def
    if "Linear" in name:
      return linear_def
    if "BatchNorm" in name:
      return bn_def
    return None

  mgr.get_definition.side_effect = get_def
  return mgr


@pytest.fixture
def fwd_generator(mock_semantics: MagicMock) -> WeightScriptGenerator:
  """
  Creates a WeightScriptGenerator configured for Torch -> JAX (Flax).

  Args:
      mock_semantics: The mocked knowledge base.

  Returns:
      WeightScriptGenerator: Configured instance.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  # Patch get_adapter within the generator init scope or globally
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get:
    # Return actual adapter instances (they safe-init to Ghost mode if libs missing)
    mock_get.side_effect = lambda n: TorchAdapter() if n == "torch" else FlaxNNXAdapter()
    return WeightScriptGenerator(mock_semantics, config)


def test_ast_layer_extraction(fwd_generator: WeightScriptGenerator, tmp_path):
  """
  Verify GraphExtractor identifies layers 'conv', 'fc', 'bn' from AST.
  """
  f = tmp_path / "model.py"
  f.write_text(MODEL_CODE, encoding="utf-8")

  out = tmp_path / "script.py"
  success = fwd_generator.generate(f, out)

  assert success
  content = out.read_text(encoding="utf-8")

  # Check for presence of key layer names in the generated rules dictionary repr
  assert "'layer': 'conv'" in content
  assert "'layer': 'fc'" in content
  assert "'layer': 'bn'" in content


def test_file_read_error(fwd_generator: WeightScriptGenerator, tmp_path):
  """Verify handling of missing source files."""
  missing = tmp_path / "missing.py"
  out = tmp_path / "script.py"
  assert not fwd_generator.generate(missing, out)


def test_parse_error(fwd_generator: WeightScriptGenerator, tmp_path):
  """Verify handling of syntax errors in source."""
  f = tmp_path / "bad.py"
  f.write_text("class Broken( Syntax { Error ", encoding="utf-8")
  out = tmp_path / "script.py"
  assert not fwd_generator.generate(f, out)


def test_missing_adapters(mock_semantics):
  """Verify handling when adapters fail to load."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value=None):
    gen = WeightScriptGenerator(mock_semantics, config)
    assert not gen.generate(Path("x.py"), Path("y.py"))


def test_mapping_rule_logic_fwd(fwd_generator: WeightScriptGenerator):
  """
  Verify Forward (Torch -> JAX) rules dictionary construction.
  Expects permutation OIHW -> HWIO (2, 3, 1, 0).
  """
  from ml_switcheroo.core.graph import LogicalNode

  # Manually constructed registry simulating Extraction
  registry = {
    "conv1": LogicalNode("conv1", "Conv2d", {}),
    "fc1": LogicalNode("fc1", "Linear", {}),
  }

  rules = fwd_generator._flatten_mapping_rules(registry)

  # Locate Conv rule for Weight
  conv_weight = next(r for r in rules if r["layer"] == "conv1" and "weight" in r["src_key"])

  # Check renaming: src 'weight' -> tgt 'kernel'
  assert conv_weight["src_key"] == "conv1.weight"
  assert conv_weight["src_suffix"] == "weight"
  assert conv_weight["tgt_suffix"] == "kernel"

  # Check permutation (OIHW->HWIO -> (2,3,1,0))
  # Source: OIHW (0, 1, 2, 3)
  # Target: HWIO (2, 3, 1, 0)
  assert conv_weight["perm"] == (2, 3, 1, 0)


def test_fwd_script_content(fwd_generator: WeightScriptGenerator, tmp_path):
  """
  Verify generated script content contains specific framework logic strings.
  """
  f = tmp_path / "model.py"
  f.write_text(MODEL_CODE, encoding="utf-8")
  out = tmp_path / "params.py"

  fwd_generator.generate(f, out)
  code = out.read_text(encoding="utf-8")

  # Check for imports defined in adapters
  assert "import orbax.checkpoint" in code
  assert "import torch" in code

  # Check logic calls
  assert "MAPPING_RULES =" in code
  assert "permute(arr, p)" in code
  assert "migrate(input_path, output_path)" in code

  # Check for Load logic (Torch)
  assert "torch.load" in code
  # The string below comes from TorchAdapter.get_weight_load_code
  assert "map_location='cpu'" in code

  # Check for Save Logic (JAX/Orbax)
  # Corrected assertion to match JAXStackMixin.get_weight_save_code output
  assert "orbax.checkpoint.PyTreeCheckpointer()" in code
  assert "checkpointer.save(" in code
  assert "unflatten_dict" in code

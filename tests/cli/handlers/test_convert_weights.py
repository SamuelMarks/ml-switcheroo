"""Test suite for the Convert Weights module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.graph import LogicalNode


@pytest.fixture
def mock_config():
  """Provides a mock configuration for testing."""
  config = MagicMock(spec=RuntimeConfig)
  config.effective_source = "torch"
  config.effective_target = "jax"
  return config


@pytest.fixture
def mock_semantics():
  """Provides a mock semantics for testing."""
  semantics = MagicMock(spec=SemanticsManager)
  semantics.get_definition.return_value = (
    "Linear",
    {
      "variants": {
        "torch": {"args": {"weight": "weight", "bias": "bias"}},
        "jax": {"args": {"weight": "kernel", "bias": "bias"}, "layout_map": {"weight": "OI->IO"}},
      }
    },
  )
  return semantics


def test_generate_success(mock_config, mock_semantics, tmp_path):
  """Generates successfully."""
  source_file = tmp_path / "model.py"
  source_file.write_text("import torch.nn as nn\nclass Model:\n  def __init__(self):\n    self.l1 = nn.Linear(10, 10)\n")
  out_file = tmp_path / "script.py"
  generator = WeightScriptGenerator(mock_semantics, mock_config)
  generator.source_adapter = MagicMock()
  generator.source_adapter.get_weight_conversion_imports.return_value = ["import torch"]
  generator.source_adapter.get_weight_load_code.return_value = "raw_state = torch.load(input_path)"
  generator.source_adapter.get_tensor_to_numpy_expr.return_value = "raw_val.numpy()"
  generator.target_adapter = MagicMock()
  generator.target_adapter.get_weight_conversion_imports.return_value = ["import jax"]
  generator.target_adapter.get_weight_save_code.return_value = "pass"
  assert generator.generate(source_file, out_file) is True
  assert out_file.exists()
  script = out_file.read_text()
  assert "MAPPING_RULES = " in script
  assert "'perm': (1, 0)" in script


def test_generate_no_adapters(mock_config, mock_semantics, tmp_path):
  """Generates no adapters."""
  generator = WeightScriptGenerator(mock_semantics, mock_config)
  generator.source_adapter = None
  assert generator.generate(Path("src.py"), Path("out.py")) is False


def test_generate_read_error(mock_config, mock_semantics, tmp_path):
  """Generates read correctly handling an error."""
  generator = WeightScriptGenerator(mock_semantics, mock_config)
  generator.source_adapter = MagicMock()
  generator.target_adapter = MagicMock()
  assert generator.generate(tmp_path / "nonexistent.py", Path("out.py")) is False


def test_generate_parse_error(mock_config, mock_semantics, tmp_path):
  """Generates parse correctly handling an error."""
  source_file = tmp_path / "model.py"
  source_file.write_text("invalid python code [")
  out_file = tmp_path / "script.py"
  generator = WeightScriptGenerator(mock_semantics, mock_config)
  generator.source_adapter = MagicMock()
  generator.target_adapter = MagicMock()
  assert generator.generate(source_file, out_file) is False


def test_generate_no_layers(mock_config, mock_semantics, tmp_path):
  """Generates no layers."""
  source_file = tmp_path / "model.py"
  source_file.write_text("x = 1")
  out_file = tmp_path / "script.py"
  generator = WeightScriptGenerator(mock_semantics, mock_config)
  generator.source_adapter = MagicMock()
  generator.target_adapter = MagicMock()
  with patch("ml_switcheroo.cli.handlers.convert_weights.GraphExtractor") as mock_extractor_class:
    mock_instance = MagicMock()
    mock_instance.layer_registry = {}
    mock_extractor_class.return_value = mock_instance
    assert generator.generate(source_file, out_file) is False


def test_generate_write_error(mock_config, mock_semantics, tmp_path):
  """Generates write correctly handling an error."""
  source_file = tmp_path / "model.py"
  source_file.write_text("import torch.nn as nn\nclass Model:\n  def __init__(self):\n    self.l1 = nn.Linear(10, 10)\n")
  out_file = tmp_path / "read_only_dir" / "script.py"
  generator = WeightScriptGenerator(mock_semantics, mock_config)
  generator.source_adapter = MagicMock()
  generator.target_adapter = MagicMock()
  with patch("pathlib.Path.write_text", side_effect=PermissionError("Permission denied")):
    assert generator.generate(source_file, out_file) is False


def test_flatten_mapping_rules_variations(mock_config, mock_semantics):
  """Verifies the behavior of flatten mapping rules variations."""
  generator = WeightScriptGenerator(mock_semantics, mock_config)
  layer_registry = {"l1": LogicalNode(id="l1", kind="Linear")}

  # When source = torch, it computes perm from OI->IO directly
  mock_config.effective_source = "torch"
  rules = generator._flatten_mapping_rules(layer_registry)
  assert len(rules) == 5
  assert rules[0]["perm"] == (1, 0)

  # When source = jax, it computes perm reversed
  mock_config.effective_source = "jax"
  generator = WeightScriptGenerator(mock_semantics, mock_config)
  rules = generator._flatten_mapping_rules(layer_registry)
  assert len(rules) == 5
  assert rules[0]["perm"] == (1, 0)  # IO->OI is also (1, 0)

  mock_semantics.get_definition.side_effect = [None, None]
  rules = generator._flatten_mapping_rules(layer_registry)
  assert len(rules) == 0

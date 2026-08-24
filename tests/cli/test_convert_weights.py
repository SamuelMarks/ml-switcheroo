"""Test module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.graph import LogicalNode


@pytest.fixture
def mock_semantics():
  """Test element."""
  sem = SemanticsManager()
  sem.get_definition = MagicMock(
    return_value=(
      "Conv2d",
      {
        "variants": {
          "torch": {"args": {"weight": "weight"}},
          "jax": {"args": {"weight": "kernel"}, "layout_map": {"weight": "OIHW->HWIO"}},
        }
      },
    )
  )
  return sem


@pytest.fixture
def mock_config():
  """Test element."""
  conf = MagicMock(spec=RuntimeConfig)
  conf.effective_source = "torch"
  conf.effective_target = "jax"
  return conf


def test_convert_weights_init(mock_semantics, mock_config):
  """Test element."""
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get_adapter:
    mock_get_adapter.side_effect = ["torch_adapter", "jax_adapter"]
    gen = WeightScriptGenerator(mock_semantics, mock_config)
    assert gen.source_fw == "torch"
    assert gen.target_fw == "jax"
    assert gen.source_adapter == "torch_adapter"
    assert gen.target_adapter == "jax_adapter"


def test_generate_missing_adapter(mock_semantics, mock_config):
  """Test element."""
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value=None):
    gen = WeightScriptGenerator(mock_semantics, mock_config)
    assert gen.generate(Path("in.py"), Path("out.py")) is False


def test_generate_file_error(mock_semantics, mock_config):
  """Test element."""
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value="adapter"):
    gen = WeightScriptGenerator(mock_semantics, mock_config)
    # Pass a non-existent file
    assert gen.generate(Path("does_not_exist.py"), Path("out.py")) is False


@patch("ml_switcheroo.cli.handlers.convert_weights.cst.parse_module")
def test_generate_ast_error(mock_parse, mock_semantics, mock_config, tmp_path):
  """Test element."""
  mock_parse.side_effect = Exception("Parse error")
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value="adapter"):
    gen = WeightScriptGenerator(mock_semantics, mock_config)

    in_file = tmp_path / "in.py"
    in_file.write_text("code")

    assert gen.generate(in_file, Path("out.py")) is False


@patch("ml_switcheroo.cli.handlers.convert_weights.cst.parse_module")
@patch("ml_switcheroo.cli.handlers.convert_weights.GraphExtractor")
def test_generate_no_layers(mock_extractor_class, mock_parse, mock_semantics, mock_config, tmp_path):
  """Test element."""
  mock_extractor = mock_extractor_class.return_value
  mock_extractor.layer_registry = {}

  mock_tree = MagicMock()
  mock_tree.visit.return_value = None
  mock_parse.return_value = mock_tree

  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value="adapter"):
    gen = WeightScriptGenerator(mock_semantics, mock_config)

    in_file = tmp_path / "in.py"
    in_file.write_text("code")

    assert gen.generate(in_file, Path("out.py")) is False


@patch("ml_switcheroo.cli.handlers.convert_weights.cst.parse_module")
@patch("ml_switcheroo.cli.handlers.convert_weights.GraphExtractor")
def test_generate_success(mock_extractor_class, mock_parse, mock_semantics, mock_config, tmp_path):
  """Test element."""
  mock_extractor = mock_extractor_class.return_value
  mock_node = LogicalNode(id="my_conv", kind="Conv2d")
  mock_extractor.layer_registry = {"my_conv": mock_node}

  # Mock parse_module to return a mock tree with a safe visit method
  mock_tree = MagicMock()
  mock_tree.visit.return_value = None
  mock_parse.return_value = mock_tree

  mock_src_adapter = MagicMock()
  mock_src_adapter.get_weight_conversion_imports.return_value = ["import torch"]
  mock_src_adapter.get_weight_load_code.return_value = "raw_state = {}"
  mock_src_adapter.get_tensor_to_numpy_expr.return_value = "val.numpy()"

  mock_tgt_adapter = MagicMock()
  mock_tgt_adapter.get_weight_conversion_imports.return_value = ["import jax"]
  mock_tgt_adapter.get_weight_save_code.return_value = "save(converted_state)"

  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get_adapter:
    mock_get_adapter.side_effect = [mock_src_adapter, mock_tgt_adapter]
    gen = WeightScriptGenerator(mock_semantics, mock_config)

    in_file = tmp_path / "in.py"
    in_file.write_text("class Model: pass")

    out_file = tmp_path / "out.py"

    assert gen.generate(in_file, out_file) is True
    assert out_file.exists()
    content = out_file.read_text()
    assert "import torch" in content
    assert "import jax" in content
    assert "MAPPING_RULES" in content
    assert "'src_key': 'my_conv.weight'" in content


def test_generate_write_error(mock_semantics, mock_config, tmp_path):
  """Test element."""
  # Mocking similar to success, but failing on write
  with (
    patch("ml_switcheroo.cli.handlers.convert_weights.cst.parse_module"),
    patch("ml_switcheroo.cli.handlers.convert_weights.GraphExtractor") as mock_extractor_class,
    patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter") as mock_get_adapter,
  ):
    mock_extractor = mock_extractor_class.return_value
    mock_extractor.layer_registry = {"my_conv": LogicalNode(id="my_conv", kind="Conv2d")}
    mock_get_adapter.return_value = MagicMock()

    gen = WeightScriptGenerator(mock_semantics, mock_config)

    in_file = tmp_path / "in.py"
    in_file.write_text("code")

    # Out file points to a directory to force permission/is_dir error
    out_dir = tmp_path / "out_dir"
    out_dir.mkdir()

    assert gen.generate(in_file, out_dir) is False


def test_flatten_mapping_rules_reverse(mock_semantics):
  """Test element."""
  # Test jax -> torch direction to hit the else branch for inverse permutation
  conf = MagicMock(spec=RuntimeConfig)
  conf.effective_source = "jax"
  conf.effective_target = "torch"

  # Redefine mock semantics so torch provides layout_map
  mock_semantics.get_definition.return_value = (
    "Conv2d",
    {
      "variants": {
        "jax": {"args": {"weight": "kernel"}},
        "torch": {"args": {"weight": "weight"}, "layout_map": {"weight": "HWIO->OIHW"}},
      }
    },
  )

  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter"):
    gen = WeightScriptGenerator(mock_semantics, conf)

    layer_registry = {"my_conv": LogicalNode(id="my_conv", kind="Conv2d")}
    rules = gen._flatten_mapping_rules(layer_registry)

    assert len(rules) > 0
    rule = next(r for r in rules if r["src_suffix"] == "kernel")  # source is JAX now, which defines "kernel" for "weight"
    assert rule["perm"] is not None  # Should compute OIHW -> HWIO inverse


def test_flatten_mapping_rules_no_def(mock_semantics, mock_config):
  """Test element."""
  mock_semantics.get_definition.return_value = None
  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter"):
    gen = WeightScriptGenerator(mock_semantics, mock_config)
    layer_registry = {"my_conv": LogicalNode(id="my_conv", kind="UnknownOp")}
    rules = gen._flatten_mapping_rules(layer_registry)
    assert len(rules) == 0


def test_generate_script_includes_safetensors(mock_semantics):
  """Verify that the generated script for torch/jax includes safetensors logic."""
  conf = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=False)
  gen = WeightScriptGenerator(mock_semantics, conf)

  rules = [
    {
      "layer": "my_conv",
      "src_suffix": "weight",
      "tgt_suffix": "kernel",
      "src_key": "my_conv.weight",
      "tgt_key": "my_conv.kernel",
      "perm": (3, 2, 0, 1),
    }
  ]
  script = gen._generate_script(rules)

  assert "safetensors.torch" in script
  assert "safetensors.flax" in script
  assert ".safetensors" in script

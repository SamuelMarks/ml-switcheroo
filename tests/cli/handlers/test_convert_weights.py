"""Unit tests for `WeightScriptGenerator` (`ml_switcheroo.cli.handlers.convert_weights`)."""

from pathlib import Path
from unittest.mock import patch

from pydantic import JsonValue

from ml_switcheroo.cli.handlers.convert_weights import WeightScriptGenerator
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.semantics.manager import SemanticsManager

_NL = chr(10)


def _build_weight_semantics() -> SemanticsManager:
  """Build an in-memory `SemanticsManager` with `Conv2d` layout rules for Torch and JAX.

  Returns:
      Configured `SemanticsManager` instance.
  """
  sem = SemanticsManager(load_defaults=False)
  conv_def: dict[str, JsonValue] = {
    "operation": "Conv2d",
    "std_args": ["in_channels", "out_channels", "kernel_size", "weight"],
    "variants": {
      "torch": {"api": "torch.nn.Conv2d", "args": {"weight": "weight"}, "layout_map": {"weight": "HWIO->OIHW"}},
      "jax": {"api": "flax.nnx.Conv", "args": {"weight": "kernel"}, "layout_map": {"weight": "OIHW->HWIO"}},
    },
  }
  no_arrow_def: dict[str, JsonValue] = {
    "operation": "Linear",
    "std_args": ["weight"],
    "variants": {
      "torch": {"api": "torch.nn.Linear"},
      "jax": {"api": "flax.nnx.Linear", "layout_map": {"weight": "FLATTEN"}},
    },
  }
  for op_id, raw in (("Conv2d", conv_def), ("Linear", no_arrow_def)):
    op_model = OperationDef.model_validate(raw)
    sem.data[op_id] = raw
    sem.operations[op_id] = op_model
    sem._reverse_index[op_id] = (op_id, raw)
    sem._typed_reverse_index[op_id] = (op_id, op_model)
  sem._build_index()
  return sem


def test_weight_script_generator_real_end_to_end_and_reverse_layouts(tmp_path: Path) -> None:
  """Verify real Torch->JAX and JAX->Torch weight migration script generation with safetensors & permutations.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  sem = _build_weight_semantics()
  model_py = tmp_path / "model.py"
  model_py.write_text(
    _NL.join(
      [
        "import torch.nn as nn",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        "        self.conv = nn.Conv2d(3, 16, 3)",
        "        self.fc = nn.Linear(16, 4)",
        "        self.custom = UnknownLayer()",
      ]
    ),
    encoding="utf-8",
  )

  # Forward: Torch -> JAX
  out_fwd = tmp_path / "torch_to_jax.py"
  gen_fwd = WeightScriptGenerator(sem, RuntimeConfig(source_framework="torch", target_framework="jax"))
  assert gen_fwd.generate(model_py, out_fwd) is True
  fwd_code = out_fwd.read_text(encoding="utf-8")
  assert "MAPPING_RULES" in fwd_code
  assert "safetensors.torch" in fwd_code
  assert "safetensors.flax" in fwd_code
  assert "'src_key': 'conv.weight'" in fwd_code
  assert "'tgt_key': 'conv.kernel'" in fwd_code

  # Inverse: JAX -> Torch (computes inverse layout permutation when target `torch` defines `layout_map`)
  sem_inv = SemanticsManager(load_defaults=False)
  inv_def: dict[str, JsonValue] = {
    "operation": "Conv2d",
    "std_args": ["weight"],
    "variants": {
      "jax": {"api": "Conv2d", "args": {"weight": "kernel"}},
      "torch": {"api": "Conv2d", "args": {"weight": "weight"}, "layout_map": {"weight": "HWIO->OIHW"}},
    },
  }
  sem_inv.data["Conv2d"] = inv_def
  sem_inv.operations["Conv2d"] = OperationDef.model_validate(inv_def)
  sem_inv._reverse_index["Conv2d"] = ("Conv2d", inv_def)
  sem_inv._typed_reverse_index["Conv2d"] = ("Conv2d", sem_inv.operations["Conv2d"])

  out_inv = tmp_path / "jax_to_torch.py"
  gen_inv = WeightScriptGenerator(sem_inv, RuntimeConfig(source_framework="jax", target_framework="torch"))
  assert gen_inv.generate(model_py, out_inv) is True
  inv_code = out_inv.read_text(encoding="utf-8")
  assert "'src_key': 'conv.kernel'" in inv_code


def test_weight_script_generator_error_and_edge_paths(tmp_path: Path) -> None:
  """Verify missing adapters, missing/invalid/layerless files, and output write `OSError` handling.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  sem = _build_weight_semantics()
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")

  # Missing adapter in `generate` and `_generate_script`
  import pytest

  from ml_switcheroo.exceptions import ConfigurationError

  with patch("ml_switcheroo.cli.handlers.convert_weights.get_adapter", return_value=None):
    missing_gen = WeightScriptGenerator(sem, cfg)
    assert missing_gen.generate(tmp_path / "in.py", tmp_path / "out.py") is False
    with pytest.raises(ConfigurationError):
      missing_gen._generate_script([])

  gen = WeightScriptGenerator(sem, cfg)

  # Missing source file
  assert gen.generate(tmp_path / "missing.py", tmp_path / "out.py") is False

  # Syntax error in source file
  bad_py = tmp_path / "bad.py"
  bad_py.write_text("invalid python {{", encoding="utf-8")
  assert gen.generate(bad_py, tmp_path / "out.py") is False

  # Valid Python file with zero stateful layers
  no_layers_py = tmp_path / "nolayers.py"
  no_layers_py.write_text("def f(): pass", encoding="utf-8")
  assert gen.generate(no_layers_py, tmp_path / "out.py") is False

  # Write failure when output path is an existing directory
  valid_py = tmp_path / "valid.py"
  valid_py.write_text(f"class M:{_NL}    def __init__(self):{_NL}        self.conv = Conv2d(){_NL}", encoding="utf-8")
  out_dir = tmp_path / "out_is_dir"
  out_dir.mkdir()
  assert gen.generate(valid_py, out_dir) is False

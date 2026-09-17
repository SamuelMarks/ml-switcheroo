"""End-to-end tests for CLI with Intermediate Representation (IR)."""

import json
from pathlib import Path
from typing import Any, Dict

from ml_switcheroo.cli.commands import handle_convert

SAMPLE_TORCH = """
import torch.nn as nn

class CliNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv(x)
"""

SAMPLE_IR = """{
  "name": "CliNet",
  "nodes": [
    {"id": "x", "kind": "Input"},
    {"id": "conv", "kind": "Conv2d", "inputs": ["x"]}
  ],
  "edges": [{"source": "x", "target": "conv"}]
}"""


def test_cli_convert_target_ir(tmp_path: Path) -> None:
  """Test CLI conversion from PyTorch to IR JSON."""
  in_file = tmp_path / "model.py"
  in_file.write_text(SAMPLE_TORCH, encoding="utf-8")
  out_file = tmp_path / "out_ir.json"

  exit_code = handle_convert(
    input_path=in_file,
    output_path=out_file,
    source="torch",
    target="ir",
    verify=False,
    strict=False,
    intermediate=None,
    plugin_settings={},
    json_trace_path=None,
    enable_sharding=False,
  )
  assert exit_code == 0
  assert out_file.exists()

  data: Dict[str, Any] = json.loads(out_file.read_text(encoding="utf-8"))
  assert data["name"] == "CliNet"
  assert len(data["nodes"]) >= 1


def test_cli_convert_source_ir(tmp_path: Path) -> None:
  """Test CLI conversion from IR JSON to PyTorch."""
  in_file = tmp_path / "model.json"
  in_file.write_text(SAMPLE_IR, encoding="utf-8")
  out_file = tmp_path / "out_torch.py"

  exit_code = handle_convert(
    input_path=in_file,
    output_path=out_file,
    source="ir",
    target="torch",
    verify=False,
    strict=False,
    intermediate=None,
    plugin_settings={},
    json_trace_path=None,
    enable_sharding=False,
  )
  assert exit_code == 0
  assert out_file.exists()

  code = out_file.read_text(encoding="utf-8")
  assert "class CliNet(nn.Module):" in code
  assert "self.conv = nn.Conv2d()" in code


def test_cli_convert_intermediate_ir(tmp_path: Path) -> None:
  """Test CLI conversion using --intermediate ir."""
  in_file = tmp_path / "model.py"
  in_file.write_text(SAMPLE_TORCH, encoding="utf-8")
  out_file = tmp_path / "out_jax.py"

  exit_code = handle_convert(
    input_path=in_file,
    output_path=out_file,
    source="torch",
    target="jax",
    verify=False,
    strict=False,
    intermediate="ir",
    plugin_settings={},
    json_trace_path=None,
    enable_sharding=False,
  )
  assert exit_code == 0
  assert out_file.exists()
  code = out_file.read_text(encoding="utf-8")
  assert len(code) > 0

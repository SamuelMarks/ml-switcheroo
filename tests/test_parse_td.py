"""Unit tests for scripts/parse_td.py TableGen parser."""

import json
from pathlib import Path
import runpy
from typing import Any, Dict, List
from unittest.mock import patch

from scripts.parse_td import main, parse_td_files


def test_parse_td_files_comprehensive(tmp_path: Path) -> None:
  """Test parse_td_files across various directory structures and TableGen formats.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  # 1. Dialect file mapped in DIR_TO_DIALECT (arith)
  dialect_arith = tmp_path / "mlir" / "Dialect" / "arith" / "IR"
  dialect_arith.mkdir(parents=True)
  content_arith = (
    'def Arith_AddIOp : Arith_Op<"addi">' + chr(10) + 'def Arith_SubIOp : Op<Arith_Dialect, "subi">' + chr(10)
  )
  (dialect_arith / "ArithOps.td").write_text(content_arith, encoding="utf-8")

  # 2. Non-.td file inside Dialect folder (ignored)
  (dialect_arith / "README.txt").write_text("Ignored documentation file", encoding="utf-8")

  # 3. Dialect with unmapped dir_name (falls back to dir_name)
  dialect_custom = tmp_path / "mlir" / "Dialect" / "my_custom" / "IR"
  dialect_custom.mkdir(parents=True)
  content_custom = 'def Custom_Op : CustomBase<"custom.op_a">' + chr(10)
  (dialect_custom / "CustomOps.td").write_text(content_custom, encoding="utf-8")

  # 4. Dialect path with empty dir_name after /Dialect/
  dialect_empty = tmp_path / "mlir" / "Dialect"
  content_empty = 'def Root_Op : RootBase<"root.op">' + chr(10)
  (dialect_empty / "RootOps.td").write_text(content_empty, encoding="utf-8")

  # 5. File inside /IR/ path (defaults to builtin dialect)
  ir_dir = tmp_path / "mlir" / "IR"
  ir_dir.mkdir(parents=True)
  content_ir = 'def Builtin_ModuleOp : Builtin_Op<"builtin.module">' + chr(10)
  (ir_dir / "BuiltinOps.td").write_text(content_ir, encoding="utf-8")

  # 6. File outside of Dialect and IR paths (defaults to unknown dialect)
  other_dir = tmp_path / "mlir" / "Other"
  other_dir.mkdir(parents=True)
  content_other = 'def Other_Op : OtherBase<"other.op">' + chr(10)
  (other_dir / "OtherOps.td").write_text(content_other, encoding="utf-8")

  parsed: Dict[str, List[str]] = parse_td_files(str(tmp_path))

  assert "arith" in parsed
  assert parsed["arith"] == ["addi", "subi"]

  assert "my_custom" in parsed
  assert parsed["my_custom"] == ["custom.op_a"]

  assert "builtin" in parsed
  assert parsed["builtin"] == ["builtin.module"]

  assert "unknown" in parsed
  assert "other.op" in parsed["unknown"]
  assert "rootops.td" in parsed


def test_parse_td_main(tmp_path: Path) -> None:
  """Test main execution function for parse_td.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  dialect_math = tmp_path / "include" / "mlir" / "Dialect" / "math"
  dialect_math.mkdir(parents=True)
  content_math = 'def Math_SinOp : Math_Op<"sin">' + chr(10) + 'def Math_CosOp : Op<Math_Dialect, "cos">' + chr(10)
  (dialect_math / "MathOps.td").write_text(content_math, encoding="utf-8")

  out_file = tmp_path / "output_ops.json"
  exit_code = main(llvm_include_dir=str(tmp_path), output_path=str(out_file))

  assert exit_code == 0
  assert out_file.exists()

  with open(out_file, "r", encoding="utf-8") as f:
    data: Dict[str, Any] = json.load(f)

  assert "math" in data
  assert sorted(data["math"]) == ["cos", "sin"]


def test_parse_td_main_entrypoint(tmp_path: Path, monkeypatch: Any) -> None:
  """Test running scripts/parse_td.py as __main__ module.

  Args:
      tmp_path: Temporary directory fixture from pytest.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.chdir(tmp_path)
  with patch("sys.argv", ["scripts/parse_td.py"]):
    with patch("scripts.parse_td.parse_td_files", return_value={"tensor": ["extract"]}):
      with patch("sys.exit") as mock_exit:
        runpy.run_module("scripts.parse_td", run_name="__main__")
        mock_exit.assert_called_once_with(0)

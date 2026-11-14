"""Unit tests for `scripts/parse_td.py` MLIR TableGen operation parser."""

import json
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.parse_td import main, parse_td_files

_NL = chr(10)


def test_parse_td_files_dialects_and_patterns(tmp_path: Path) -> None:
  """Verify `parse_td_files` across `DIR_TO_DIALECT`, custom dialects, `/IR/`, root `/Dialect/`, and empty files.

  Args:
      tmp_path: Pytest temporary directory fixture.
  """
  dialect_arith = tmp_path / "mlir" / "Dialect" / "Arith" / "IR"
  dialect_arith.mkdir(parents=True)
  (dialect_arith / "ArithOps.td").write_text(
    f'def Arith_AddIOp : Arith_Op<"addi">{_NL}def Arith_SubIOp : Op<Arith_Dialect, "subi">{_NL}',
    encoding="utf-8",
  )
  (dialect_arith / "README.txt").write_text('def Ignored : Arith_Op<"ignored">', encoding="utf-8")

  dialect_custom = tmp_path / "mlir" / "Dialect" / "my_custom" / "IR"
  dialect_custom.mkdir(parents=True)
  (dialect_custom / "CustomOps.td").write_text(f'def Custom_Op : CustomBase<"custom.op_a">{_NL}', encoding="utf-8")

  dialect_root = tmp_path / "mlir" / "Dialect"
  (dialect_root / "RootOps.td").write_text(f'def Root_Op : RootBase<"root.op">{_NL}', encoding="utf-8")

  ir_dir = tmp_path / "mlir" / "IR"
  ir_dir.mkdir(parents=True)
  (ir_dir / "BuiltinOps.td").write_text(f'def Builtin_ModuleOp : Builtin_Op<"builtin.module">{_NL}', encoding="utf-8")

  other_dir = tmp_path / "mlir" / "Other"
  other_dir.mkdir(parents=True)
  (other_dir / "OtherOps.td").write_text(f'def Other_Op : OtherBase<"other.op">{_NL}', encoding="utf-8")
  (other_dir / "EmptyOps.td").write_text(f"// Just comments{_NL}", encoding="utf-8")

  parsed = parse_td_files(str(tmp_path))
  assert parsed["arith"] == ["addi", "subi"]
  assert parsed["my_custom"] == ["custom.op_a"]
  assert parsed["builtin"] == ["builtin.module"]
  assert "other.op" in parsed["unknown"]
  assert "rootops.td" in parsed


def test_parse_td_main_and_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `main()` JSON output generation and `__main__` module execution.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  dialect_math = tmp_path / "include" / "mlir" / "Dialect" / "math"
  dialect_math.mkdir(parents=True)
  (dialect_math / "MathOps.td").write_text(
    f'def Math_SinOp : Math_Op<"sin">{_NL}def Math_CosOp : Op<Math_Dialect, "cos">{_NL}',
    encoding="utf-8",
  )

  out_file = tmp_path / "output_ops.json"
  assert main(llvm_include_dir=str(tmp_path), output_path=str(out_file)) == 0
  data = json.loads(out_file.read_text(encoding="utf-8"))
  assert sorted(data["math"]) == ["cos", "sin"]

  monkeypatch.chdir(tmp_path)
  with (
    patch("sys.argv", ["scripts/parse_td.py"]),
    patch("scripts.parse_td.parse_td_files", return_value={"tensor": ["extract"]}),
    pytest.raises(SystemExit) as exc_info,
  ):
    runpy.run_module("scripts.parse_td", run_name="__main__")
  assert exc_info.value.code == 0

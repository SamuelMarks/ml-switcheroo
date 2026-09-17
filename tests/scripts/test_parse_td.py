"""Tests for scripts/parse_td.py."""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

import scripts.parse_td as parse_td


def test_parse_td_files_dialects_and_regexes(tmp_path: Path) -> None:
  """Test parsing various dialects, paths, and operation definitions."""
  # 1. Dialect folder with known dialect
  arith_dir = tmp_path / "Dialect" / "Arith"
  arith_dir.mkdir(parents=True)
  (arith_dir / "ArithOps.td").write_text(
    """def Arith_AddIOp : Arith_Op<"arith.addi">;
def Arith_SubIOp : Op<Arith_Dialect, "arith.subi">;
""",
    encoding="utf-8",
  )
  # Non-td file should be ignored
  (arith_dir / "ignore.txt").write_text(
    """def IgnoredOp : Arith_Op<"arith.ignored">;
""",
    encoding="utf-8",
  )

  # 2. Dialect folder with unknown dialect name
  custom_dir = tmp_path / "Dialect" / "my_custom"
  custom_dir.mkdir(parents=True)
  (custom_dir / "CustomOps.td").write_text(
    """def CustomOp : Custom_Op<"custom.op">;
""",
    encoding="utf-8",
  )

  # 3. Path containing /Dialect/ directly without subpath
  edge_dir = tmp_path / "Dialect"
  (edge_dir / "direct.td").write_text(
    """def DirectOp : Direct_Op<"direct.op">;
""",
    encoding="utf-8",
  )

  # 4. IR folder (builtin dialect)
  ir_dir = tmp_path / "IR"
  ir_dir.mkdir(parents=True)
  (ir_dir / "BuiltinOps.td").write_text(
    """def ModuleOp : Builtin_Op<"builtin.module">;
""",
    encoding="utf-8",
  )

  # 5. Other folder (unknown dialect)
  other_dir = tmp_path / "Other"
  other_dir.mkdir(parents=True)
  (other_dir / "OtherOps.td").write_text(
    """def OtherOp : Other_Op<"other.op">;
""",
    encoding="utf-8",
  )

  # 6. File with no ops found
  (other_dir / "EmptyOps.td").write_text(
    """// Just comments
""",
    encoding="utf-8",
  )

  result = parse_td.parse_td_files(str(tmp_path))

  assert "arith" in result
  assert "arith.addi" in result["arith"]
  assert "arith.subi" in result["arith"]

  assert "my_custom" in result
  assert "custom.op" in result["my_custom"]

  assert "builtin" in result
  assert "builtin.module" in result["builtin"]

  assert "unknown" in result
  assert "other.op" in result["unknown"]


def test_main(tmp_path: Path) -> None:
  """Test main execution function."""
  mlir_dir = tmp_path / "include" / "mlir" / "Dialect" / "math"
  mlir_dir.mkdir(parents=True)
  (mlir_dir / "MathOps.td").write_text(
    """def Math_SinOp : Math_Op<"math.sin">;
""",
    encoding="utf-8",
  )

  out_file = tmp_path / "ops.json"
  exit_code = parse_td.main(llvm_include_dir=str(tmp_path / "include" / "mlir"), output_path=str(out_file))
  assert exit_code == 0
  assert out_file.exists()

  with open(out_file, "r", encoding="utf-8") as f:
    saved = json.load(f)
  assert "math" in saved
  assert "math.sin" in saved["math"]


def test_main_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test __main__ execution block."""
  import runpy

  monkeypatch.chdir(tmp_path)
  with patch("scripts.parse_td.parse_td_files", return_value={"test": ["op"]}):
    with pytest.raises(SystemExit) as excinfo:
      runpy.run_module("scripts.parse_td", run_name="__main__")
    assert excinfo.value.code == 0

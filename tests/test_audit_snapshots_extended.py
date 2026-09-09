"""Extended test suite for framework snapshot auditing.

Validates multi-target snapshot auditing across PyTorch, JAX, MLX, Keras,
AMD RDNA, and NVIDIA SASS.
"""

import json
from pathlib import Path
from typing import Any, Dict

from scripts.audit_against_snapshots import (
  _flatten_single_framework,
  audit_frameworks,
  audit_inline_snippets,
  audit_python_ast,
  load_snapshots,
  load_snapshots_multi,
)


def test_flatten_single_framework_list() -> None:
  """Test flattening list-based exhaustive ISA snapshots."""
  flat: Dict[str, Dict[str, Any]] = {}
  snap_list = [
    {"mnemonic": "FADD", "desc": "Float add"},
    {"mnemonic": "FMUL", "desc": "Float mul"},
    {"not_a_mnemonic": 123},
  ]
  _flatten_single_framework("nvidia_sass", snap_list, flat)
  assert "FADD" in flat["nvidia_sass"]
  assert "FMUL" in flat["nvidia_sass"]
  assert flat["nvidia_sass"]["FADD"]["desc"] == "Float add"


def test_flatten_single_framework_non_dict() -> None:
  """Test flattening non-dict and non-list snapshot input."""
  flat: Dict[str, Dict[str, Any]] = {}
  _flatten_single_framework("invalid", "not_dict_or_list", flat)
  assert "invalid" in flat
  assert len(flat["invalid"]) == 0


def test_load_snapshots_exhaustive(tmp_path: Path) -> None:
  """Test loading exhaustive ISA snapshots."""
  snap_dir = tmp_path / "snapshots"
  snap_dir.mkdir()

  sass_file = snap_dir / "nvidia_sass_exhaustive.json"
  sass_data = [{"mnemonic": "FADD"}, {"mnemonic": "FABS"}]
  sass_file.write_text(json.dumps(sass_data))

  rdna_file = snap_dir / "amd_rdna_exhaustive.json"
  rdna_data = [{"mnemonic": "v_add_f32"}, {"mnemonic": "v_abs_f32"}]
  rdna_file.write_text(json.dumps(rdna_data))

  snapshots = load_snapshots(snap_dir)
  assert "nvidia_sass" in snapshots
  assert "rdna" in snapshots
  assert "FADD" in snapshots["nvidia_sass"]
  assert "v_add_f32" in snapshots["rdna"]


def test_load_snapshots_multi_with_isa(tmp_path: Path) -> None:
  """Test load_snapshots_multi with directories containing ISA snapshots."""
  dir1 = tmp_path / "dir1"
  dir1.mkdir()
  sass_file = dir1 / "nvidia_sass_exhaustive.json"
  sass_file.write_text(json.dumps([{"mnemonic": "FFMA"}]))

  snapshots = load_snapshots_multi([dir1])
  assert "nvidia_sass" in snapshots
  assert "FFMA" in snapshots["nvidia_sass"]


def test_audit_frameworks_with_sass_and_macros() -> None:
  """Test audit_frameworks handling SASS instructions and Macro directives."""

  class MockManager:
    """Mock semantics manager for auditing."""

    def __init__(self) -> None:
      """Initialize mock data."""
      self.data: Dict[str, Any] = {
        "ValidOp": {
          "variants": {
            "nvidia_sass": {"api": "FADD", "args": {}},
            "rdna": {"api": "; Macro.LayerNorm", "args": {}},
            "torch": {"api": "Macro.Linear", "args": {}},
          }
        },
        "HallucinatedSass": {
          "variants": {
            "nvidia_sass": {"api": "NON_EXISTENT_OP", "args": {}},
          }
        },
      }

  mgr = MockManager()
  snapshots: Dict[str, Dict[str, Any]] = {
    "nvidia_sass": {"FADD": {"mnemonic": "FADD"}},
    "rdna": {},
    "torch": {},
  }

  errors = audit_frameworks(mgr, snapshots)  # type: ignore[arg-type]
  assert any("maps to hallucinated API: 'NON_EXISTENT_OP'" in err for err in errors)
  assert not any("Macro.LayerNorm" in err for err in errors)
  assert not any("Macro.Linear" in err for err in errors)


def test_audit_inline_snippets_sass() -> None:
  """Test audit_inline_snippets recognizing nvidia_sass prefix."""

  class MockManager:
    """Mock semantics manager."""

    def __init__(self) -> None:
      """Initialize data."""
      self.data: Dict[str, Any] = {
        "TestOp": {
          "variants": {
            "nvidia_sass": {"macro_template": "nvidia_sass.FADD(R0, R1)"},
            "Hallucinated": {"macro_template": "nvidia_sass.BAD_OP(R0)"},
          }
        }
      }

  mgr = MockManager()
  snapshots: Dict[str, Dict[str, Any]] = {
    "nvidia_sass": {"nvidia_sass.FADD": {}},
  }

  errors = audit_inline_snippets(mgr, snapshots)  # type: ignore[arg-type]
  assert any("Inline snippet hallucinated API in TestOp: 'nvidia_sass.BAD_OP'" in err for err in errors)
  assert not any("nvidia_sass.FADD" in err for err in errors)


def test_audit_python_ast_sass(tmp_path: Path) -> None:
  """Test audit_python_ast auditing nvidia_sass calls in python files."""
  py_file = tmp_path / "test_script.py"
  py_file.write_text("""import nvidia_sass
nvidia_sass.FADD()
nvidia_sass.UNKNOWN()
""")

  snapshots: Dict[str, Dict[str, Any]] = {
    "nvidia_sass": {"nvidia_sass.FADD": {}},
  }

  errors = audit_python_ast([tmp_path], snapshots)
  assert any("Programmatic API call hallucinated" in err and "nvidia_sass.UNKNOWN" in err for err in errors)
  assert not any("nvidia_sass.FADD" in err for err in errors)

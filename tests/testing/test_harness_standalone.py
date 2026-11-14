"""Test suite for the Harness Standalone module."""

import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import JsonValue

from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.testing.harness_generator import HarnessGenerator


class DynKerasAdapter:
  """Test double `DynKerasAdapter` used in harness standalone tests."""

  declared_magic_args: ClassVar[list[str]] = []
  harness_imports: ClassVar[list[str]] = []

  def get_harness_init_code(self) -> str:
    """Gets harness initialization code.

    Returns:
        Resulting `str` from get harness init code.
    """
    return ""

  def convert(self, data: JsonValue) -> str:
    """Convert input payload in test stub.

    Args:
        data: Hypothesis DataObject or input tensor payload.

    Returns:
        Transpiled source code string or rewritten CST module.
    """
    return "KerasMock(" + str(data) + ")"


def _run_harness(path: Path) -> subprocess.CompletedProcess[str]:
  """Execute `copy` to support run harness in harness standalone tests.

  Args:
      path: Filesystem path under test (`Path`).

  Returns:
      Resulting `subprocess.CompletedProcess[str]` from run harness.
  """
  env: dict[str, str] = os.environ.copy()
  env.pop("PYTHONPATH", None)
  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


def test_dynamic_shim_generation(tmp_path: Path) -> None:
  """Verify `HarnessGenerator` handles dynamic shim generation as expected.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  register_framework("mock_keras")(DynKerasAdapter)
  gen: HarnessGenerator = HarnessGenerator()
  harness_path: Path = tmp_path / "verify_shim.py"
  gen.generate(tmp_path, tmp_path, harness_path)
  content: str = harness_path.read_text()
  assert "if framework == 'mock_keras':" in content or "elif framework == 'mock_keras':" in content
  assert 'return "KerasMock(" + str(data) + ")"' in content or "return 'KerasMock(' + str(data) + ')'" in content


def test_harness_execution_match(tmp_path: Path) -> None:
  """Verify `HarnessGenerator` returns `0` for harness execution match.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  src_file: Path = tmp_path / "mod_src.py"
  src_file.write_text("def my_op(x): return x * 2")
  tgt_file: Path = tmp_path / "mod_tgt.py"
  tgt_file.write_text("def my_op(x): return x * 2")
  harness_path: Path = tmp_path / "verify_match.py"
  gen: HarnessGenerator = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path, source_framework="numpy", target_framework="numpy")
  result: subprocess.CompletedProcess[str] = _run_harness(harness_path)
  if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
  assert result.returncode == 0
  assert "✅ my_op: Match" in result.stdout

"""Test suite for the Harness Standalone module."""

import os
import subprocess
import sys
from pathlib import Path

from pydantic import JsonValue

from ml_switcheroo.frameworks import register_framework
from ml_switcheroo.testing.harness_generator import HarnessGenerator


class DynKerasAdapter:
  """Test helper class DynKerasAdapter."""

  declared_magic_args: list[str] = []
  harness_imports: list[str] = []

  def get_harness_init_code(self) -> str:
    """Gets harness initialization code.

    Returns:
        Resulting `str` from get harness init code.
    """
    return ""

  def convert(self, data: JsonValue) -> str:
    """Converts .

    Args:
        data: Hypothesis DataObject or input tensor payload.

    Returns:
        Transpiled source code string or rewritten CST module.
    """
    return "KerasMock(" + str(data) + ")"


def _run_harness(path: Path) -> subprocess.CompletedProcess[str]:
  """Helper to  run harness.

  Args:
      path: The path (Path) for  run harness.

  Returns:
      Resulting `subprocess.CompletedProcess[str]` from run harness.
  """
  env: dict[str, str] = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]
  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


def test_dynamic_shim_generation(tmp_path: Path) -> None:
  """Verifies the behavior of dynamic shim generation.

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
  """Verifies the behavior of harness execution match.

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

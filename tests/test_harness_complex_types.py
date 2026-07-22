"""Test suite for the Harness Complex Types module."""

import sys
import subprocess
import os
from pathlib import Path
from ml_switcheroo.testing.harness_generator import HarnessGenerator


def _run_harness(path: Path) -> subprocess.CompletedProcess:
  """Helper to  run harness."""
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]
  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


def test_harness_complex_list(tmp_path):
  """Verifies the behavior of harness complex list."""
  src_file = tmp_path / "mod_list_src.py"
  src_file.write_text(
    '\ndef compute(param):\n    # expect param to be list of ints\n    if not isinstance(param, list): raise ValueError("Not a list")\n    if not all(isinstance(x, int) for x in param): raise ValueError("Not ints")\n    return sum(param)\n'
  )
  tgt_file = tmp_path / "mod_list_tgt.py"
  tgt_file.write_text(src_file.read_text())
  harness_path = tmp_path / "verify.py"
  semantics = {"compute": {"std_args": [("param", "List[int]")]}}
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, harness_path, source_fw="numpy", target_fw="numpy", semantics=semantics)
  result = _run_harness(harness_path)
  if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
  assert result.returncode == 0
  assert "✅ compute: Match" in result.stdout


def test_harness_tuple_variadic(tmp_path):
  """Verifies the behavior of harness tuple variadic."""
  src_file = tmp_path / "mod_tup.py"
  src_file.write_text(
    '\ndef process(items):\n    if not isinstance(items, tuple): raise ValueError("Not tuple")\n    if not all(isinstance(x, int) for x in items): raise ValueError("Not ints")\n    return len(items)\n'
  )
  tgt_file = tmp_path / "mod_tup_tgt.py"
  tgt_file.write_text(src_file.read_text())
  semantics = {"process": {"std_args": [("items", "Tuple[int, ...]")]}}
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, tmp_path / "verify.py", source_fw="numpy", target_fw="numpy", semantics=semantics)
  result = _run_harness(tmp_path / "verify.py")
  assert result.returncode == 0
  assert "✅ process: Match" in result.stdout


def test_harness_nested_dict(tmp_path):
  """Verifies the behavior of harness nested dictionary."""
  src_file = tmp_path / "mod_dict.py"
  src_file.write_text(
    '\ndef config(data):\n    if not isinstance(data, dict): raise ValueError("Not dict")\n    # Verify values are lists\n    for v in data.values():\n        if not isinstance(v, list): raise ValueError("Value not list")\n    return 1\n'
  )
  tgt_file = tmp_path / "mod_dict_tgt.py"
  tgt_file.write_text(src_file.read_text())
  semantics = {"config": {"std_args": [("data", "Dict[str, List[int]]")]}}
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, tmp_path / "verify.py", source_fw="numpy", target_fw="numpy", semantics=semantics)
  result = _run_harness(tmp_path / "verify.py")
  assert result.returncode == 0
  assert "✅ config: Match" in result.stdout


def test_harness_recursive_conversion_list_of_arrays(tmp_path):
  """Verifies the behavior of harness recursive conversion list of arrays."""
  src_file = tmp_path / "mod_rec.py"
  src_file.write_text(
    "\nimport numpy as np\ndef batched(tensors):\n    if not isinstance(tensors, list): return -1\n    if not tensors: return 0\n    if not isinstance(tensors[0], np.ndarray): return -2\n    return tensors[0].shape[0]\n"
  )
  tgt_file = tmp_path / "mod_rec_tgt.py"
  tgt_file.write_text(src_file.read_text())
  semantics = {"batched": {"std_args": [("tensors", "List[Array]")]}}
  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, tmp_path / "verify.py", source_fw="numpy", target_fw="numpy", semantics=semantics)
  result = _run_harness(tmp_path / "verify.py")
  assert result.returncode == 0
  assert "✅ batched: Match" in result.stdout


def test_harness_hints_json_injection(tmp_path):
  """Verifies the behavior of harness hints JSON injection."""
  harness_path = tmp_path / "verify.py"
  semantics = {"op": {"std_args": [("x", "int")]}}
  gen = HarnessGenerator()
  gen.generate(tmp_path, tmp_path, harness_path, semantics=semantics)
  content = harness_path.read_text()
  assert 'hints_json_str=r\'{"op": {"x": "int"}}\'' in content

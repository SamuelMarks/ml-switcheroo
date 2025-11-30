"""
Tests for Robust Harness Fuzzer Logic.

Verifies that the generated standalone harness script:
1. Embeds the robust Fuzzer capable of parsing type hints.
2. Correctly parses JSON hints string injected during generation.
3. Generates complex container types (List, Tuple, Dict) matching hints.
4. Handles recursive structure adaptation (e.g., converting List of Tensors).
"""

import sys
import subprocess
import os
from pathlib import Path
from ml_switcheroo.testing.harness_generator import HarnessGenerator


def _run_harness(path: Path) -> subprocess.CompletedProcess:
  """Runs the generated harness in a clean subprocess."""
  # Remove PYTHONPATH modification to ensure it's truly standalone
  env = os.environ.copy()
  if "PYTHONPATH" in env:
    del env["PYTHONPATH"]

  return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, env=env)


def test_harness_complex_list(tmp_path):
  """
  Scenario: Function expects List[int].
  """
  src_file = tmp_path / "mod_list_src.py"
  src_file.write_text("""
def compute(param):
    # expect param to be list of ints
    if not isinstance(param, list): raise ValueError("Not a list")
    if not all(isinstance(x, int) for x in param): raise ValueError("Not ints")
    return sum(param)
""")

  tgt_file = tmp_path / "mod_list_tgt.py"
  tgt_file.write_text(src_file.read_text())  # Identities match

  harness_path = tmp_path / "verify.py"

  # Semantics providing hint
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
  """
  Scenario: Function expects Tuple[int, ...].
  """
  src_file = tmp_path / "mod_tup.py"
  src_file.write_text("""
def process(items):
    if not isinstance(items, tuple): raise ValueError("Not tuple")
    if not all(isinstance(x, int) for x in items): raise ValueError("Not ints")
    return len(items)
""")
  tgt_file = tmp_path / "mod_tup_tgt.py"
  tgt_file.write_text(src_file.read_text())

  semantics = {"process": {"std_args": [("items", "Tuple[int, ...]")]}}

  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, tmp_path / "verify.py", source_fw="numpy", target_fw="numpy", semantics=semantics)

  result = _run_harness(tmp_path / "verify.py")
  assert result.returncode == 0
  assert "✅ process: Match" in result.stdout


def test_harness_nested_dict(tmp_path):
  """
  Scenario: Function expects Dict[str, List[int]].
  """
  src_file = tmp_path / "mod_dict.py"
  src_file.write_text("""
def config(data):
    if not isinstance(data, dict): raise ValueError("Not dict")
    # Verify values are lists
    for v in data.values():
        if not isinstance(v, list): raise ValueError("Value not list")
    return 1
""")
  tgt_file = tmp_path / "mod_dict_tgt.py"
  tgt_file.write_text(src_file.read_text())

  semantics = {"config": {"std_args": [("data", "Dict[str, List[int]]")]}}

  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, tmp_path / "verify.py", source_fw="numpy", target_fw="numpy", semantics=semantics)

  result = _run_harness(tmp_path / "verify.py")
  assert result.returncode == 0
  assert "✅ config: Match" in result.stdout


def test_harness_recursive_conversion_list_of_arrays(tmp_path):
  """
  Scenario: Function takes List[Array].
  Verifies that `adapt` recursively converts inner items to frameworks.
  Since we use "numpy" backend in test to stay lightweight,
  conversion is idempotent, but we verify struct integrity.
  """
  src_file = tmp_path / "mod_rec.py"
  src_file.write_text("""
import numpy as np
def batched(tensors):
    if not isinstance(tensors, list): return -1
    if not tensors: return 0
    if not isinstance(tensors[0], np.ndarray): return -2
    return tensors[0].shape[0]
""")
  tgt_file = tmp_path / "mod_rec_tgt.py"
  tgt_file.write_text(src_file.read_text())

  semantics = {"batched": {"std_args": [("tensors", "List[Array]")]}}

  gen = HarnessGenerator()
  gen.generate(src_file, tgt_file, tmp_path / "verify.py", source_fw="numpy", target_fw="numpy", semantics=semantics)

  result = _run_harness(tmp_path / "verify.py")
  assert result.returncode == 0
  assert "✅ batched: Match" in result.stdout


def test_harness_hints_json_injection(tmp_path):
  """
  Verify the generator actually writes JSON hints into the script file.
  """
  harness_path = tmp_path / "verify.py"
  semantics = {"op": {"std_args": [("x", "int")]}}

  gen = HarnessGenerator()
  # Dummy paths
  gen.generate(tmp_path, tmp_path, harness_path, semantics=semantics)

  content = harness_path.read_text()
  assert 'hints_json_str=r\'{"op": {"x": "int"}}\'' in content

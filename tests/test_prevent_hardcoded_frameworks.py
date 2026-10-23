"""Unit and integration tests for scripts/prevent_hardcoded_frameworks.sh."""

import subprocess
from pathlib import Path


def test_prevent_hardcoded_frameworks_compliant(tmp_path: Path) -> None:
  """Test prevent_hardcoded_frameworks.sh exits with 0 on compliant files.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "prevent_hardcoded_frameworks.sh"
  valid_py = tmp_path / "valid.py"
  valid_py.write_text("def dispatch(adapter):" + chr(10) + "    return adapter.name" + chr(10), encoding="utf-8")

  res = subprocess.run(["bash", str(script), str(valid_py)], capture_output=True, text=True)
  assert res.returncode == 0
  assert "Error:" not in res.stdout


def test_prevent_hardcoded_frameworks_detected(tmp_path: Path) -> None:
  """Test prevent_hardcoded_frameworks.sh detects hardcoded framework conditions.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "prevent_hardcoded_frameworks.sh"

  forbidden_examples = [
    "if framework == 'torch':" + chr(10) + "    pass" + chr(10),
    'if target == "jax":' + chr(10) + "    pass" + chr(10),
    "if source == 'mlx':" + chr(10) + "    pass" + chr(10),
    'if framework == "tensorflow":' + chr(10) + "    pass" + chr(10),
    "if target == 'keras':" + chr(10) + "    pass" + chr(10),
    'if source == "numpy":' + chr(10) + "    pass" + chr(10),
    'return self.target in {"jax", "flax_nnx", "paxml"}' + chr(10),
    'if self.target_framework in ("flax", "paxml"):' + chr(10) + "    pass" + chr(10),
  ]

  for idx, code in enumerate(forbidden_examples):
    bad_py = tmp_path / f"bad_{idx}.py"
    bad_py.write_text(code, encoding="utf-8")
    res = subprocess.run(["bash", str(script), str(bad_py)], capture_output=True, text=True)
    assert res.returncode == 1
    assert "Error: Hardcoded framework routing found" in res.stdout


def test_prevent_hardcoded_frameworks_non_python_files(tmp_path: Path) -> None:
  """Test prevent_hardcoded_frameworks.sh ignores non-Python files.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "prevent_hardcoded_frameworks.sh"
  doc_file = tmp_path / "doc.md"
  doc_file.write_text("Example check: if framework == 'torch':", encoding="utf-8")

  res = subprocess.run(["bash", str(script), str(doc_file)], capture_output=True, text=True)
  assert res.returncode == 0


def test_prevent_hardcoded_frameworks_empty_args() -> None:
  """Test prevent_hardcoded_frameworks.sh with zero arguments exits cleanly."""
  script = Path(__file__).resolve().parent.parent / "scripts" / "prevent_hardcoded_frameworks.sh"
  res = subprocess.run(["bash", str(script)], capture_output=True, text=True)
  assert res.returncode == 0


def test_prevent_hardcoded_frameworks_reflection_evasion(tmp_path: Path) -> None:
  """Test prevent_hardcoded_frameworks.sh detects reflection type-checker evasions.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "prevent_hardcoded_frameworks.sh"
  evasion_examples = [
    'x = getattr({"v": node}, "__getitem__")("v")\n',
    'getattr(registry, "__setitem__")("k", val)\n',
    'getattr(cls, "__call__")()\n',
  ]
  for idx, code in enumerate(evasion_examples):
    bad_py = tmp_path / f"evasion_{idx}.py"
    bad_py.write_text(code, encoding="utf-8")
    res = subprocess.run(["bash", str(script), str(bad_py)], capture_output=True, text=True)
    assert res.returncode == 1
    assert "Error: Reflection type-checker evasion found" in res.stdout

  core_dir = tmp_path / "ml_switcheroo" / "core"
  core_dir.mkdir(parents=True)
  bad_dict_py = core_dir / "bad_dict.py"
  bad_dict_py.write_text('_MAP = {\n  "torch": "nn",\n}\n', encoding="utf-8")
  res_dict = subprocess.run(["bash", str(script), str(bad_dict_py)], capture_output=True, text=True)
  assert res_dict.returncode == 1
  assert "Error: Hardcoded framework dictionary key found" in res_dict.stdout

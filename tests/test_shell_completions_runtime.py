"""Runtime execution and syntax verification tests for shell completions."""

from pathlib import Path
import shutil
import subprocess


def test_bash_completion_runtime_execution() -> None:
  """Test that ml_switcheroo.bash is syntactically valid and registers completion handler."""
  completions_dir = Path(__file__).resolve().parent.parent / "scripts" / "completions"
  bash_script = completions_dir / "ml_switcheroo.bash"

  # 1. Syntax check
  res_syntax = subprocess.run(["bash", "-n", str(bash_script)], capture_output=True, text=True)
  assert res_syntax.returncode == 0, f"Bash syntax check failed: {res_syntax.stderr}"

  # 2. Registration check
  cmd = f'. "{bash_script}" && complete -p ml_switcheroo'
  res_reg = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
  assert res_reg.returncode == 0
  assert "complete -F _ml_switcheroo_completion ml_switcheroo" in res_reg.stdout


def test_zsh_completion_syntax() -> None:
  """Test that ml_switcheroo.zsh is syntactically valid in zsh."""
  if not shutil.which("zsh"):
    return

  completions_dir = Path(__file__).resolve().parent.parent / "scripts" / "completions"
  zsh_script = completions_dir / "ml_switcheroo.zsh"

  res = subprocess.run(["zsh", "-n", str(zsh_script)], capture_output=True, text=True)
  assert res.returncode == 0, f"Zsh syntax check failed: {res.stderr}"


def test_fish_completion_syntax_and_structure() -> None:
  """Test that ml_switcheroo.fish contains valid fish completion directives."""
  completions_dir = Path(__file__).resolve().parent.parent / "scripts" / "completions"
  fish_script = completions_dir / "ml_switcheroo.fish"
  content = fish_script.read_text(encoding="utf-8")

  # Verify core directives and absence of removed backends
  assert "complete -c ml_switcheroo" in content
  assert "-l source" in content
  assert "-l target" in content
  assert "torch" in content
  assert "ml_switcheroo_ir" in content
  assert "wasm" not in content
  assert "cpp" not in content

  # If fish is installed, run syntax validation
  if shutil.which("fish"):
    res = subprocess.run(["fish", "-n", str(fish_script)], capture_output=True, text=True)
    assert res.returncode == 0, f"Fish syntax check failed: {res.stderr}"

"""Unit and integration tests for scripts/suggest_gen_llm_loop.sh."""

import os
from pathlib import Path
import stat
import subprocess


def test_missing_code2prompt() -> None:
  """Test suggest_gen_llm_loop.sh exits with code 2 when code2prompt is not installed."""
  script = Path(__file__).resolve().parent.parent / "scripts" / "suggest_gen_llm_loop.sh"
  # Standard system PATH without code2prompt
  env = {"PATH": "/bin:/usr/bin"}

  res = subprocess.run(["/bin/sh", str(script)], env=env, capture_output=True, text=True)
  assert res.returncode == 2
  assert "Install code2prompt then try again" in res.stdout


def test_missing_pbcopy(tmp_path: Path) -> None:
  """Test suggest_gen_llm_loop.sh exits with code 2 when pbcopy is missing.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "suggest_gen_llm_loop.sh"

  mock_bin_dir = tmp_path / "bin"
  mock_bin_dir.mkdir()
  fake_c2p = mock_bin_dir / "code2prompt"
  fake_c2p.write_text("#!/bin/sh" + chr(10) + "exit 0" + chr(10), encoding="utf-8")
  fake_c2p.chmod(fake_c2p.stat().st_mode | stat.S_IEXEC)

  # PATH has code2prompt but pbcopy is shadowed
  fake_no_pbcopy = tmp_path / "no_pbcopy"
  fake_no_pbcopy.mkdir()
  for standard_cmd in ["sh", "echo", "printf", "cat", "dirname"]:
    pass

  # We mock `command` or ensure pbcopy is absent
  # On macOS pbcopy is in /usr/bin. We can create a sandbox PATH where pbcopy does not exist:
  sandbox_bin = tmp_path / "sandbox_bin"
  sandbox_bin.mkdir()
  (sandbox_bin / "code2prompt").write_text("#!/bin/sh" + chr(10) + "exit 0" + chr(10), encoding="utf-8")
  (sandbox_bin / "code2prompt").chmod(0o755)

  # Copy minimal required binaries (like sh, printf, cat) without pbcopy
  for b in ["sh", "printf", "cat", "test", "["]:
    bin_path = Path("/bin") / b
    if bin_path.exists():
      (sandbox_bin / b).symlink_to(bin_path)
    else:
      usr_bin = Path("/usr/bin") / b
      if usr_bin.exists():
        (sandbox_bin / b).symlink_to(usr_bin)

  res = subprocess.run(["/bin/sh", str(script)], env={"PATH": str(sandbox_bin)}, capture_output=True, text=True)
  assert res.returncode == 2
  assert "Set `pbcopy` `alias` then try again" in res.stdout


def test_loop_quit_and_invalid_input(tmp_path: Path) -> None:
  """Test loop input handling including invalid characters and clean exit commands.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "suggest_gen_llm_loop.sh"

  mock_bin = tmp_path / "bin"
  mock_bin.mkdir()
  for cmd in ["code2prompt", "pbcopy", "pbpaste"]:
    c = mock_bin / cmd
    c.write_text("#!/bin/sh" + chr(10) + "cat" + chr(10), encoding="utf-8")
    c.chmod(c.stat().st_mode | stat.S_IEXEC)

  env = os.environ.copy()
  env["PATH"] = f"{mock_bin}:{env.get('PATH', '')}"

  # Input sequence: empty line, non-digit invalid input, then quit ('q')
  stdin_input = chr(10) + "invalid_str" + chr(10) + "q" + chr(10)
  res = subprocess.run(
    ["/bin/sh", str(script)],
    env=env,
    input=stdin_input,
    capture_output=True,
    text=True,
  )
  assert res.returncode == 0
  assert "Invalid input. Please enter a valid number." in res.stdout


def test_loop_exec_on_num_file_nonexistent(tmp_path: Path) -> None:
  """Test exec_on_num when target prompt file does not exist.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  script = Path(__file__).resolve().parent.parent / "scripts" / "suggest_gen_llm_loop.sh"

  mock_bin = tmp_path / "bin"
  mock_bin.mkdir()
  for cmd in ["code2prompt", "pbcopy", "pbpaste"]:
    c = mock_bin / cmd
    c.write_text("#!/bin/sh" + chr(10) + "cat" + chr(10), encoding="utf-8")
    c.chmod(c.stat().st_mode | stat.S_IEXEC)

  env = os.environ.copy()
  env["PATH"] = f"{mock_bin}:{env.get('PATH', '')}"
  env["BASE_DIR"] = str(tmp_path / "empty_dir")
  (tmp_path / "empty_dir").mkdir()

  # Request number 1, then 'n' (next number), then quit
  stdin_input = "1" + chr(10) + "n" + chr(10) + "q" + chr(10)
  res = subprocess.run(
    ["/bin/sh", str(script)],
    env=env,
    input=stdin_input,
    capture_output=True,
    text=True,
  )
  assert res.returncode == 0
  assert "File nonexistent" in res.stdout

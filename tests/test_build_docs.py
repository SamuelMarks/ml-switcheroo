"""Tests for scripts/build_docs.py."""

import sys
import subprocess
from pathlib import Path
from unittest import mock
import pytest

# Add scripts directory to sys.path to import it
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir.resolve()))

import build_docs  # noqa: E402


@pytest.fixture
def mock_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  """Sets up mock project environment."""
  project_root = tmp_path / "project"
  docs_dir = project_root / "docs"
  build_dir = docs_dir / "_build"

  monkeypatch.setattr(build_docs, "PROJECT_ROOT", project_root)
  monkeypatch.setattr(build_docs, "DOCS_DIR", docs_dir)
  monkeypatch.setattr(build_docs, "BUILD_DIR", build_dir)

  project_root.mkdir()
  docs_dir.mkdir()

  return project_root, docs_dir, build_dir


def test_clean(mock_env, monkeypatch):
  """Tests cleaning of build directories and copied root files."""
  project_root, docs_dir, build_dir = mock_env

  build_dir.mkdir()
  api_dir = docs_dir / "api"
  api_dir.mkdir()
  ops_dir = docs_dir / "ops"
  ops_dir.mkdir()

  dest = docs_dir / "README.md"
  dest.write_text("test")

  monkeypatch.setattr(build_docs, "ROOT_FILES", ("README.md",))

  build_docs.clean()

  assert not build_dir.exists()
  assert not api_dir.exists()
  assert not ops_dir.exists()
  assert not dest.exists()


def test_clean_no_dirs(mock_env):
  """Tests clean when directories do not exist."""
  build_docs.clean()  # Should not raise


def test_copy_root_files(mock_env, monkeypatch, capsys):
  """Tests copying root files."""
  project_root, docs_dir, _ = mock_env

  (project_root / "README.md").write_text("test")
  monkeypatch.setattr(build_docs, "ROOT_FILES", ("README.md", "MISSING.md"))

  build_docs.copy_root_files()

  assert (docs_dir / "README.md").exists()
  out, _ = capsys.readouterr()
  assert "Warning: MISSING.md not found" in out


def test_build_wheel_success(mock_env, monkeypatch):
  """Tests successful wheel build."""
  project_root, _, _ = mock_env
  dist_dir = project_root / "dist"
  dist_dir.mkdir()

  mock_run = mock.Mock()
  monkeypatch.setattr(build_docs.subprocess, "run", mock_run)

  build_docs.build_wheel()

  assert not dist_dir.exists()
  mock_run.assert_called_once_with(["uv", "build", "--wheel"], cwd=project_root, check=True, capture_output=True)


def test_build_wheel_failure(mock_env, monkeypatch, capsys):
  """Tests failed wheel build."""
  project_root, _, _ = mock_env

  error = subprocess.CalledProcessError(1, ["uv", "build"], stderr=b"error message")
  mock_run = mock.Mock(side_effect=error)
  monkeypatch.setattr(build_docs.subprocess, "run", mock_run)

  with pytest.raises(SystemExit) as exc:
    build_docs.build_wheel()

  assert exc.value.code == 1
  out, _ = capsys.readouterr()
  assert "Failed to build wheel" in out
  assert "error message" in out


def test_calculate_unique_variants_success(monkeypatch, capsys):
  """Tests variant calculation success."""

  class MockManager:
    """Mock manager."""

    _reverse_index = [1, 2, 3]

  # Create a dummy module to avoid importing the real SemanticsManager which could fail or change
  import sys

  mock_module = type(sys)("ml_switcheroo.semantics.manager")
  mock_module.SemanticsManager = MockManager
  sys.modules["ml_switcheroo.semantics.manager"] = mock_module
  monkeypatch.setenv("CI", "false")

  build_docs.calculate_unique_variants()
  out, _ = capsys.readouterr()
  assert "Calculated unique cross-framework variants: 3" in out


def test_calculate_unique_variants_ci_fail(monkeypatch, capsys):
  """Tests variant calculation failure in CI."""

  class MockManager:
    """Mock manager."""

    _reverse_index = [1, 2, 3]  # Below 1860

  import sys

  mock_module = type(sys)("ml_switcheroo.semantics.manager")
  mock_module.SemanticsManager = MockManager
  sys.modules["ml_switcheroo.semantics.manager"] = mock_module
  monkeypatch.setenv("CI", "true")

  with pytest.raises(SystemExit):
    build_docs.calculate_unique_variants()


def test_calculate_unique_variants_exception(monkeypatch, capsys):
  """Tests variant calculation handles exceptions."""
  import sys

  if "ml_switcheroo.semantics.manager" in sys.modules:
    del sys.modules["ml_switcheroo.semantics.manager"]
  monkeypatch.setitem(sys.modules, "ml_switcheroo.semantics.manager", None)

  build_docs.calculate_unique_variants()
  out, _ = capsys.readouterr()
  assert "Failed to calculate variants" in out


def test_build_not_all(mock_env, monkeypatch):
  """Tests sphinx build (not all)."""
  _, docs_dir, build_dir = mock_env

  mock_calc = mock.Mock()
  monkeypatch.setattr(build_docs, "calculate_unique_variants", mock_calc)
  mock_wheel = mock.Mock()
  monkeypatch.setattr(build_docs, "build_wheel", mock_wheel)

  mock_run = mock.Mock()
  mock_run.return_value.returncode = 0
  monkeypatch.setattr(build_docs.subprocess, "run", mock_run)
  monkeypatch.delenv("BUILD_ALL_DOCS", raising=False)

  ret = build_docs.build(build_all=False)

  assert ret == 0
  mock_calc.assert_called_once()
  mock_wheel.assert_called_once()

  cmd = mock_run.call_args[0][0]
  assert cmd == [
    sys.executable,
    "-m",
    "sphinx",
    "-b",
    "html",
    str(docs_dir),
    str(build_dir / "html"),
    str(docs_dir / "index.md"),
  ]
  env = mock_run.call_args[1]["env"]
  assert env["BUILD_ALL_DOCS"] == "0"


def test_build_all(mock_env, monkeypatch):
  """Tests full sphinx build."""
  _, docs_dir, build_dir = mock_env

  monkeypatch.setattr(build_docs, "calculate_unique_variants", mock.Mock())
  monkeypatch.setattr(build_docs, "build_wheel", mock.Mock())

  mock_run = mock.Mock()
  mock_run.return_value.returncode = 0
  monkeypatch.setattr(build_docs.subprocess, "run", mock_run)
  monkeypatch.setenv("BUILD_ALL_DOCS", "1")

  build_docs.build(build_all=True)

  cmd = mock_run.call_args[0][0]
  assert cmd == [sys.executable, "-m", "sphinx", "-b", "html", str(docs_dir), str(build_dir / "html")]


def test_main_success(mock_env, monkeypatch, capsys):
  """Tests main execution block."""
  project_root, docs_dir, build_dir = mock_env

  mock_clean = mock.Mock()
  monkeypatch.setattr(build_docs, "clean", mock_clean)
  mock_copy = mock.Mock()
  monkeypatch.setattr(build_docs, "copy_root_files", mock_copy)
  mock_build = mock.Mock(return_value=0)
  monkeypatch.setattr(build_docs, "build", mock_build)

  monkeypatch.setattr(build_docs, "ROOT_FILES", ("README.md",))
  (docs_dir / "README.md").write_text("test")  # To test finally block cleanup

  test_args = ["build_docs.py", "--build-all"]
  with mock.patch.object(sys, "argv", test_args):
    with pytest.raises(SystemExit) as exc:
      build_docs.main()

  assert exc.value.code == 0
  mock_clean.assert_called_once()
  mock_copy.assert_called_once()
  mock_build.assert_called_once_with(build_all=True)

  assert not (docs_dir / "README.md").exists()  # Cleaned up

  out, _ = capsys.readouterr()
  assert "Documentation built successfully" in out


def test_main_failure(mock_env, monkeypatch):
  """Tests main when build fails."""
  monkeypatch.setattr(build_docs, "clean", mock.Mock())
  monkeypatch.setattr(build_docs, "copy_root_files", mock.Mock())
  monkeypatch.setattr(build_docs, "build", mock.Mock(return_value=1))

  test_args = ["build_docs.py"]
  with mock.patch.object(sys, "argv", test_args):
    with pytest.raises(SystemExit) as exc:
      build_docs.main()

  assert exc.value.code == 1


def test_main_clean_exception(mock_env, monkeypatch):
  """Tests main when clean throws an exception."""
  project_root, docs_dir, _ = mock_env

  monkeypatch.setattr(build_docs, "ROOT_FILES", ("README.md",))
  (docs_dir / "README.md").write_text("test")  # To test finally block cleanup

  mock_clean = mock.Mock(side_effect=Exception("Failed to clean"))
  monkeypatch.setattr(build_docs, "clean", mock_clean)

  test_args = ["build_docs.py"]
  with mock.patch.object(sys, "argv", test_args):
    with pytest.raises(Exception):
      build_docs.main()

  assert not (docs_dir / "README.md").exists()  # Cleaned up in finally block


def test_main_sys_exit_mocked(mock_env, monkeypatch):
  """Tests main execution block with runpy."""
  import runpy

  # When runpy is used, it executes the file directly and doesn't use the patched
  # functions in the already imported `build_docs` module.
  # So we mock the underlying components it calls instead, like subprocess and sys.argv
  monkeypatch.setattr(subprocess, "run", mock.Mock(return_value=mock.Mock(returncode=0)))

  test_args = ["build_docs.py"]
  with mock.patch.object(sys, "argv", test_args):
    with pytest.raises(SystemExit) as exc:
      runpy.run_path(str(scripts_dir / "build_docs.py"), run_name="__main__")
  assert exc.value.code == 0

"""Tests for scripts/build_docs.py."""

import sys
import subprocess
from pathlib import Path
from unittest import mock
import pytest
from typing import Tuple, Any

# Add scripts directory to sys.path to import it
scripts_dir: Path = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir.resolve()))

import build_docs  # noqa: E402


@pytest.fixture
def mock_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tuple[Path, Path, Path]:
  """Sets up mock project environment.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

  Returns:
      Tuple[Path, Path, Path]: Environment paths.
  """
  project_root: Path = tmp_path / "project"
  docs_dir: Path = project_root / "docs"
  build_dir: Path = docs_dir / "_build"

  monkeypatch.setattr(build_docs, "PROJECT_ROOT", project_root)
  monkeypatch.setattr(build_docs, "DOCS_DIR", docs_dir)
  monkeypatch.setattr(build_docs, "BUILD_DIR", build_dir)

  project_root.mkdir()
  docs_dir.mkdir()

  return project_root, docs_dir, build_dir


def test_clean(mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
  """Tests cleaning of build directories and copied root files.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  project_root, docs_dir, build_dir = mock_env

  build_dir.mkdir()
  api_dir: Path = docs_dir / "api"
  api_dir.mkdir()
  ops_dir: Path = docs_dir / "ops"
  ops_dir.mkdir()

  dest: Path = docs_dir / "README.md"
  dest.write_text("test")

  monkeypatch.setattr(build_docs, "ROOT_FILES", ("README.md",))

  build_docs.clean()

  assert not build_dir.exists()
  assert not api_dir.exists()
  assert not ops_dir.exists()
  assert not dest.exists()


def test_clean_no_dirs(mock_env: Tuple[Path, Path, Path]) -> None:
  """Tests clean when directories do not exist.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
  """
  build_docs.clean()  # Should not raise


def test_copy_root_files(
  mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Tests copying root files.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
      capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
  """
  project_root, docs_dir, _ = mock_env

  (project_root / "README.md").write_text("test")
  monkeypatch.setattr(build_docs, "ROOT_FILES", ("README.md", "MISSING.md"))

  build_docs.copy_root_files()

  assert (docs_dir / "README.md").exists()
  out, _ = capsys.readouterr()
  assert "Warning: MISSING.md not found" in out


def test_build_wheel_success(mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
  """Tests successful wheel build.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  project_root, _, _ = mock_env
  dist_dir: Path = project_root / "dist"
  dist_dir.mkdir()

  mock_run: mock.MagicMock = mock.Mock()
  monkeypatch.setattr(build_docs.subprocess, "run", mock_run)

  build_docs.build_wheel()

  assert not dist_dir.exists()
  mock_run.assert_called_once_with(["uv", "build", "--wheel"], cwd=project_root, check=True, capture_output=True)


def test_build_wheel_failure(
  mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Tests failed wheel build.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
      capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
  """
  project_root, _, _ = mock_env

  error: subprocess.CalledProcessError = subprocess.CalledProcessError(1, ["uv", "build"], stderr=b"error message")
  mock_run: mock.MagicMock = mock.Mock(side_effect=error)
  monkeypatch.setattr(build_docs.subprocess, "run", mock_run)

  with pytest.raises(SystemExit) as exc:
    build_docs.build_wheel()

  assert exc.value.code == 1
  out, _ = capsys.readouterr()
  assert "Failed to build wheel" in out
  assert "error message" in out


def test_calculate_unique_variants_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
  """Tests variant calculation success.

  Args:
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
      capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
  """

  class MockManager:
    """Mock manager."""

    _reverse_index: list = [1, 2, 3]

  # Create a dummy module to avoid importing the real SemanticsManager which could fail or change
  import sys

  mock_module: Any = type(sys)("ml_switcheroo.semantics.manager")
  mock_module.SemanticsManager = MockManager
  monkeypatch.setitem(sys.modules, "ml_switcheroo.semantics.manager", mock_module)
  monkeypatch.setenv("CI", "false")

  build_docs.calculate_unique_variants()
  out, _ = capsys.readouterr()
  assert "Calculated unique cross-framework variants: 3" in out


def test_calculate_unique_variants_ci_fail(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
  """Tests variant calculation failure in CI.

  Args:
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
      capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
  """

  class MockManager:
    """Mock manager."""

    _reverse_index: list = [1, 2, 3]  # Below 1860

  import sys

  mock_module: Any = type(sys)("ml_switcheroo.semantics.manager")
  mock_module.SemanticsManager = MockManager
  monkeypatch.setitem(sys.modules, "ml_switcheroo.semantics.manager", mock_module)
  monkeypatch.setenv("CI", "true")

  with pytest.raises(SystemExit):
    build_docs.calculate_unique_variants()


def test_calculate_unique_variants_exception(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
  """Tests variant calculation handles exceptions.

  Args:
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
      capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
  """
  import sys

  if "ml_switcheroo.semantics.manager" in sys.modules:
    del sys.modules["ml_switcheroo.semantics.manager"]
  monkeypatch.setitem(sys.modules, "ml_switcheroo.semantics.manager", None)  # type: ignore

  build_docs.calculate_unique_variants()
  out, _ = capsys.readouterr()
  assert "Failed to calculate variants" in out


def test_build_not_all(mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
  """Tests sphinx build (not all).

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  _, docs_dir, build_dir = mock_env

  mock_calc: mock.MagicMock = mock.Mock()
  monkeypatch.setattr(build_docs, "calculate_unique_variants", mock_calc)
  mock_wheel: mock.MagicMock = mock.Mock()
  monkeypatch.setattr(build_docs, "build_wheel", mock_wheel)

  mock_run: mock.MagicMock = mock.Mock()
  mock_run.return_value.returncode = 0
  monkeypatch.setattr(build_docs.subprocess, "run", mock_run)
  monkeypatch.delenv("BUILD_ALL_DOCS", raising=False)

  ret: int = build_docs.build(build_all=False)

  assert ret == 0
  mock_calc.assert_called_once()
  mock_wheel.assert_called_once()

  cmd: list = mock_run.call_args[0][0]
  assert cmd == [
    sys.executable,
    "-m",
    "sphinx",
    "-j",
    "auto",
    "-b",
    "html",
    str(docs_dir),
    str(build_dir / "html"),
    str(docs_dir / "index.md"),
  ]
  env: dict = mock_run.call_args[1]["env"]
  assert env["BUILD_ALL_DOCS"] == "0"


def test_build_all(mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
  """Tests full sphinx build.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  _, docs_dir, build_dir = mock_env

  monkeypatch.setattr(build_docs, "calculate_unique_variants", mock.Mock())
  monkeypatch.setattr(build_docs, "build_wheel", mock.Mock())

  mock_run: mock.MagicMock = mock.Mock()
  mock_run.return_value.returncode = 0
  monkeypatch.setattr(build_docs.subprocess, "run", mock_run)
  monkeypatch.setenv("BUILD_ALL_DOCS", "1")

  build_docs.build(build_all=True)

  cmd: list = mock_run.call_args[0][0]
  assert cmd == [
    sys.executable,
    "-m",
    "sphinx",
    "-j",
    "auto",
    "-b",
    "html",
    str(docs_dir),
    str(build_dir / "html"),
  ]


def test_main_success(
  mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  """Tests main execution block.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
      capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
  """
  project_root, docs_dir, build_dir = mock_env

  mock_clean: mock.MagicMock = mock.Mock()
  monkeypatch.setattr(build_docs, "clean", mock_clean)
  mock_copy: mock.MagicMock = mock.Mock()
  monkeypatch.setattr(build_docs, "copy_root_files", mock_copy)
  mock_build: mock.MagicMock = mock.Mock(return_value=0)
  monkeypatch.setattr(build_docs, "build", mock_build)

  monkeypatch.setattr(build_docs, "ROOT_FILES", ("README.md",))
  (docs_dir / "README.md").write_text("test")  # To test finally block cleanup

  test_args: list = ["build_docs.py", "--build-all"]
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


def test_main_failure(mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
  """Tests main when build fails.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(build_docs, "clean", mock.Mock())
  monkeypatch.setattr(build_docs, "copy_root_files", mock.Mock())
  monkeypatch.setattr(build_docs, "build", mock.Mock(return_value=1))

  test_args: list = ["build_docs.py"]
  with mock.patch.object(sys, "argv", test_args):
    with pytest.raises(SystemExit) as exc:
      build_docs.main()

  assert exc.value.code == 1


def test_main_clean_exception(mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
  """Tests main when clean throws an exception.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  project_root, docs_dir, _ = mock_env

  monkeypatch.setattr(build_docs, "ROOT_FILES", ("README.md",))
  (docs_dir / "README.md").write_text("test")  # To test finally block cleanup

  mock_clean: mock.MagicMock = mock.Mock(side_effect=Exception("Failed to clean"))
  monkeypatch.setattr(build_docs, "clean", mock_clean)

  test_args: list = ["build_docs.py"]
  with mock.patch.object(sys, "argv", test_args):
    with pytest.raises(Exception):
      build_docs.main()

  assert not (docs_dir / "README.md").exists()  # Cleaned up in finally block


def test_main_sys_exit_mocked(mock_env: Tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
  """Tests main execution block with runpy.

  Args:
      mock_env (Tuple[Path, Path, Path]): Mock environment.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  import runpy

  # When runpy is used, it executes the file directly and doesn't use the patched
  # functions in the already imported `build_docs` module.
  # So we mock the underlying components it calls instead, like subprocess and sys.argv
  monkeypatch.setattr(subprocess, "run", mock.Mock(return_value=mock.Mock(returncode=0)))

  test_args: list = ["build_docs.py"]
  with mock.patch.object(sys, "argv", test_args):
    with pytest.raises(SystemExit) as exc:
      runpy.run_path(str(scripts_dir / "build_docs.py"), run_name="__main__")
  assert exc.value.code == 0

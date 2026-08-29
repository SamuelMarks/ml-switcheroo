"""Test suite for the Hooks module."""

import os
from pathlib import Path
from unittest import mock

import ml_switcheroo.sphinx_ext.hooks
from ml_switcheroo.sphinx_ext.hooks import add_static_path, copy_wheel_and_reqs


class MockApp:
  """Docstring."""

  def __init__(self, has_config: bool = True, has_builder: bool = True) -> None:
    """Initializes the MockApp instance."""
    if has_config:
      self.config: mock.MagicMock = mock.Mock()
      self.config.html_static_path = []
    if has_builder:
      self.builder: mock.MagicMock = mock.Mock()
      self.builder.outdir = "mock_outdir"


def test_add_static_path_success(tmp_path: Path) -> None:
  """Adds static path successfully.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_file: Path = tmp_path / "hooks.py"
  mock_file.touch()
  static_dir: Path = tmp_path / "static"
  static_dir.mkdir()
  app: MockApp = MockApp()
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    add_static_path(app)  # type: ignore
  assert str(static_dir.resolve()) in app.config.html_static_path


def test_add_static_path_missing_dir(tmp_path: Path) -> None:
  """Adds static path missing directory.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_file: Path = tmp_path / "hooks.py"
  mock_file.touch()
  app: MockApp = MockApp()
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    add_static_path(app)  # type: ignore
  assert len(app.config.html_static_path) == 0


def test_add_static_path_no_config(tmp_path: Path) -> None:
  """Adds static path no configuration.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  mock_file: Path = tmp_path / "hooks.py"
  mock_file.touch()
  static_dir: Path = tmp_path / "static"
  static_dir.mkdir()
  app: MockApp = MockApp(has_config=False)
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    add_static_path(app)  # type: ignore


def test_copy_wheel_and_reqs_exception() -> None:
  """Verifies the behavior of copy wheel and requirements correctly handling an exception."""
  app: MockApp = MockApp()
  copy_wheel_and_reqs(app, Exception("mock error"))  # type: ignore


def test_copy_wheel_and_reqs_no_builder() -> None:
  """Verifies the behavior of copy wheel and requirements no builder."""
  app: MockApp = MockApp(has_builder=False)
  copy_wheel_and_reqs(app, None)  # type: ignore


def test_copy_wheel_and_reqs_success(tmp_path: Path) -> None:
  """Verifies the behavior of copy wheel and requirements successfully.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  root_dir: Path = tmp_path / "root"
  root_dir.mkdir()
  src_dir: Path = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file: Path = src_dir / "hooks.py"
  mock_file.touch()
  dist_dir: Path = root_dir / "dist"
  dist_dir.mkdir()
  reqs_file: Path = root_dir / "requirements.txt"
  reqs_file.write_text("numpy")
  wheel1: Path = dist_dir / "old-1.0-py3-none-any.whl"
  wheel1.write_text("old")
  os.utime(wheel1, (100, 100))
  wheel2: Path = dist_dir / "new-2.0-py3-none-any.whl"
  wheel2.write_text("new")
  os.utime(wheel2, (200, 200))
  app: MockApp = MockApp()
  outdir: Path = tmp_path / "outdir"
  outdir.mkdir()
  app.builder.outdir = str(outdir)
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)  # type: ignore
  static_dst: Path = outdir / "_static"
  assert static_dst.exists()
  assert (static_dst / "requirements.txt").exists()
  assert (static_dst / "requirements.txt").read_text() == "numpy"
  assert (static_dst / "new-2.0-py3-none-any.whl").exists()
  assert not (static_dst / "old-1.0-py3-none-any.whl").exists()


def test_copy_wheel_and_reqs_newer_existing(tmp_path: Path) -> None:
  """Verifies the behavior of copy wheel and requirements newer existing.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  root_dir: Path = tmp_path / "root"
  root_dir.mkdir()
  src_dir: Path = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file: Path = src_dir / "hooks.py"
  mock_file.touch()
  dist_dir: Path = root_dir / "dist"
  dist_dir.mkdir()
  wheel2: Path = dist_dir / "new-2.0-py3-none-any.whl"
  wheel2.write_text("new")
  os.utime(wheel2, (200, 200))
  app: MockApp = MockApp()
  outdir: Path = tmp_path / "outdir"
  outdir.mkdir()
  app.builder.outdir = str(outdir)
  static_dst: Path = outdir / "_static"
  static_dst.mkdir()
  target_wheel: Path = static_dst / "new-2.0-py3-none-any.whl"
  target_wheel.write_text("newer content")
  os.utime(target_wheel, (300, 300))
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)  # type: ignore
  assert target_wheel.read_text() == "newer content"


def test_copy_wheel_and_reqs_no_reqs_no_wheels(tmp_path: Path) -> None:
  """Verifies the behavior of copy wheel and requirements no requirements no wheels.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  root_dir: Path = tmp_path / "root"
  root_dir.mkdir()
  src_dir: Path = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file: Path = src_dir / "hooks.py"
  mock_file.touch()
  dist_dir: Path = root_dir / "dist"
  dist_dir.mkdir()
  app: MockApp = MockApp()
  outdir: Path = tmp_path / "outdir"
  outdir.mkdir()
  app.builder.outdir = str(outdir)
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)  # type: ignore
  static_dst: Path = outdir / "_static"
  assert static_dst.exists()
  assert not (static_dst / "requirements.txt").exists()
  assert len(list(static_dst.glob("*.whl"))) == 0


def test_copy_wheel_and_reqs_no_dist_dir(tmp_path: Path) -> None:
  """Verifies the behavior of copy wheel and requirements no dist directory.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  root_dir: Path = tmp_path / "root"
  root_dir.mkdir()
  src_dir: Path = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file: Path = src_dir / "hooks.py"
  mock_file.touch()
  app: MockApp = MockApp()
  outdir: Path = tmp_path / "outdir"
  outdir.mkdir()
  app.builder.outdir = str(outdir)
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)  # type: ignore
  static_dst: Path = outdir / "_static"
  assert static_dst.exists()
  assert len(list(static_dst.glob("*.whl"))) == 0

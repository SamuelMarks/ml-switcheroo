"""Test suite for the Hooks module."""

import os
from unittest import mock
from ml_switcheroo.sphinx_ext.hooks import add_static_path, copy_wheel_and_reqs
import ml_switcheroo.sphinx_ext.hooks


class MockApp:
  """Mock App class for testing purposes."""

  def __init__(self, has_config=True, has_builder=True):
    """Initializes the MockApp instance."""
    if has_config:
      self.config = mock.Mock()
      self.config.html_static_path = []
    if has_builder:
      self.builder = mock.Mock()
      self.builder.outdir = "mock_outdir"


def test_add_static_path_success(tmp_path):
  """Adds static path successfully."""
  mock_file = tmp_path / "hooks.py"
  mock_file.touch()
  static_dir = tmp_path / "static"
  static_dir.mkdir()
  app = MockApp()
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    add_static_path(app)
  assert str(static_dir.resolve()) in app.config.html_static_path


def test_add_static_path_missing_dir(tmp_path):
  """Adds static path missing directory."""
  mock_file = tmp_path / "hooks.py"
  mock_file.touch()
  app = MockApp()
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    add_static_path(app)
  assert len(app.config.html_static_path) == 0


def test_add_static_path_no_config(tmp_path):
  """Adds static path no configuration."""
  mock_file = tmp_path / "hooks.py"
  mock_file.touch()
  static_dir = tmp_path / "static"
  static_dir.mkdir()
  app = MockApp(has_config=False)
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    add_static_path(app)


def test_copy_wheel_and_reqs_exception():
  """Verifies the behavior of copy wheel and requirements correctly handling an exception."""
  app = MockApp()
  copy_wheel_and_reqs(app, Exception("mock error"))


def test_copy_wheel_and_reqs_no_builder():
  """Verifies the behavior of copy wheel and requirements no builder."""
  app = MockApp(has_builder=False)
  copy_wheel_and_reqs(app, None)


def test_copy_wheel_and_reqs_success(tmp_path):
  """Verifies the behavior of copy wheel and requirements successfully."""
  root_dir = tmp_path / "root"
  root_dir.mkdir()
  src_dir = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file = src_dir / "hooks.py"
  mock_file.touch()
  dist_dir = root_dir / "dist"
  dist_dir.mkdir()
  reqs_file = root_dir / "requirements.txt"
  reqs_file.write_text("numpy")
  wheel1 = dist_dir / "old-1.0-py3-none-any.whl"
  wheel1.write_text("old")
  os.utime(wheel1, (100, 100))
  wheel2 = dist_dir / "new-2.0-py3-none-any.whl"
  wheel2.write_text("new")
  os.utime(wheel2, (200, 200))
  app = MockApp()
  outdir = tmp_path / "outdir"
  outdir.mkdir()
  app.builder.outdir = str(outdir)
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)
  static_dst = outdir / "_static"
  assert static_dst.exists()
  assert (static_dst / "requirements.txt").exists()
  assert (static_dst / "requirements.txt").read_text() == "numpy"
  assert (static_dst / "new-2.0-py3-none-any.whl").exists()
  assert not (static_dst / "old-1.0-py3-none-any.whl").exists()


def test_copy_wheel_and_reqs_newer_existing(tmp_path):
  """Verifies the behavior of copy wheel and requirements newer existing."""
  root_dir = tmp_path / "root"
  root_dir.mkdir()
  src_dir = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file = src_dir / "hooks.py"
  mock_file.touch()
  dist_dir = root_dir / "dist"
  dist_dir.mkdir()
  wheel2 = dist_dir / "new-2.0-py3-none-any.whl"
  wheel2.write_text("new")
  os.utime(wheel2, (200, 200))
  app = MockApp()
  outdir = tmp_path / "outdir"
  outdir.mkdir()
  app.builder.outdir = str(outdir)
  static_dst = outdir / "_static"
  static_dst.mkdir()
  target_wheel = static_dst / "new-2.0-py3-none-any.whl"
  target_wheel.write_text("newer content")
  os.utime(target_wheel, (300, 300))
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)
  assert target_wheel.read_text() == "newer content"


def test_copy_wheel_and_reqs_no_reqs_no_wheels(tmp_path):
  """Verifies the behavior of copy wheel and requirements no requirements no wheels."""
  root_dir = tmp_path / "root"
  root_dir.mkdir()
  src_dir = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file = src_dir / "hooks.py"
  mock_file.touch()
  dist_dir = root_dir / "dist"
  dist_dir.mkdir()
  app = MockApp()
  outdir = tmp_path / "outdir"
  outdir.mkdir()
  app.builder.outdir = str(outdir)
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)
  static_dst = outdir / "_static"
  assert static_dst.exists()
  assert not (static_dst / "requirements.txt").exists()
  assert len(list(static_dst.glob("*.whl"))) == 0


def test_copy_wheel_and_reqs_no_dist_dir(tmp_path):
  """Verifies the behavior of copy wheel and requirements no dist directory."""
  root_dir = tmp_path / "root"
  root_dir.mkdir()
  src_dir = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file = src_dir / "hooks.py"
  mock_file.touch()
  app = MockApp()
  outdir = tmp_path / "outdir"
  outdir.mkdir()
  app.builder.outdir = str(outdir)
  with mock.patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)
  static_dst = outdir / "_static"
  assert static_dst.exists()
  assert len(list(static_dst.glob("*.whl"))) == 0

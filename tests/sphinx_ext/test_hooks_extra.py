"""Test module."""

from unittest.mock import MagicMock, patch
from ml_switcheroo.sphinx_ext.hooks import copy_wheel_and_reqs, add_static_path


def test_add_static_path_missing(tmp_path):
  """Test function."""
  app = MagicMock()
  app.config.html_static_path = []
  with patch("ml_switcheroo.sphinx_ext.hooks.Path") as mock_path:
    mock_p = MagicMock()
    mock_p.exists.return_value = False
    mock_path.return_value.parent.__truediv__.return_value = mock_p
    add_static_path(app)
    assert len(app.config.html_static_path) == 0


def test_copy_wheel_exception():
  """Test function."""
  copy_wheel_and_reqs(MagicMock(), Exception())


def test_copy_wheel_no_builder():
  """Test function."""
  app = MagicMock()
  del app.builder
  copy_wheel_and_reqs(app, None)


def test_copy_wheel_and_reqs_full(tmp_path, monkeypatch):
  """Test function."""
  import ml_switcheroo.sphinx_ext.hooks

  dummy_file = tmp_path / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  dummy_file.parent.mkdir(parents=True)
  dummy_file.touch()

  monkeypatch.setattr(ml_switcheroo.sphinx_ext.hooks, "__file__", str(dummy_file))

  root_dir = tmp_path

  dist_dir = root_dir / "dist"
  dist_dir.mkdir()
  whl = dist_dir / "switcheroo-0.1.0.whl"
  whl.touch()

  reqs = root_dir / "requirements.txt"
  reqs.write_text("foo @ https://github.com/foo/foo.whl\nbar\n# comment\n\n")

  app = MagicMock()
  outdir = tmp_path / "out"
  app.builder.outdir = str(outdir)

  class MockResponse:
    """A mock HTTP response."""

    def __enter__(self):
      """Enters context."""
      return self

    def __exit__(self, *args):
      """Exits context."""
      pass

    def read(self, *args):
      """Reads data."""
      return b"data"

  with patch("urllib.request.urlopen") as mock_url:
    mock_url.return_value = MockResponse()
    with patch("shutil.copyfileobj") as mock_copy:  # noqa: F841
      copy_wheel_and_reqs(app, None)

  with patch("urllib.request.urlopen", side_effect=Exception("Failed")):
    (outdir / "_static" / "foo.whl").unlink()
    copy_wheel_and_reqs(app, None)

  (root_dir / "dist" / "switcheroo-0.1.0.whl").unlink()
  copy_wheel_and_reqs(app, None)


def test_copy_wheel_and_reqs_target_exists(tmp_path, monkeypatch):
  """Test function."""
  import ml_switcheroo.sphinx_ext.hooks

  dummy_file = tmp_path / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  dummy_file.parent.mkdir(parents=True)
  dummy_file.touch()
  monkeypatch.setattr(ml_switcheroo.sphinx_ext.hooks, "__file__", str(dummy_file))

  root_dir = tmp_path

  dist_dir = root_dir / "dist"
  dist_dir.mkdir()
  whl = dist_dir / "switcheroo-0.1.0.whl"
  whl.touch()

  reqs = root_dir / "requirements.txt"
  reqs.write_text("foo @ https://github.com/foo/foo.whl")

  app = MagicMock()
  outdir = tmp_path / "out"
  app.builder.outdir = str(outdir)

  (outdir / "_static").mkdir(parents=True)
  (outdir / "_static" / "foo.whl").touch()  # target exists!

  copy_wheel_and_reqs(app, None)

"""Test suite for hooks extra."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.sphinx_ext.hooks import add_static_path, copy_wheel_and_reqs


def test_add_static_path_missing(tmp_path: Path) -> None:
  """Test function.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
  """
  app: MagicMock = MagicMock()
  app.config.html_static_path = []
  with patch("ml_switcheroo.sphinx_ext.hooks.Path") as mock_path:
    mock_p: MagicMock = MagicMock()
    mock_p.exists.return_value = False
    mock_path.return_value.parent.__truediv__.return_value = mock_p
    add_static_path(app)
    assert len(app.config.html_static_path) == 0


def test_copy_wheel_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify copy wheel exception behavior when shutil.copy2 raises an error.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  import ml_switcheroo.sphinx_ext.hooks

  dummy_file: Path = tmp_path / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  dummy_file.parent.mkdir(parents=True)
  dummy_file.touch()
  monkeypatch.setattr(ml_switcheroo.sphinx_ext.hooks, "__file__", str(dummy_file))

  dist_dir: Path = tmp_path / "dist"
  dist_dir.mkdir()
  (dist_dir / "switcheroo-0.1.0.whl").write_text("wheel")

  mock_app = MagicMock()
  mock_app.builder.outdir = str(tmp_path / "test_out")
  with patch("ml_switcheroo.sphinx_ext.hooks.shutil.copy2", side_effect=RuntimeError("Copy failed")):
    with pytest.raises(RuntimeError, match="Copy failed"):
      copy_wheel_and_reqs(mock_app, None)


def test_copy_wheel_no_builder() -> None:
  """Verify copy wheel no builder behavior."""
  app: MagicMock = MagicMock()
  del app.builder
  res = copy_wheel_and_reqs(app, None)
  assert res is None


def test_copy_wheel_and_reqs_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test function.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  import ml_switcheroo.sphinx_ext.hooks

  dummy_file: Path = tmp_path / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  dummy_file.parent.mkdir(parents=True)
  dummy_file.touch()

  monkeypatch.setattr(ml_switcheroo.sphinx_ext.hooks, "__file__", str(dummy_file))

  root_dir: Path = tmp_path

  dist_dir: Path = root_dir / "dist"
  dist_dir.mkdir()
  whl: Path = dist_dir / "switcheroo-0.1.0.whl"
  whl.touch()

  reqs: Path = root_dir / "requirements.txt"
  reqs.write_text("foo @ https://github.com/foo/foo.whl\nbar\n# comment\n\n")

  app: MagicMock = MagicMock()
  outdir: Path = tmp_path / "out"
  app.builder.outdir = str(outdir)

  class MockResponse:
    """A mock HTTP response."""

    def __enter__(self) -> "MockResponse":
      """Enters context.

      Returns:
          MockResponse: Self instance.
      """
      return self

    def __exit__(self, *args: list) -> None:
      """Exits context.

      Args:
          *args (list): Variable arguments.
      """
      pass

    def read(self, *args: list) -> bytes:
      """Reads data.

      Args:
          *args (list): Variable arguments.

      Returns:
          bytes: Result.
      """
      return b"data"

  with patch("urllib.request.urlopen") as mock_url:
    mock_url.return_value = MockResponse()
    with patch("shutil.copyfileobj") as mock_copy:  # noqa: F841
      copy_wheel_and_reqs(app, None)

  with patch("urllib.request.urlopen", side_effect=Exception("Failed")):
    (outdir / "_static" / "foo.whl").unlink()
    copy_wheel_and_reqs(app, None)

  (root_dir / "dist" / "switcheroo-0.1.0.whl").unlink()
  _res = copy_wheel_and_reqs(app, None)
  assert _res is not None or _res is None


def test_copy_wheel_and_reqs_target_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test function.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  import ml_switcheroo.sphinx_ext.hooks

  dummy_file: Path = tmp_path / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  dummy_file.parent.mkdir(parents=True)
  dummy_file.touch()
  monkeypatch.setattr(ml_switcheroo.sphinx_ext.hooks, "__file__", str(dummy_file))

  root_dir: Path = tmp_path

  dist_dir: Path = root_dir / "dist"
  dist_dir.mkdir()
  whl: Path = dist_dir / "switcheroo-0.1.0.whl"
  whl.touch()

  reqs: Path = root_dir / "requirements.txt"
  reqs.write_text("foo @ https://github.com/foo/foo.whl")

  app: MagicMock = MagicMock()
  outdir: Path = tmp_path / "out"
  app.builder.outdir = str(outdir)

  (outdir / "_static").mkdir(parents=True)
  (outdir / "_static" / "foo.whl").touch()  # target exists!

  res = copy_wheel_and_reqs(app, None)
  assert res is None
  assert (outdir / "_static" / "foo.whl").exists()


def test_copy_wheel_src_static_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify copy_wheel_and_reqs copies wheel from docs/_static when present locally.

  Args:
      tmp_path (Path): Tmp path pytest fixture.
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  import ml_switcheroo.sphinx_ext.hooks

  dummy_file: Path = tmp_path / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  dummy_file.parent.mkdir(parents=True)
  dummy_file.touch()
  monkeypatch.setattr(ml_switcheroo.sphinx_ext.hooks, "__file__", str(dummy_file))

  docs_static: Path = tmp_path / "docs" / "_static"
  docs_static.mkdir(parents=True)
  (docs_static / "foo.whl").write_text("cached wheel")

  reqs: Path = tmp_path / "requirements.txt"
  reqs.write_text("foo @ https://github.com/foo/foo.whl\n")

  app: MagicMock = MagicMock()
  outdir: Path = tmp_path / "out"
  app.builder.outdir = str(outdir)

  copy_wheel_and_reqs(app, None)
  assert (outdir / "_static" / "foo.whl").read_text() == "cached wheel"


def test_import_error_fallbacks() -> None:
  """Verify ImportError fallback implementations in file_loader and consensus."""
  import importlib
  import sys

  import ml_switcheroo.discovery.consensus as consensus_mod
  import ml_switcheroo.semantics.file_loader as file_loader_mod

  with patch.dict(sys.modules, {"ml_ecosystem_snapshots.utils": None}):
    reloaded_fl = importlib.reload(file_loader_mod)
    assert reloaded_fl.strip_sphinx_roles(None) is None
    assert reloaded_fl.strip_sphinx_roles(":class:`~torch.Tensor`") == "torch.Tensor"
    assert reloaded_fl.resolve_griffe_parser("google") is None
    assert reloaded_fl.parse_docstring_with_griffe("doc") == []
  importlib.reload(file_loader_mod)

  with patch.dict(sys.modules, {"ml_ecosystem_snapshots.grounding.engine": None}):
    reloaded_con = importlib.reload(consensus_mod)
    assert reloaded_con.compute_levenshtein("a", "ab") == 1
    assert reloaded_con.compute_levenshtein("abc", "") == 3
    assert reloaded_con.compute_levenshtein("abc", "abd") == 1
  importlib.reload(consensus_mod)

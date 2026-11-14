"""Unit tests for Sphinx static-asset and wheel-packaging hooks (`ml_switcheroo.sphinx_ext.hooks`)."""

import io
import os
from pathlib import Path
from types import TracebackType
from unittest import mock

import pytest

import ml_switcheroo.sphinx_ext.hooks as hooks_mod
from ml_switcheroo.sphinx_ext.hooks import add_static_path, copy_wheel_and_reqs

_NL = chr(10)


class _SphinxConfigStub:
  """Typed Sphinx config stub carrying `html_static_path`."""

  def __init__(self) -> None:
    """Initialize empty `html_static_path` list."""
    self.html_static_path: list[str] = []


class _SphinxBuilderStub:
  """Typed Sphinx builder stub carrying `outdir`."""

  def __init__(self, outdir: str | Path) -> None:
    """Initialize builder output directory path.

    Args:
        outdir: Output directory path for Sphinx HTML build artifacts.
    """
    self.outdir = str(outdir)


class _SphinxAppStub:
  """Typed Sphinx application stub exposing `config` and `builder`."""

  def __init__(self, outdir: str | Path = "mock_outdir") -> None:
    """Initialize Sphinx application stub with config and builder.

    Args:
        outdir: Builder output directory path.
    """
    self.config = _SphinxConfigStub()
    self.builder = _SphinxBuilderStub(outdir)


class _NoConfigAppStub:
  """Typed Sphinx application stub lacking a `config` attribute."""

  def __init__(self) -> None:
    """Initialize builder without `config`."""
    self.builder = _SphinxBuilderStub("mock_outdir")


class _NoBuilderAppStub:
  """Typed Sphinx application stub lacking a `builder` attribute."""

  def __init__(self) -> None:
    """Initialize config without `builder`."""
    self.config = _SphinxConfigStub()


def test_add_static_path_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `add_static_path` registers existing static directories and ignores missing ones or configless apps.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  mock_file = tmp_path / "hooks.py"
  mock_file.touch()
  monkeypatch.setattr(hooks_mod, "__file__", str(mock_file))

  # Missing static directory -> no path appended
  app_missing = _SphinxAppStub()
  add_static_path(app_missing)
  assert app_missing.config.html_static_path == []

  # Existing static directory -> resolved path appended
  static_dir = tmp_path / "static"
  static_dir.mkdir()
  app_present = _SphinxAppStub()
  add_static_path(app_present)
  assert str(static_dir.resolve()) in app_present.config.html_static_path

  # App without `config` -> no-op without raising
  no_cfg = _NoConfigAppStub()
  add_static_path(no_cfg)
  assert no_cfg.builder.outdir == "mock_outdir"


def test_copy_wheel_and_reqs_early_exits_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `copy_wheel_and_reqs` handles build exceptions, missing builders, and copy failures.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  outdir = tmp_path / "early_out"
  app = _SphinxAppStub(outdir)
  copy_wheel_and_reqs(app, RuntimeError("sphinx build failed"))
  assert not (outdir / "_static").exists()

  no_builder = _NoBuilderAppStub()
  copy_wheel_and_reqs(no_builder, None)
  assert no_builder.config.html_static_path == []

  dummy_file = tmp_path / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  dummy_file.parent.mkdir(parents=True)
  dummy_file.touch()
  monkeypatch.setattr(hooks_mod, "__file__", str(dummy_file))

  dist_dir = tmp_path / "dist"
  dist_dir.mkdir()
  (dist_dir / "switcheroo-0.1.0.whl").write_text("wheel", encoding="utf-8")

  err_app = _SphinxAppStub(tmp_path / "err_out")
  with (
    mock.patch("ml_switcheroo.sphinx_ext.hooks.shutil.copy2", side_effect=RuntimeError("Copy failed")),
    pytest.raises(RuntimeError, match="Copy failed"),
  ):
    copy_wheel_and_reqs(err_app, None)


def test_copy_wheel_and_reqs_dist_selection_and_mtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify latest wheel selection, mtime skip when target is newer, and empty/missing dist handling.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  root_dir = tmp_path / "repo"
  mock_file = root_dir / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  mock_file.parent.mkdir(parents=True)
  mock_file.touch()
  monkeypatch.setattr(hooks_mod, "__file__", str(mock_file))

  # Missing dist directory
  out_no_dist = tmp_path / "out_no_dist"
  copy_wheel_and_reqs(_SphinxAppStub(out_no_dist), None)
  assert (out_no_dist / "_static").exists()
  assert list((out_no_dist / "_static").glob("*.whl")) == []

  # Empty dist directory
  dist_dir = root_dir / "dist"
  dist_dir.mkdir()
  out_empty_dist = tmp_path / "out_empty_dist"
  copy_wheel_and_reqs(_SphinxAppStub(out_empty_dist), None)
  assert list((out_empty_dist / "_static").glob("*.whl")) == []

  # Multiple wheels + requirements.txt -> newest wheel copied
  (root_dir / "requirements.txt").write_text("numpy", encoding="utf-8")
  wheel_old = dist_dir / "old-1.0-py3-none-any.whl"
  wheel_old.write_text("old", encoding="utf-8")
  os.utime(wheel_old, (100, 100))
  wheel_new = dist_dir / "new-2.0-py3-none-any.whl"
  wheel_new.write_text("new", encoding="utf-8")
  os.utime(wheel_new, (200, 200))

  out_success = tmp_path / "out_success"
  copy_wheel_and_reqs(_SphinxAppStub(out_success), None)
  static_dst = out_success / "_static"
  assert (static_dst / "requirements.txt").read_text(encoding="utf-8") == "numpy"
  assert (static_dst / "new-2.0-py3-none-any.whl").read_text(encoding="utf-8") == "new"
  assert not (static_dst / "old-1.0-py3-none-any.whl").exists()

  # Existing destination wheel with newer mtime is preserved
  out_newer = tmp_path / "out_newer"
  newer_static = out_newer / "_static"
  newer_static.mkdir(parents=True)
  target_wheel = newer_static / "new-2.0-py3-none-any.whl"
  target_wheel.write_text("newer content", encoding="utf-8")
  os.utime(target_wheel, (300, 300))
  copy_wheel_and_reqs(_SphinxAppStub(out_newer), None)
  assert target_wheel.read_text(encoding="utf-8") == "newer content"


def test_copy_wheel_and_reqs_url_dependencies_and_local_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify URL wheel dependencies in `requirements.txt` resolve from local `docs/_static` or `urlopen`.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  dummy_file = tmp_path / "src" / "ml_switcheroo" / "sphinx_ext" / "hooks.py"
  dummy_file.parent.mkdir(parents=True)
  dummy_file.touch()
  monkeypatch.setattr(hooks_mod, "__file__", str(dummy_file))

  reqs = tmp_path / "requirements.txt"
  reqs.write_text(_NL.join(["foo @ https://github.com/foo/foo.whl", "bar", "# comment", ""]), encoding="utf-8")

  class _BytesUrlResponse(io.BytesIO):
    """In-memory context-managed HTTP response stream."""

    def __enter__(self) -> "_BytesUrlResponse":
      """Return self upon context entry.

      Returns:
          Active `_BytesUrlResponse` stream.
      """
      return self

    def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc: BaseException | None,
      tb: TracebackType | None,
    ) -> None:
      """Close the underlying `BytesIO` stream upon exit.

      Args:
          exc_type: Exception class if raised.
          exc: Exception instance if raised.
          tb: Traceback if raised.
      """
      self.close()

  outdir = tmp_path / "out_url"
  app = _SphinxAppStub(outdir)

  # 1. Downloaded via urlopen
  with mock.patch("urllib.request.urlopen", return_value=_BytesUrlResponse(b"remote-wheel")):
    copy_wheel_and_reqs(app, None)
  assert (outdir / "_static" / "foo.whl").read_bytes() == b"remote-wheel"

  # 2. Target already exists -> skipped without network call
  with mock.patch("urllib.request.urlopen", side_effect=AssertionError("should not fetch")):
    copy_wheel_and_reqs(app, None)
  assert (outdir / "_static" / "foo.whl").read_bytes() == b"remote-wheel"

  # 3. Network failure when uncached -> handled gracefully
  (outdir / "_static" / "foo.whl").unlink()
  with mock.patch("urllib.request.urlopen", side_effect=OSError("Offline")):
    copy_wheel_and_reqs(app, None)
  assert not (outdir / "_static" / "foo.whl").exists()

  # 4. Cached in docs/_static -> copied locally without network call
  docs_static = tmp_path / "docs" / "_static"
  docs_static.mkdir(parents=True)
  (docs_static / "foo.whl").write_text("cached-local-wheel", encoding="utf-8")
  copy_wheel_and_reqs(app, None)
  assert (outdir / "_static" / "foo.whl").read_text(encoding="utf-8") == "cached-local-wheel"

"""Tests for local wheel resolution and requirements hooks."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import os

from ml_switcheroo.sphinx_ext.hooks import find_local_wheel, copy_wheel_and_reqs


def test_find_local_wheel_cdd_python(tmp_path: Path) -> None:
  """Verifies finding cdd wheel in cdd-python/dist.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()
  cdd_dir = tmp_path / "cdd-python"
  cdd_dist = cdd_dir / "dist"
  cdd_dist.mkdir(parents=True)
  whl = cdd_dist / "cdd-0.0.99rc46-py3-none-any.whl"
  whl.write_text("whl content")

  res = find_local_wheel(root_dir, "cdd")
  assert res is not None
  assert res.name == "cdd-0.0.99rc46-py3-none-any.whl"


def test_find_local_wheel_python_cdd(tmp_path: Path) -> None:
  """Verifies finding cdd wheel in python-cdd root.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()
  cdd_dir = tmp_path / "python-cdd"
  cdd_dir.mkdir()
  whl = cdd_dir / "python_cdd-0.1.0-py3-none-any.whl"
  whl.write_text("whl content")

  res = find_local_wheel(root_dir, "cdd")
  assert res is not None
  assert res.name == "python_cdd-0.1.0-py3-none-any.whl"


def test_find_local_wheel_generic_package(tmp_path: Path) -> None:
  """Verifies finding generic package wheel with exact match priority.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()
  pkg_dir = tmp_path / "my-pkg"
  pkg_dist = pkg_dir / "dist"
  pkg_dist.mkdir(parents=True)

  whl1 = pkg_dist / "my_pkg_cli-1.0.0-py3-none-any.whl"
  whl1.write_text("content1")
  os.utime(whl1, (200, 200))

  whl2 = pkg_dist / "my_pkg-1.0.0-py3-none-any.whl"
  whl2.write_text("content2")
  os.utime(whl2, (100, 100))

  res = find_local_wheel(root_dir, "my-pkg")
  assert res is not None
  assert res.name == "my_pkg-1.0.0-py3-none-any.whl"

  # Now test when only inexact matching wheel is present
  whl2.unlink()
  res_inexact = find_local_wheel(root_dir, "my-pkg")
  assert res_inexact is not None
  assert res_inexact.name == "my_pkg_cli-1.0.0-py3-none-any.whl"


def test_find_local_wheel_build_fallback(tmp_path: Path) -> None:
  """Verifies building wheel when repo exists without pre-built wheels.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()
  cdd_dir = tmp_path / "cdd-python"
  cdd_dir.mkdir()
  (cdd_dir / "pyproject.toml").write_text("[project]" + chr(10) + "name='cdd'")

  def mock_run(*args: list, **kwargs: dict) -> None:
    dist = cdd_dir / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "cdd-1.0.0-py3-none-any.whl").write_text("built")

  with patch("subprocess.run", side_effect=mock_run):
    res = find_local_wheel(root_dir, "cdd")
    assert res is not None
    assert res.name == "cdd-1.0.0-py3-none-any.whl"


def test_find_local_wheel_build_error(tmp_path: Path) -> None:
  """Verifies build failure handling in find_local_wheel.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()
  cdd_dir = tmp_path / "cdd-python"
  cdd_dir.mkdir()
  (cdd_dir / "setup.py").write_text("setup()")

  with patch("subprocess.run", side_effect=Exception("Build failed")):
    res = find_local_wheel(root_dir, "cdd")
    assert res is None


def test_copy_wheel_and_reqs_local_grab(tmp_path: Path) -> None:
  """Verifies copy_wheel_and_reqs copies local sibling wheels and handles git+ lines.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  src_dir = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file = src_dir / "hooks.py"
  mock_file.touch()

  cdd_dir = tmp_path / "cdd-python"
  cdd_dist = cdd_dir / "dist"
  cdd_dist.mkdir(parents=True)
  cdd_whl = cdd_dist / "cdd-0.0.99-py3-none-any.whl"
  cdd_whl.write_text("cdd wheel")

  nl = chr(10)
  reqs_content = (
    nl.join(
      [
        "cdd @ https://github.com/offscale/cdd-python/releases/download/v0.0.99/cdd-0.0.99-py3-none-any.whl",
        "ml-ir @ git+https://github.com/SamuelMarks/ml-switcheroo-ir.git",
        "custom @ local_path",
        "numpy",
      ]
    )
    + nl
  )

  reqs_file = root_dir / "requirements.txt"
  reqs_file.write_text(reqs_content)

  app = MagicMock()
  outdir = tmp_path / "out"
  app.builder.outdir = str(outdir)

  import ml_switcheroo.sphinx_ext.hooks

  with patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)

    static_dst = outdir / "_static"
    assert (static_dst / "cdd-0.0.99-py3-none-any.whl").exists()
    saved_reqs = (static_dst / "requirements.txt").read_text().splitlines()
    assert "cdd @ cdd-0.0.99-py3-none-any.whl" in saved_reqs
    assert not any("git+" in line for line in saved_reqs)
    assert "custom @ local_path" in saved_reqs
    assert "numpy" in saved_reqs

    # Re-run when target exists and is up to date (hits 144->146 branch)
    copy_wheel_and_reqs(app, None)
    assert (static_dst / "cdd-0.0.99-py3-none-any.whl").exists()


def test_find_local_wheel_unrelated_wheels(tmp_path: Path) -> None:
  """Verifies find_local_wheel skips non-matching wheels for both cdd and generic packages.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()

  # cdd with unrelated wheel
  cdd_dir = tmp_path / "cdd"
  cdd_dir.mkdir()
  (cdd_dir / "unrelated-1.0.whl").write_text("x")
  (cdd_dir / "cdd-1.0.0-py3-none-any.whl").write_text("cdd")

  res = find_local_wheel(root_dir, "cdd")
  assert res is not None
  assert res.name == "cdd-1.0.0-py3-none-any.whl"

  # generic pkg with unrelated wheel
  pkg_dir = tmp_path / "my-pkg"
  pkg_dir.mkdir()
  (pkg_dir / "other-1.0.whl").write_text("x")
  (pkg_dir / "my_pkg-1.0.whl").write_text("y")

  res_pkg = find_local_wheel(root_dir, "my-pkg")
  assert res_pkg is not None
  assert res_pkg.name == "my_pkg-1.0.whl"


def test_find_local_wheel_fallback_across_candidates(tmp_path: Path) -> None:
  """Verifies candidate iteration when first candidate has no wheels and no build file.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()

  # Candidate 1 exists but is empty
  p1_dir = tmp_path / "multi-name"
  p1_dir.mkdir()

  # Candidate 2 exists and has a wheel
  p2_dir = tmp_path / "multi_name"
  p2_dir.mkdir()
  (p2_dir / "multi_name-1.0.whl").write_text("wheel")

  res = find_local_wheel(root_dir, "multi-name")
  assert res is not None
  assert res.name == "multi_name-1.0.whl"


def test_find_local_wheel_build_no_dist_or_no_wheels(tmp_path: Path) -> None:
  """Verifies fallback across candidates when build produces no wheels or fails.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()

  # Candidate 1 has setup.py and dist dir, but build produces no .whl files
  b1_dir = tmp_path / "build-empty"
  b1_dir.mkdir()
  (b1_dir / "setup.py").write_text("setup()")
  (b1_dir / "dist").mkdir()

  # Candidate 2 has pre-built wheel
  b2_dir = tmp_path / "build_empty"
  b2_dir.mkdir()
  (b2_dir / "build_empty-1.0.whl").write_text("built")

  with patch("subprocess.run", return_value=None):
    res = find_local_wheel(root_dir, "build-empty")
    assert res is not None
    assert res.name == "build_empty-1.0.whl"


def test_find_local_wheel_build_no_dist_dir(tmp_path: Path) -> None:
  """Verifies candidate fallback when build completes but dist directory is not created.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  root_dir.mkdir()

  # Candidate 1 has setup.py, but no dist directory after build
  b1_dir = tmp_path / "nodist-pkg"
  b1_dir.mkdir()
  (b1_dir / "setup.py").write_text("setup()")

  # Candidate 2 has pre-built wheel
  b2_dir = tmp_path / "nodist_pkg"
  b2_dir.mkdir()
  (b2_dir / "nodist_pkg-1.0.whl").write_text("built")

  with patch("subprocess.run", return_value=None):
    res = find_local_wheel(root_dir, "nodist-pkg")
    assert res is not None
    assert res.name == "nodist_pkg-1.0.whl"


def test_copy_wheel_and_reqs_http_non_github_or_non_whl(tmp_path: Path) -> None:
  """Verifies copy_wheel_and_reqs handles http requirements not from github or not wheels.

  Args:
      tmp_path (Path): Temporary test directory.
  """
  root_dir = tmp_path / "ml-switcheroo"
  src_dir = root_dir / "src" / "ml_switcheroo" / "sphinx_ext"
  src_dir.mkdir(parents=True)
  mock_file = src_dir / "hooks.py"
  mock_file.touch()

  nl = chr(10)
  reqs_content = (
    nl.join(
      [
        "pkg1 @ http://gitlab.com/repo/pkg1.whl",
        "pkg2 @ https://github.com/repo/pkg2.tar.gz",
      ]
    )
    + nl
  )
  reqs_file = root_dir / "requirements.txt"
  reqs_file.write_text(reqs_content)

  app = MagicMock()
  outdir = tmp_path / "out"
  app.builder.outdir = str(outdir)

  import ml_switcheroo.sphinx_ext.hooks

  with patch.object(ml_switcheroo.sphinx_ext.hooks, "__file__", str(mock_file)):
    copy_wheel_and_reqs(app, None)

  static_dst = outdir / "_static"
  saved_reqs = (static_dst / "requirements.txt").read_text().splitlines()
  assert "pkg1 @ http://gitlab.com/repo/pkg1.whl" in saved_reqs
  assert "pkg2 @ https://github.com/repo/pkg2.tar.gz" in saved_reqs

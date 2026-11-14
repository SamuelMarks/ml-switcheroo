"""Sphinx Build Hooks.

Provides lifecycle hooks to:
1. Register static asset paths (CSS/JS).
2. Copy the distribution wheel to the static directory for the WASM demo.
"""

import typing

if typing.TYPE_CHECKING:
  pass

import json
import logging
import os
import shutil
from pathlib import Path

ON_DEVICE_WHEEL_PACKAGES: tuple[str, ...] = (
  "ml_switcheroo_ir",
  "ml_ecosystem_snapshots",
  "cdd",
  "ml_switcheroo_compiler",
  "zero_jax",
  "zero_chex",
  "zero_optax",
  "zero_grain",
  "zero_orbax",
  "zero_flax",
  "zero_pytorch",
  "zero_keras",
  "zero_tensorflow",
  "zero_mlx",
  "zero_pax",
  "zero_zoo",
)


class _SphinxConfigProtocol(typing.Protocol):
  """Protocol for Sphinx config object exposing `html_static_path`.

  Attributes:
      html_static_path: The html static path (`list[str]`).
  """

  html_static_path: list[str]


class _SphinxBuilderProtocol(typing.Protocol):
  """Protocol for Sphinx builder object exposing `outdir`.

  Attributes:
      outdir: The outdir (`str | Path`).
  """

  outdir: str | Path


@typing.runtime_checkable
class SphinxAppConfigProtocol(typing.Protocol):
  """Protocol for Sphinx application exposing `config`.

  Attributes:
      config: The config (`_SphinxConfigProtocol`).
  """

  config: _SphinxConfigProtocol


@typing.runtime_checkable
class SphinxAppBuilderProtocol(typing.Protocol):
  """Protocol for Sphinx application exposing `builder`.

  Attributes:
      builder: The builder (`_SphinxBuilderProtocol`).
  """

  builder: _SphinxBuilderProtocol


@typing.runtime_checkable
class SphinxBareAppProtocol(typing.Protocol):
  """Minimal structural protocol for stub or mock Sphinx application instances."""


SphinxAppUnion: typing.TypeAlias = SphinxAppConfigProtocol | SphinxAppBuilderProtocol | SphinxBareAppProtocol | None


def _has_sphinx_config(app: SphinxAppUnion) -> typing.TypeGuard[SphinxAppConfigProtocol]:
  """Check if `app` exposes a `config` attribute.

  Args:
      app: Candidate Sphinx application.

  Returns:
      True if `app.config` exists.
  """
  return isinstance(app, SphinxAppConfigProtocol)


def _has_sphinx_builder(app: SphinxAppUnion) -> typing.TypeGuard[SphinxAppBuilderProtocol]:
  """Check if `app` exposes a `builder` attribute.

  Args:
      app: Candidate Sphinx application.

  Returns:
      True if `app.builder` exists.
  """
  return isinstance(app, SphinxAppBuilderProtocol)


def add_static_path(app: SphinxAppUnion) -> None:
  """Add the extension's static directory to HTML build configuration.

  Connected to 'builder-inited' event.

  Args:
      app: The Sphinx application object.
  """
  static_path = Path(__file__).parent / "static"
  if static_path.exists() and _has_sphinx_config(app):
    app.config.html_static_path.append(str(static_path.resolve()))


def _candidate_repo_names(pkg: str, clean_pkg: str) -> list[str]:
  """Return ordered sibling directory name candidates for ``pkg``.

  Args:
      pkg: Raw package name.
      clean_pkg: Lowercased stripped package name.

  Returns:
      List of candidate sibling folder names.
  """
  if clean_pkg in ("cdd", "python-cdd", "cdd-python"):
    return ["python-cdd", "cdd-python", "cdd"]
  if clean_pkg in ("zero_pytorch", "zero-pytorch", "zero_torch", "zero-torch"):
    return ["zero-pytorch", "zero_pytorch", "zero-torch", "zero_torch"]
  return [pkg, pkg.replace("-", "_"), pkg.replace("_", "-")]


def _select_matching_wheel(wheels: list[Path], clean_pkg: str) -> Path | None:
  """Select the newest exact or prefix-matching wheel file from ``wheels``.

  Args:
      wheels: Candidate ``.whl`` paths found in a sibling repository.
      clean_pkg: Lowercased stripped package name.

  Returns:
      Newest matching ``Path``, or ``None`` if no wheel matches ``clean_pkg``.
  """
  matching: list[Path] = []
  exact_matching: list[Path] = []
  prefix = clean_pkg.replace("-", "_")

  for whl in wheels:
    whl_name = whl.name.lower()
    norm_whl_name = whl_name.replace("-", "_")
    if clean_pkg in ("cdd", "python-cdd", "cdd-python"):
      if whl_name.startswith(("cdd-", "cdd_", "python_cdd")):
        matching.append(whl)
        exact_matching.append(whl)
    elif clean_pkg in ("zero_pytorch", "zero-pytorch", "zero_torch", "zero-torch"):
      if norm_whl_name.startswith(("zero_torch_", "zero_pytorch_")):
        matching.append(whl)
        exact_matching.append(whl)
    elif prefix in norm_whl_name:
      matching.append(whl)
      if whl_name.startswith((f"{prefix}-", f"{clean_pkg}-")):
        exact_matching.append(whl)

  if exact_matching:
    return sorted(exact_matching, key=os.path.getmtime)[-1]
  if matching:
    return sorted(matching, key=os.path.getmtime)[-1]
  return None


def _try_build_candidate_wheel(candidate_dir: Path, dist_dir: Path) -> Path | None:
  """Attempt to build a wheel via ``uv build --wheel`` inside ``candidate_dir`` when project metadata exists.

  Args:
      candidate_dir: Sibling repository directory.
      dist_dir: Expected ``dist/`` subdirectory inside ``candidate_dir``.

  Returns:
      Newest built wheel ``Path`` if build succeeded, otherwise ``None``.
  """
  if not ((candidate_dir / "pyproject.toml").exists() or (candidate_dir / "setup.py").exists()):
    return None
  import subprocess

  try:
    subprocess.run(["uv", "build", "--wheel"], cwd=candidate_dir, check=True, capture_output=True)
    if dist_dir.is_dir():
      built = list(dist_dir.glob("*.whl"))
      if built:
        return sorted(built, key=os.path.getmtime)[-1]
  except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
    logging.getLogger(__name__).debug("Wheel build candidate failed: %s", exc)
  return None


def find_local_wheel(root_dir: Path, pkg: str) -> Path | None:
  """Locate a local wheel for a given package in sibling repository directories.

  Checks sibling directories for existing .whl distribution artifacts (or builds
  them if the repository exists but no wheels are found).

  Args:
      root_dir: Root directory of the current project.
      pkg: Package name (e.g., 'cdd', 'python-cdd', 'ml-switcheroo-ir', 'zero_pytorch').

  Returns:
      Path to the newest matching wheel file if found, otherwise None.
  """
  clean_pkg = pkg.strip().lower()
  for name in _candidate_repo_names(pkg, clean_pkg):
    candidate_dir = root_dir.parent / name
    if not candidate_dir.is_dir():
      continue

    dist_dir = candidate_dir / "dist"
    wheels = list(dist_dir.glob("*.whl")) if dist_dir.is_dir() else []
    wheels.extend(candidate_dir.glob("*.whl"))

    selected = _select_matching_wheel(wheels, clean_pkg)
    if selected is not None:
      return selected

    built = _try_build_candidate_wheel(candidate_dir, dist_dir)
    if built is not None:
      return built

  return None


def _copy_if_newer(src: Path, dst: Path) -> None:
  """Copy ``src`` to ``dst`` when ``dst`` is missing or older than ``src``.

  Args:
      src: Source artifact path.
      dst: Target artifact path.
  """
  if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
    shutil.copy2(src, dst)


def bundle_on_device_wheels(
  root_dir: Path,
  static_dst: Path,
  main_wheel_name: str | None = None,
) -> list[str]:
  """Bundle `ml-switcheroo-compiler` and all `zero-*` wheels into `static_dst` and write `wheels.json`.

  Args:
      root_dir: Project root directory.
      static_dst: Destination `_static` directory.
      main_wheel_name: Optional filename of the built `ml_switcheroo` wheel.

  Returns:
      Ordered list of bundled wheel filenames written to `wheels.json`.
  """
  static_dst.mkdir(exist_ok=True, parents=True)
  bundled: list[str] = []

  for pkg in ON_DEVICE_WHEEL_PACKAGES:
    local_whl = find_local_wheel(root_dir, pkg)
    if local_whl is not None and local_whl.exists():
      _copy_if_newer(local_whl, static_dst / local_whl.name)
      bundled.append(local_whl.name)

  if main_wheel_name and main_wheel_name not in bundled:
    bundled.append(main_wheel_name)

  wheels_manifest = static_dst / "wheels.json"
  wheels_manifest.write_text(json.dumps({"wheels": bundled}, indent=2) + "\n", encoding="utf-8")
  return bundled


def _stage_github_release_wheel(url_or_spec: str, root_dir: Path, static_dst: Path) -> str:
  """Stage a GitHub release ``.whl`` into ``static_dst`` from cache or HTTPS download.

  Args:
      url_or_spec: Direct GitHub wheel URL.
      root_dir: Project root directory.
      static_dst: Destination ``_static`` directory.

  Returns:
      Staged wheel filename.
  """
  import urllib.request

  filename = url_or_spec.split("/")[-1].split("?")[0]
  target_path = static_dst / filename
  if not target_path.exists():
    src_static = root_dir / "docs" / "_static" / filename
    if src_static.exists():
      shutil.copy2(src_static, target_path)
    else:
      print(f"Downloading {url_or_spec} for WASM demo...")
      try:
        req = urllib.request.Request(url_or_spec, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
          shutil.copyfileobj(response, out_file)
      except (OSError, ValueError, TypeError, RuntimeError) as e:
        print(f"Warning: Failed to download {url_or_spec}: {e}")
  return filename


def _rewrite_requirement_line(line: str, root_dir: Path, static_dst: Path) -> str | None:
  """Rewrite a single ``requirements.txt`` line for Pyodide/WASM consumption.

  Args:
      line: Stripped non-comment requirement line.
      root_dir: Project root directory.
      static_dst: Destination ``_static`` directory.

  Returns:
      Rewritten requirement specifier string, or ``None`` if omitted for WASM.
  """
  if " @ " not in line:
    return line

  pkg_raw, url_raw = line.split(" @ ", 1)
  pkg = pkg_raw.strip()
  url_or_spec = url_raw.strip().split(";")[0].strip()
  if "git+" in url_or_spec:
    return None

  local_wheel = find_local_wheel(root_dir, pkg)
  if local_wheel is not None:
    _copy_if_newer(local_wheel, static_dst / local_wheel.name)
    print(f"Grabbed {local_wheel.name} from {local_wheel.parent} for WASM demo...")
    return f"{pkg} @ {local_wheel.name}"

  if url_or_spec.startswith(("http://", "https://")) and "github.com" in url_or_spec and ".whl" in url_or_spec:
    filename = _stage_github_release_wheel(url_or_spec, root_dir, static_dst)
    return f"{pkg} @ {filename}"

  return line


def copy_wheel_and_reqs(app: SphinxAppUnion, exception: Exception | None) -> None:
  """Post-build hook to copy the latest .whl file into _static for WASM usage.

  Connected to 'build-finished' event.

  Args:
      app: The Sphinx application object.
      exception: Any exception raised during the build, or None if the build
        succeeded.
  """
  if exception or not _has_sphinx_builder(app):
    return

  root_dir = Path(__file__).parent.parents[2]
  dist_dir = root_dir / "dist"
  static_dst = Path(str(app.builder.outdir)) / "_static"
  static_dst.mkdir(exist_ok=True, parents=True)

  reqs_file = root_dir / "requirements.txt"
  if reqs_file.exists():
    new_lines: list[str] = []
    for raw_line in reqs_file.read_text(encoding="utf-8").splitlines():
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      rewritten = _rewrite_requirement_line(line, root_dir, static_dst)
      if rewritten is not None:
        new_lines.append(rewritten)
    (static_dst / "requirements.txt").write_text("\n".join(new_lines), encoding="utf-8")

  main_wheel_name: str | None = None
  if dist_dir.exists():
    wheels = list(dist_dir.glob("*.whl"))
    if wheels:
      latest = sorted(wheels, key=os.path.getmtime)[-1]
      main_wheel_name = latest.name
      _copy_if_newer(latest, static_dst / latest.name)

  bundle_on_device_wheels(root_dir, static_dst, main_wheel_name)

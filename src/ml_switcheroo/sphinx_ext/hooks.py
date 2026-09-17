"""Sphinx Build Hooks.

Provides lifecycle hooks to:
1. Register static asset paths (CSS/JS).
2. Copy the distribution wheel to the static directory for the WASM demo.
"""

import typing

if typing.TYPE_CHECKING:
  import sphinx.application

import os
import sphinx
import shutil
from pathlib import Path


def add_static_path(app: "sphinx.application.Sphinx") -> None:
  """Add the extension's static directory to HTML build configuration.

  Connected to 'builder-inited' event.

  Args:
      app: The Sphinx application object.
  """
  static_path = Path(__file__).parent / "static"
  if static_path.exists() and hasattr(app, "config"):
    app.config.html_static_path.append(str(static_path.resolve()))


def find_local_wheel(root_dir: Path, pkg: str) -> typing.Optional[Path]:
  """Locate a local wheel for a given package in sibling repository directories.

  Checks sibling directories for existing .whl distribution artifacts (or builds
  them if the repository exists but no wheels are found).

  Args:
      root_dir: Root directory of the current project.
      pkg: Package name (e.g., 'cdd', 'python-cdd', 'ml-switcheroo-ir').

  Returns:
      Path to the newest matching wheel file if found, otherwise None.
  """
  clean_pkg = pkg.strip().lower()
  candidate_names: typing.List[str]
  if clean_pkg in ("cdd", "python-cdd", "cdd-python"):
    candidate_names = ["python-cdd", "cdd-python", "cdd"]
  else:
    candidate_names = [pkg, pkg.replace("-", "_"), pkg.replace("_", "-")]

  for name in candidate_names:
    candidate_dir = root_dir.parent / name
    if not candidate_dir.is_dir():
      continue

    # Look in candidate_dir/dist and candidate_dir
    dist_dir = candidate_dir / "dist"
    wheels: typing.List[Path] = []
    if dist_dir.is_dir():
      wheels.extend(dist_dir.glob("*.whl"))
    wheels.extend(candidate_dir.glob("*.whl"))

    # Filter matching wheels
    matching: typing.List[Path] = []
    exact_matching: typing.List[Path] = []
    prefix = clean_pkg.replace("-", "_")
    for whl in wheels:
      whl_name = whl.name.lower()
      norm_whl_name = whl_name.replace("-", "_")
      if clean_pkg in ("cdd", "python-cdd", "cdd-python"):
        if whl_name.startswith("cdd-") or whl_name.startswith("cdd_") or whl_name.startswith("python_cdd"):
          matching.append(whl)
          exact_matching.append(whl)
      elif prefix in norm_whl_name:
        matching.append(whl)
        if whl_name.startswith(f"{prefix}-") or whl_name.startswith(f"{clean_pkg}-"):
          exact_matching.append(whl)

    if exact_matching:
      return sorted(exact_matching, key=os.path.getmtime)[-1]
    if matching:
      return sorted(matching, key=os.path.getmtime)[-1]

    # Try building wheel if repository exists without pre-built wheels
    if (candidate_dir / "pyproject.toml").exists() or (candidate_dir / "setup.py").exists():
      try:
        import subprocess

        subprocess.run(["uv", "build", "--wheel"], cwd=candidate_dir, check=True, capture_output=True)
        if dist_dir.is_dir():
          built = list(dist_dir.glob("*.whl"))
          if built:
            return sorted(built, key=os.path.getmtime)[-1]
      except Exception:
        pass

  return None


def copy_wheel_and_reqs(app: "sphinx.application.Sphinx", exception: typing.Optional[Exception]) -> None:
  """Post-build hook to copy the latest .whl file into _static for WASM usage.

  Connected to 'build-finished' event.

  Args:
      app: The Sphinx application object.
      exception: Any exception raised during the build, or None if the build
        succeeded.
  """
  if exception or not hasattr(app, "builder"):
    return

  # Resolve relative to this file inside 'src/ml_switcheroo/sphinx_ext'
  # Project root is 3 levels up
  here = Path(__file__).parent
  root_dir = here.parents[2]
  dist_dir = root_dir / "dist"

  static_dst = Path(app.builder.outdir) / "_static"
  static_dst.mkdir(exist_ok=True, parents=True)

  reqs_file = root_dir / "requirements.txt"
  if reqs_file.exists():
    import urllib.request

    with open(reqs_file, "r", encoding="utf-8") as f:
      lines = f.readlines()

    new_lines: typing.List[str] = []
    for raw_line in lines:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      if " @ " in line:
        pkg, url_or_spec = line.split(" @ ", 1)
        pkg = pkg.strip()
        url_or_spec = url_or_spec.strip()

        # Check for local sibling repository first (e.g. ../python-cdd or ../cdd-python)
        local_wheel = find_local_wheel(root_dir, pkg)
        if local_wheel is not None:
          target_path = static_dst / local_wheel.name
          if not target_path.exists() or target_path.stat().st_mtime < local_wheel.stat().st_mtime:
            shutil.copy2(local_wheel, target_path)
          new_lines.append(f"{pkg} @ {local_wheel.name}")
          print(f"Grabbed {local_wheel.name} from {local_wheel.parent} for WASM demo...")
        elif "@ http" in line and "github.com" in line and ".whl" in line:
          # On ghpages or when local repo is not available: download from GitHub releases
          filename = url_or_spec.split("/")[-1]
          target_path = static_dst / filename
          if not target_path.exists():
            print(f"Downloading {url_or_spec} for WASM demo...")
            try:
              req = urllib.request.Request(url_or_spec, headers={"User-Agent": "Mozilla/5.0"})
              with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            except Exception as e:
              print(f"Warning: Failed to download {url_or_spec}: {e}")
          new_lines.append(f"{pkg} @ {filename}")
        elif "git+" in url_or_spec:
          # Git sources are omitted for browser WASM environments (Pyodide cannot clone git repos)
          continue
        else:
          new_lines.append(line)
      else:
        new_lines.append(line)

    with open(static_dst / "requirements.txt", "w", encoding="utf-8") as f:
      f.write("\n".join(new_lines))

  if dist_dir.exists():
    wheels = list(dist_dir.glob("*.whl"))
    if wheels:
      latest = sorted(wheels, key=os.path.getmtime)[-1]
      target_file = static_dst / latest.name
      # Copy if newer or missing
      if not target_file.exists() or target_file.stat().st_mtime < latest.stat().st_mtime:
        shutil.copy2(latest, target_file)

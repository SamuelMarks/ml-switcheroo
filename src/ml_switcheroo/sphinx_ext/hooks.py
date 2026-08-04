"""Sphinx Build Hooks.

Provides lifecycle hooks to:
1. Register static asset paths (CSS/JS).
2. Copy the distribution wheel to the static directory for the WASM demo.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Optional


def add_static_path(app: Any) -> None:
  """Adds the extension's static directory to HTML build configuration.

  Connected to 'builder-inited' event.

  Args:
      app: The Sphinx application object.
  """
  static_path = Path(__file__).parent / "static"
  if static_path.exists() and hasattr(app, "config"):
    app.config.html_static_path.append(str(static_path.resolve()))


def copy_wheel_and_reqs(app: Any, exception: Optional[Exception]) -> None:
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

    new_lines = []
    for line in lines:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      if "@ http" in line and "github.com" in line and ".whl" in line:
        pkg, url = line.split(" @ ", 1)
        filename = url.split("/")[-1]
        target_path = static_dst / filename
        if not target_path.exists():
          print(f"Downloading {url} for WASM demo...")
          try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
              shutil.copyfileobj(response, out_file)
          except Exception as e:
            print(f"Warning: Failed to download {url}: {e}")
        new_lines.append(f"{pkg} @ {filename}")
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

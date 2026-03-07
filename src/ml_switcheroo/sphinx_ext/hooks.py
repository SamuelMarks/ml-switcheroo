"""
Sphinx Build Hooks.

Provides lifecycle hooks to:
1. Register static asset paths (CSS/JS).
2. Copy the distribution wheel to the static directory for the WASM demo.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Optional


def add_static_path(app: Any) -> None:
  """
  Adds the extension's static directory to HTML build configuration.

  Connected to 'builder-inited' event.
  """
  static_path = Path(__file__).parent / "static"  # pragma: no cover
  if static_path.exists() and hasattr(app, "config"):  # pragma: no cover
    app.config.html_static_path.append(str(static_path.resolve()))  # pragma: no cover


def copy_wheel_and_reqs(app: Any, exception: Optional[Exception]) -> None:
  """
  Post-build hook to copy the latest .whl file into _static for WASM usage.

  Connected to 'build-finished' event.
  """
  if exception or not hasattr(app, "builder"):  # pragma: no cover
    return  # pragma: no cover

  # Resolve relative to this file inside 'src/ml_switcheroo/sphinx_ext'
  # Project root is 3 levels up
  here = Path(__file__).parent  # pragma: no cover
  root_dir = here.parents[2]  # pragma: no cover
  dist_dir = root_dir / "dist"  # pragma: no cover

  static_dst = Path(app.builder.outdir) / "_static"  # pragma: no cover
  static_dst.mkdir(exist_ok=True, parents=True)  # pragma: no cover

  reqs_file = root_dir / "requirements.txt"  # pragma: no cover
  if reqs_file.exists():  # pragma: no cover
    shutil.copy2(reqs_file, static_dst / "requirements.txt")  # pragma: no cover

  if dist_dir.exists():  # pragma: no cover
    wheels = list(dist_dir.glob("*.whl"))  # pragma: no cover
    if wheels:  # pragma: no cover
      latest = sorted(wheels, key=os.path.getmtime)[-1]  # pragma: no cover
      target_file = static_dst / latest.name  # pragma: no cover
      # Copy if newer or missing
      if not target_file.exists() or target_file.stat().st_mtime < latest.stat().st_mtime:  # pragma: no cover
        shutil.copy2(latest, target_file)  # pragma: no cover

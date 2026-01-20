import os
import sys
import inspect
import warnings
from datetime import datetime

# -- Warning Suppression -----------------------------------------------------
# Silence common upstream warnings in dependencies during doc generation
# Suppress Keras TF2ONNX np.object warning
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.src.export.tf2onnx_lib")

# -- Project information -----------------------------------------------------
project = "ml-switcheroo"
author = "Samuel Marks"
copyright = f"{datetime.now().year}, {author}"
version = "0.0.1"
release = version

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")))

# -- General configuration ---------------------------------------------------
extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.napoleon",
  "sphinx.ext.linkcode",  # Using linkcode for GitHub source linking
  "sphinx.ext.intersphinx",
  "autoapi.extension",
  "sphinxcontrib.autodoc_pydantic",
  "myst_parser",
  "sphinxcontrib.mermaid",
  "sphinx_copybutton",
  "sphinx_material",
  "ml_switcheroo.sphinx_ext",
]

# -- Warning Suppression -----------------------------------------------------
# Silence autoapi import resolution warnings (static analysis noise)
suppress_warnings = ["autoapi.python_import_resolution"]

# -- AutoAPI Configuration ---------------------------------------------------
autoapi_dirs = ["../src"]
autoapi_type = "python"
autoapi_root = "api"
autoapi_options = [
  "members",
  "undoc-members",
  "show-inheritance",
  "show-module-summary",
  "special-members",
  "imported-members",
]
# Strict ignore patterns
autoapi_ignore = ["*migrations*", "*/tests/*", "*test_*.py", "*/sphinx_ext/*"]

# -- Mermaid Configuration ---------------------------------------------------
# Optional: Explicitly set the CDN version if rendering fails locally
mermaid_version = "11.12.0"

# -- MyST Parser Configuration -----------------------------------------------
myst_enable_extensions = [
  "amsmath",
  "colon_fence",
  "deflist",
  "dollarmath",
  "fieldlist",
  "html_admonition",
  "html_image",
  "linkify",
  "replacements",
  "smartquotes",
  "tasklist",
]
myst_heading_anchors = 3

# VITAL: This tells MyST to convert ```mermaid blocks into the \
# '.. mermaid::' directive provided by sphinxcontrib-mermaid
myst_fence_as_directive = ["mermaid"]

# -- Theme Configuration -----------------------------------------------------
html_theme = "sphinx_material"
html_title = "ml-switcheroo"
html_short_title = "switcheroo"

html_theme_options = {
  "nav_title": "ml-switcheroo",
  "color_primary": "blue",
  "color_accent": "light-blue",
  "repo_url": "https://github.com/SamuelMarks/ml-switcheroo",
  "repo_name": "ml-switcheroo",
  "globaltoc_depth": 2,
  "globaltoc_collapse": True,
  "globaltoc_includehidden": True,
  "master_doc": False,
  "version_dropdown": False,
  "table_classes": ["plain"],
}

html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Linkcode Resolution Logic -----------------------------------------------
def linkcode_resolve(domain, info):
  """
  Resolve a GitHub URL for the given Python object.
  Required by sphinx.ext.linkcode to generate [source] links.
  """
  if domain != "py":
    return None

  if not info["module"]:
    return None

  # Try to import the module to inspect it
  mod = sys.modules.get(info["module"])
  if not mod:
    # If not already imported, try importlib (though autoapi usually handles this via static analysis,
    # linkcode requires runtime inspection for line numbers)
    return None

  # Traverse attributes to find the object
  obj = mod
  for part in info["fullname"].split("."):
    try:
      obj = getattr(obj, part)
    except AttributeError:
      return None

  # Unwrap decorators if needed
  while hasattr(obj, "__wrapped__"):
    obj = obj.__wrapped__

  # Introspect source file and lines
  try:
    fn = inspect.getsourcefile(obj)
    source, lineno = inspect.getsourcelines(obj)
  except (TypeError, OSError):
    return None

  if not fn:
    return None

  # Get relative path to repo root
  # assuming conf.py is in docs/ and project root is parent "../"
  root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

  try:
    rel_path = os.path.relpath(fn, start=root_path)
  except ValueError:
    return None

  # Ignore if file is outside repo (e.g. system installs or site-packages)
  if rel_path.startswith(".."):
    return None

  end_lineno = lineno + len(source) - 1

  # Construct GitHub URL (assuming master branch)
  blob_url = f"https://github.com/SamuelMarks/ml-switcheroo/blob/master/{rel_path}#L{lineno}-L{end_lineno}"
  return blob_url

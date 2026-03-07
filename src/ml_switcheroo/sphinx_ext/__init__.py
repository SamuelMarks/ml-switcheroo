"""
Sphinx Extension for ML-Switcheroo.

This package provides custom Sphinx directives and hooks to generate
interactive documentation, including:
1.  **WASM Demo**: Embeds a client-side transpiler demo (Pyodide).
2.  **Auto-Docs**: Automatically generates API reference pages for the
    Abstract Operations defined in the Semantic Knowledge Base.
3.  **Visualization**: Injects assets for TikZ and AST rendering.
"""

from typing import Any, Dict

from ml_switcheroo import __version__
from ml_switcheroo.sphinx_ext.directive import SwitcherooDemo
from ml_switcheroo.sphinx_ext.hooks import add_static_path, copy_wheel_and_reqs
from ml_switcheroo.sphinx_ext.autogen_ops import generate_op_docs


def setup(app: Any) -> Dict[str, Any]:
  """
  Sphinx Extension Setup Hook.

  Registers directives, connects build events, and adds static assets.
  """
  # --- Directives ---
  app.add_directive("switcheroo_demo", SwitcherooDemo)  # pragma: no cover

  # --- CodeMirror Assets (Editor) ---
  # Base
  app.add_css_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css")  # pragma: no cover
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js")  # pragma: no cover

  # Modes
  app.add_js_file(
    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js"
  )  # pragma: no cover
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/xml/xml.min.js")  # pragma: no cover
  app.add_js_file(
    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/javascript/javascript.min.js"
  )  # pragma: no cover
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/css/css.min.js")  # pragma: no cover
  app.add_js_file(
    "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/htmlmixed/htmlmixed.min.js"
  )  # pragma: no cover

  # --- Mermaid JS (AST Visualizer) ---
  app.add_js_file("https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js")  # pragma: no cover

  # --- TikZJax (Graph Renderer) ---
  # Use CDN for fonts to fix relative path issues
  app.add_css_file("https://tikzjax.com/v1/fonts.css")  # pragma: no cover
  app.add_js_file(None, body="window.TIKZJAX_URL = '_static/tikzjax/tikzjax.js';")  # pragma: no cover

  # --- Local Extension Assets ---
  app.add_css_file("switcheroo_demo.css")  # pragma: no cover
  app.add_css_file("trace_graph.css")  # pragma: no cover
  app.add_js_file("trace_render.js")  # pragma: no cover
  app.add_js_file("switcheroo_demo.js")  # pragma: no cover

  # --- Docs UI Assets (Feature 3) ---
  app.add_css_file("op_tabs.css")  # pragma: no cover
  app.add_js_file("op_tabs.js")  # pragma: no cover

  import os  # pragma: no cover

  # --- Build Hooks ---
  # 1. Register static path
  app.connect("builder-inited", add_static_path)  # pragma: no cover
  # 2. Copy WASM Wheel
  app.connect("build-finished", copy_wheel_and_reqs)  # pragma: no cover
  # 3. Generate Operation Docs (Feature 4)
  if os.environ.get("HOMEPAGE_ONLY") != "1":  # pragma: no cover
    app.connect("builder-inited", generate_op_docs)  # pragma: no cover

  return {  # pragma: no cover
    "version": __version__,
    "parallel_read_safe": True,
    "parallel_write_safe": True,
  }

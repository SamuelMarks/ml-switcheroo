"""
Sphinx Extension for ML-Switcheroo WASM Demo.

Registers the `switcheroo_demo` directive and configures build hooks
to support the client-side interactive experience.
"""

from typing import Any, Dict

from ml_switcheroo import __version__
from ml_switcheroo.sphinx_ext.directive import SwitcherooDemo
from ml_switcheroo.sphinx_ext.hooks import add_static_path, copy_wheel_and_reqs


def setup(app: Any) -> Dict[str, Any]:
  """
  Sphinx Extension Setup Hook.
  """
  app.add_directive("switcheroo_demo", SwitcherooDemo)

  # CodeMirror Assets
  app.add_css_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js")
  app.add_js_file("https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js")

  # Mermaid JS (Critical for AST Visualizer)
  app.add_js_file("https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js")

  # Local Assets
  app.add_css_file("switcheroo_demo.css")
  app.add_css_file("trace_graph.css")
  app.add_js_file("trace_render.js")
  app.add_js_file("switcheroo_demo.js")

  # Build Hooks
  app.connect("builder-inited", add_static_path)
  app.connect("build-finished", copy_wheel_and_reqs)

  return {
    "version": __version__,
    "parallel_read_safe": True,
    "parallel_write_safe": True,
  }

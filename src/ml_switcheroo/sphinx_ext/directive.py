"""
Sphinx Directive for embedding the WASM Demo.

Contains the `SwitcherooDemo` class which ties the scanning and rendering logic
into a Docutils node.
"""

from typing import List
from docutils import nodes
from docutils.parsers.rst import Directive

from ml_switcheroo.sphinx_ext.registry import scan_registry
from ml_switcheroo.sphinx_ext.rendering import render_demo_html


class SwitcherooDemo(Directive):
  """
  Sphinx Directive to embed the interactive WASM demo.

  Usage:
      .. switcheroo_demo::
  """

  has_content = True

  def run(self) -> List[nodes.raw]:
    """
    Main execution entry point for the directive.

    1. Scans the registry for frameworks and examples.
    2. Renders the HTML template with dynamic dropdowns.

    Returns:
        List[nodes.raw]: A list containing the raw HTML node.
    """
    hierarchy, examples_json, tier_metadata_json = scan_registry()
    html = render_demo_html(hierarchy, examples_json, tier_metadata_json)
    return [nodes.raw("", html, format="html")]

"""
HTML/SVG DSL Core Package.

This package defines the intermediate representation for the HTML Visual DSL.
This allows converting ASTs into a visual Grid-based explanation format.
"""

from ml_switcheroo.core.html.nodes import HtmlNode, GridBox, SvgArrow, HtmlDocument

__all__ = ["HtmlNode", "GridBox", "SvgArrow", "HtmlDocument"]

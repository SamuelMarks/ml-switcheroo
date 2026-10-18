"""Compatibility re-export for ONNX Markdown Specification Importer.

Delegates to `ml_switcheroo.importers.onnx_reader.OnnxSpecImporter`.
"""

from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter, clean_html_tags

__all__ = ["OnnxSpecImporter", "clean_html_tags"]

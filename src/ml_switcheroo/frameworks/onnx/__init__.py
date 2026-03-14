"""ONNX Framework Plugin.
Registers the ONNX framework adapter and sets up submodules.
"""

from ml_switcheroo.frameworks.onnx.adapter import OnnxFramework

__all__ = ["OnnxFramework"]

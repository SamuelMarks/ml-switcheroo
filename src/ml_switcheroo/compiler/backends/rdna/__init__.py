"""
RDNA Backend Package.

Contains the backend implementation for synthesizing AMD RDNA assembly
from the Logical Graph representation.
"""

from ml_switcheroo.compiler.backends.rdna.synthesizer import (
  RdnaSynthesizer,
  RdnaBackend,
)

__all__ = ["RdnaSynthesizer", "RdnaBackend"]

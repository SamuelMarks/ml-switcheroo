"""SASS Backend Package.

Contains the backend implementation for synthesizing NVIDIA SASS assembly
from the Logical Graph representation.
"""

from ml_switcheroo.core.compiler.backends.sass.synthesizer import (
  SassSynthesizer,
)
from ml_switcheroo.core.compiler.backends.sass.backend import SassBackend
from ml_switcheroo.core.compiler.backends.sass.emitter import SassEmitter

__all__ = ["SassSynthesizer", "SassBackend", "SassEmitter"]

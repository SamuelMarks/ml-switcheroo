"""
Fuzzer Package.

This package provides the input generation engine for verification tests.
It exposes the main `InputFuzzer` class which orchestrates the generation
of random inputs (Arrays, Scalars, Containers) based on semantic type hints
and constraints.
"""

from ml_switcheroo.testing.fuzzer.core import InputFuzzer

__all__ = ["InputFuzzer"]

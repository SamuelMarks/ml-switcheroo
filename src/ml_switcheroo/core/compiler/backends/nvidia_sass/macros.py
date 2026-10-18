"""Facade for NVIDIA SASS macros."""

# ruff: noqa: F403, F401

from ml_switcheroo.core.compiler.backends.nvidia_sass.activation import *
from ml_switcheroo.core.compiler.backends.nvidia_sass.conv import *
from ml_switcheroo.core.compiler.backends.nvidia_sass.gemm import *
from ml_switcheroo.core.compiler.backends.nvidia_sass.misc import *
from ml_switcheroo.core.compiler.backends.nvidia_sass.normalization import *
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister as NvidiaSassRegister
from ml_switcheroo.core.compiler.isa.allocator import RegisterAllocatorProtocol as RegisterAllocatorProtocol

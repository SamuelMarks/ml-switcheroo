"""Test module."""

import inspect
from typing import Any, Dict, List

import ml_switcheroo.core.compiler.backends.nvidia_sass.macros as macros
import ml_switcheroo.core.compiler.backends.nvidia_sass.macros_extra as macros_extra
from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import RegisterAllocatorProtocol
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode, NvidiaSassRegister


class DummyAllocator(RegisterAllocatorProtocol):
  """Docstring."""

  def get_register(self, var_name: str) -> NvidiaSassRegister:
    """Docstring."""
    return NvidiaSassRegister(name=f"REG_{var_name}")

  def allocate_temp(self) -> NvidiaSassRegister:
    """Docstring."""
    return NvidiaSassRegister(name="TEMP")


def test_all_macros() -> None:
  """Docstring."""
  allocator: DummyAllocator = DummyAllocator()
  metadata: Dict[str, Any] = {"k": 3, "seq_len": 5}
  node_id: str = "test_node"

  for module in [macros, macros_extra]:
    for name, obj in inspect.getmembers(module, inspect.isfunction):
      if name.startswith("expand_"):
        nodes: List[NvidiaSassNode] = obj(allocator, node_id, metadata)
        assert len(nodes) > 0

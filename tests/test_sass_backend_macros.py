"""Test module."""

import inspect
from typing import Dict, Any
import ml_switcheroo.core.compiler.backends.sass.macros as macros
import ml_switcheroo.core.compiler.backends.sass.macros_extra as macros_extra
from ml_switcheroo.core.compiler.frontends.sass.cst import SassRegister


class DummyAllocator:
  """Test element."""

  def get_register(self, var_name: str) -> SassRegister:
    """Test element."""
    return SassRegister(name=f"REG_{var_name}")

  def allocate_temp(self) -> SassRegister:
    """Test element."""
    return SassRegister(name="TEMP")


def test_all_macros():
  """Test element."""
  allocator = DummyAllocator()
  metadata: Dict[str, Any] = {"k": 3, "seq_len": 5}
  node_id = "test_node"

  for module in [macros, macros_extra]:
    for name, obj in inspect.getmembers(module, inspect.isfunction):
      if name.startswith("expand_"):
        nodes = obj(allocator, node_id, metadata)
        assert len(nodes) > 0

"""Test suite for nvidia sass backend macros."""

import typing

import inspect

import ml_switcheroo.core.compiler.backends.nvidia_sass.macros as macros
import ml_switcheroo.core.compiler.backends.nvidia_sass.macros as macros_extra
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassRegister


class DummyAllocator:
  """Test helper class DummyAllocator."""

  def get_register(self, var_name: typing.Any) -> typing.Any:  # noqa: F811
    """Execute get register helper logic.

    Args:
        var_name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return NvidiaSassRegister(name=f"REG_{var_name}")

  def allocate_temp(self) -> typing.Any:  # noqa: F811
    """Execute allocate temp helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return NvidiaSassRegister(name="TEMP")


def test_all_macros() -> None:
  """Verify all macros behavior."""
  allocator: typing.Any = DummyAllocator()
  metadata: typing.Any = {"k": 3, "seq_len": 5}
  node_id: typing.Any = "test_node"

  for module in [macros, macros_extra]:
    for name, obj in inspect.getmembers(module, inspect.isfunction):
      if name.startswith("expand_"):
        nodes: typing.Any = obj(allocator, node_id, metadata)
        assert len(nodes) > 0

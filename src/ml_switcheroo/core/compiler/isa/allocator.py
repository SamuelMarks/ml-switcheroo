"""Common ISA Register Allocation Pipeline."""

import typing

from ml_switcheroo.core.compiler.ir import LogicalGraph

# We use generic type bounds so the allocator can return architecture-specific register AST nodes.
T = typing.TypeVar("T")
RegT_co = typing.TypeVar("RegT_co", covariant=True)


class RegisterAllocatorProtocol(typing.Protocol[RegT_co]):
  """Protocol matching the register allocation interface expected by macro expanders."""

  def get_register(self, var_name: str) -> RegT_co:
    """Retrieve or allocate a physical register for a symbolic variable name.

    Args:
        var_name: Symbolic variable or node identifier.

    Returns:
        Architecture-specific register CST node.
    """
    ...

  def allocate_temp(self) -> RegT_co:
    """Allocate a fresh temporary physical register.

    Returns:
        Architecture-specific temporary register CST node.
    """
    ...


class GenericRegisterAllocator(typing.Generic[T]):
  """Generic Register Allocator for Hardware ISAs.

  Implements standard liveness analysis, linear scan register allocation,
  and spilling infrastructure that can be specialized by AMD RDNA and NVIDIA SASS.
  """

  def __init__(self, max_registers: int, register_factory: typing.Callable[[int], T]) -> None:
    """Initialize the register allocator with pool capacity and register factory.

    Args:
        max_registers: Maximum number of physical registers in the pool.
        register_factory: Callable constructing an architecture register node from an integer index.
    """
    self.max_registers = max_registers
    self.register_factory = register_factory

    self._var_to_reg_id: dict[str, int] = {}
    self._free_pool: list[int] = list(range(max_registers))
    self._liveness_map: dict[str, int] = {}

  def free_register(self, var_name: str) -> None:
    """Release the register assigned to ``var_name`` back into the available pool.

    Args:
        var_name: Symbolic variable identifier whose register can be reused.
    """
    if var_name in self._var_to_reg_id:
      reg_id = self._var_to_reg_id.pop(var_name)
      self._free_pool.append(reg_id)

  def get_register(self, var_name: str) -> T:
    """Retrieve the existing register for ``var_name`` or allocate the next available register.

    Args:
        var_name: Symbolic variable identifier.

    Returns:
        Constructed register node of type ``T``.

    Raises:
        ValueError: If the free register pool is exhausted.
    """
    if var_name in self._var_to_reg_id:
      return self.register_factory(self._var_to_reg_id[var_name])

    if not self._free_pool:
      # Basic spilling stub/raise error based on backend
      raise ValueError(f"Register overflow! Exceeded {self.max_registers} registers.")

    # Simplistic linear scan - pop first available
    reg_id = self._free_pool.pop(0)
    self._var_to_reg_id[var_name] = reg_id
    return self.register_factory(reg_id)

  def allocate_temp(self) -> T:
    """Allocate a uniquely named temporary register.

    Returns:
        Constructed temporary register node of type ``T``.
    """
    import uuid

    temp_name = f"__temp_{uuid.uuid4().hex[:8]}__"
    return self.get_register(temp_name)

  def reset(self) -> None:
    """Reset all variable bindings, free register pool, and liveness counters."""
    self._var_to_reg_id.clear()
    self._free_pool = list(range(self.max_registers))
    self._liveness_map.clear()

  def build_liveness(self, graph: LogicalGraph) -> None:
    """Compute reference-count liveness for each source node in a LogicalGraph.

    Args:
        graph: The LogicalGraph whose edges define consumer counts.
    """
    self._liveness_map.clear()
    for edge in graph.edges:
      if edge.source not in self._liveness_map:
        self._liveness_map[edge.source] = 0
      self._liveness_map[edge.source] += 1

  def record_usage(self, var_name: str) -> None:
    """Decrement remaining liveness count for ``var_name`` and free its register when zero.

    Args:
        var_name: Symbolic variable identifier that was just consumed.
    """
    if var_name in self._liveness_map:
      self._liveness_map[var_name] -= 1
      if self._liveness_map[var_name] <= 0:
        self.free_register(var_name)

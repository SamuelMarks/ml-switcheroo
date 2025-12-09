"""
Base Protocol and Registry for Framework Adapters.

This module defines the interface that all Framework Adapters must implement.
"""

from typing import Any, Protocol, Type, Dict, List, Tuple, Optional
from ml_switcheroo.semantics.schema import StructuralTraits


class FrameworkAdapter(Protocol):
  """
  Protocol definition for a Framework Adapter.
  """

  # --- 1. Verification Logic ---
  def convert(self, data: Any) -> Any:
    """Converts input data (usually array-like) to this framework's tensor type."""
    ...

  @classmethod
  def get_import_stmts(cls) -> str:
    """Returns import statements needed for generated test harness."""
    ...

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    """Returns python code to create a tensor from a numpy variable."""
    ...

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    """Returns python code to convert a tensor back to a numpy array."""
    ...

  # --- 2. Discovery Metadata ---
  @property
  def search_modules(self) -> List[str]:
    """
    List of python modules to scan when running `ml_switcheroo sync`.
    e.g. ["torch", "torch.nn"]
    """
    ...

  @property
  def display_name(self) -> str:
    """Friendly name for UI/CLI (e.g. 'PyTorch')."""
    ...

  @property
  def ui_priority(self) -> int:
    """
    Integer defining the sort order in UI tables and Dropdowns.
    Lower numbers appear first (0 = Highest Priority).
    """
    ...

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """
    Regex patterns used to categorize discovered APIs into Semantic Tiers.
    Returns: Dict mapping tier identifiers to lists of regex pattern strings.
    """
    ...

  # --- 3. Import Management ---
  @property
  def import_alias(self) -> Tuple[str, str]:
    """
    Default module import path and alias.
    Returns: (module_path, local_alias) -> e.g. ("jax.numpy", "jnp")
    """
    ...

  # --- 4. Hierarchy ---
  @property
  def inherits_from(self) -> Optional[str]:
    """
    Key of a parent framework to inherit mappings from.
    e.g. 'paxml' inherits from 'jax'. Returns None if no parent.
    """
    ...

  # --- 5. Structural Rewriting (Zero-Edit) ---
  @property
  def structural_traits(self) -> StructuralTraits:
    """
    Defines how classes and functions should be transformed to match this framework.
    Returns: A Pydantic model containing AST rewriting rules.
    """
    ...

  @property
  def rng_seed_methods(self) -> List[str]:
    """
    Returns a list of method names that modify global random state.
    Used by PurityScanner to detect side effects.
    """
    ...

  # --- 6. Hardware Abstraction (Zero-Edit) ---
  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Returns Python source code to instantiate a device on this backend."""
    ...

  # --- 7. IO Serialization (Zero-Edit) ---
  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Returns Python source code to perform IO operations."""
    ...

  def get_serialization_imports(self) -> List[str]:
    """Returns a list of import statements required for serialization to work."""
    ...

  # --- 8. Documentation & Demo ---
  @classmethod
  def get_example_code(cls) -> str:
    """
    Returns a Python code snippet demonstrating a typical operation in this
    framework. Used by the Web Frontend to populate the demo editor.
    """
    ...


# Global Registry
_ADAPTER_REGISTRY: Dict[str, Type[FrameworkAdapter]] = {}


def register_framework(name: str):
  """Decorator to register a framework adapter class."""

  def wrapper(cls):
    _ADAPTER_REGISTRY[name] = cls
    return cls

  return wrapper


def get_adapter(name: str) -> Optional[FrameworkAdapter]:
  """Factory to instantiate an adapter."""
  cls = _ADAPTER_REGISTRY.get(name)
  if cls:
    return cls()
  return None

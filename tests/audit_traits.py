"""Module docstring."""

from ml_switcheroo.frameworks import available_frameworks, get_adapter
from ml_switcheroo.frameworks.base import FrameworkAdapter, StructuralTraits
from typing import Optional


def analyze_frameworks() -> None:
  """Analyze frameworks."""
  print(f"{'Framework':<15} | {'Base Class Def':<20} | {'Forward Def':<15} | {'Req Super':<10}")
  print("-" * 65)
  for fw in available_frameworks():
    adapter: Optional[FrameworkAdapter] = get_adapter(fw)
    if not adapter:
      continue
    traits: StructuralTraits = adapter.structural_traits
    base: str = traits.module_base if traits.module_base else "N/A"
    fwd: str = traits.forward_method if traits.forward_method else "N/A"
    req_super: str = str(traits.requires_super_init)
    print(f"{fw:<15} | {base:<20} | {fwd:<15} | {req_super:<10}")


if __name__ == "__main__":
  analyze_frameworks()

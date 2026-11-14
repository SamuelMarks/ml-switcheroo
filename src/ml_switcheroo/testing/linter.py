"""Structural Validation Linter Facade (The Anti-Pollution Check).

Re-exports :class:`ml_switcheroo.analysis.linter.StructuralLinter` and provides
``validate_transpilation`` expanding source framework aliases, search modules,
and inheritance parents via ``get_adapter``.
"""

import typing

from ml_switcheroo.analysis.linter import StructuralLinter as StructuralLinter
from ml_switcheroo.frameworks import get_adapter


@typing.runtime_checkable
class _SearchModulesProtocol(typing.Protocol):
  """Protocol for adapters exposing ``search_modules``.

  Attributes:
      search_modules: Sequence of module names searched by the adapter.
  """

  @property
  def search_modules(self) -> list[str]:
    """Return adapter search module names.

    Returns:
        List of search module strings.
    """
    ...


def validate_transpilation(code: str, source_framework: str) -> tuple[bool, list[str]]:
  """Lint generated code against a specific ``source_framework`` and its aliases/parents.

  Args:
      code: Generated Python source code.
      source_framework: Source framework identifier that should have been eliminated.

  Returns:
      tuple[bool, list[str]]: ``(is_valid, list_of_errors)``.
  """
  forbidden = {source_framework}

  adapter = get_adapter(source_framework)
  if adapter:
    try:
      mod, _ = adapter.import_alias
      forbidden.add(mod.split(".")[0])
    except AttributeError:
      pass

    if isinstance(adapter, _SearchModulesProtocol):
      for mod in adapter.search_modules:
        forbidden.add(str(mod).split(".")[0])

    try:
      inh = adapter.inherits_from
      if inh:
        forbidden.add(inh)
    except AttributeError:
      pass

  linter = StructuralLinter(forbidden)
  errors = linter.check(code)
  return len(errors) == 0, errors


__all__ = ["StructuralLinter", "validate_transpilation"]

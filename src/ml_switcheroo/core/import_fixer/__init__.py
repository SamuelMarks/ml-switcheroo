"""
Import Fixer Package.

This package provides the ``ImportFixer`` class, a LibCST transformer responsible for:
1.  **Injection**: Adding required imports for the target framework.
2.  **Pruning**: Removing unused imports from the source framework.
3.  **Refinement**: Collapsing aliases and cleaning up re-exports.

It uses ``ImportResolver`` to pre-calculate necessary changes before transformation.
"""

from ml_switcheroo.core.import_fixer.attributes_mixin import AttributeMixin
from ml_switcheroo.core.import_fixer.base import BaseImportFixer
from ml_switcheroo.core.import_fixer.imports_mixin import ImportMixin
from ml_switcheroo.core.import_fixer.injection_mixin import InjectionMixin
from ml_switcheroo.core.import_fixer.resolution import ImportResolver, ResolutionPlan, ImportReq


class ImportFixer(AttributeMixin, ImportMixin, InjectionMixin, BaseImportFixer):
  """
  Composite Transformer for managing imports and namespacing.

  Inherits functionality from:
  - :class:`AttributeMixin`: simplifying dotted attribute access.
  - :class:`ImportMixin`: rewriting import statements using the plan.
  - :class:`InjectionMixin`: injecting missing top-level imports from the plan.
  - :class:`BaseImportFixer`: State management and configuration.
  """


__all__ = ["ImportFixer", "ImportResolver", "ResolutionPlan", "ImportReq"]

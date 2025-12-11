"""
Sync Mechanism for Test Template Generation.

Resolves fragmentation between Adapter Classes and JSON Configuration.
It extracts metadata (imports, conversion syntax) from registered FrameworkAdapters
and formats it into the template dictionary structure used by `TestGenerator`.

This module now integrates with the Distributed Semantics architecture:
while it primarily reads from Python code (Adapters), its output is designed
to be merged with templates loaded from JSON overlays in the `SemanticsManager`.
"""

from typing import Dict, Optional, Protocol, runtime_checkable, Any, Type
from ml_switcheroo.testing.adapters import _ADAPTER_REGISTRY, FrameworkAdapter


@runtime_checkable
class TemplateProvider(Protocol):
  """
  Protocol for Adapters that wish to provide auto-generated templates.

  Implement this protocol on your Adapter class to allow `TestGenerator`
  to automatically support your framework without writing JSON config.
  """

  @classmethod
  def get_import_stmts(cls) -> str:
    """
    Returns the import block required for this framework.
    e.g., "import torch" or "import jax.numpy as jnp".
    """
    ...

  @classmethod
  def get_creation_syntax(cls, var_name: str) -> str:
    """
    Returns Python code string to convert a numpy variable to this framework's tensor.
    e.g., f"torch.from_numpy({var_name})"
    """
    ...

  @classmethod
  def get_numpy_conversion_syntax(cls, var_name: str) -> str:
    """
    Returns Python code string to convert a tensor back to numpy/python object.
    e.g., f"{var_name}.detach().cpu().numpy()"
    """
    ...


class TemplateGenerator:
  """
  Generates test templates by introspecting the Adapter Registry.
  """

  @staticmethod
  def generate_templates(base_templates: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Dict[str, str]]:
    """
    Merges existing templates with those auto-generated from registered Adapters.

    Args:
        base_templates: Existing dictionary of templates (e.g. from Snapshot JSONs).

    Returns:
        A unified dictionary of {framework: {template_keys...}}.
    """
    import copy

    templates = copy.deepcopy(base_templates) if base_templates else {}

    for fw_name, adapter_cls in _ADAPTER_REGISTRY.items():
      # If the adapter implements the TemplateProvider protocol (duck typing check)
      if _is_template_provider(adapter_cls):
        generated = _extract_template(adapter_cls)  # type: ignore

        if fw_name in templates:
          # Update source keys, preserving JSON overrides (e.g. 'jit_wrap')
          # We merge: JSON keys override Code keys except for essential syntax
          # Actually, usually Overlay/JSON preference > Code Default
          # So we update code defaults into 'templates' only if keys missing?
          # Let's assume Code is Default, JSON is Override.
          # So take Generated, update with Existing.
          merged = generated.copy()
          merged.update(templates[fw_name])
          templates[fw_name] = merged
        else:
          templates[fw_name] = generated

    return templates


def _is_template_provider(cls_obj: Type[FrameworkAdapter]) -> bool:
  """Checks if class implements required static methods."""
  return (
    hasattr(cls_obj, "get_import_stmts")
    and hasattr(cls_obj, "get_creation_syntax")
    and hasattr(cls_obj, "get_numpy_conversion_syntax")
  )


def _extract_template(cls_obj: Any) -> Dict[str, str]:
  """Generates the dictionary structure expected by TestGenerator."""
  # We use placeholder '{np_var}' and '{res_var}' which TestGenerator expects
  return {
    "import": cls_obj.get_import_stmts(),
    "convert_input": cls_obj.get_creation_syntax("{np_var}"),
    "to_numpy": cls_obj.get_numpy_conversion_syntax("{res_var}"),
  }

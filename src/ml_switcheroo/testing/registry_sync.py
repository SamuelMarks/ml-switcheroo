"""
Sync Mechanism for Test Template Generation.

This module resolves the fragmentation between `FrameworkAdapter` classes (defined in code)
and Test Generation Templates (traditionally defined in JSON). It allows automatic
derivation of templates by introspecting registered adapters, ensuring consistent
logic for imports and tensor creation across both the `EquivalenceRunner` and `TestGenerator`.

It provides:
1.  `TemplateGenerator`: A class to build template dictionaries from Adapter metadata.
2.  `TemplateRegistry`: A Protocol for adapters to expose their import and conversion syntax.
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
        base_templates: Existing dictionary of templates (e.g. from JSON).

    Returns:
        A unified dictionary of {framework: {template_keys...}}.
    """
    # Deep copy to avoid mutating the prompt input if passed
    import copy

    templates = copy.deepcopy(base_templates) if base_templates else {}

    for fw_name, adapter_cls in _ADAPTER_REGISTRY.items():
      # If the adapter implements the TemplateProvider protocol (duck typing check)
      if _is_template_provider(adapter_cls):
        generated = _extract_template(adapter_cls)  # type: ignore

        if fw_name in templates:
          # Update source keys, effectively preserving 'jit_wrap' if present in base
          templates[fw_name].update(generated)
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

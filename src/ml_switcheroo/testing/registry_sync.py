"""
Registry Template Synchronizer.

Extracts test generation templates from registered Framework Adapters.
This links the 'Zero-Edit' extensibility of Adapters to the Validation Engine.
"""

from typing import Dict, Any

from ml_switcheroo.frameworks import available_frameworks, get_adapter


class TemplateGenerator:
  """
  Generates configuration templates for verification harnesses
  by inspecting registered adapters.
  """

  @staticmethod
  def generate_templates() -> Dict[str, Dict[str, str]]:
    """
    Iterates all registered frameworks and extracts test templates.

    Returns:
        Dict mapping framework_key -> template_dict.
        Example:
        {
            "torch": {
                "import": "import torch",
                "convert_input": "torch.tensor({np_var})",
                "to_numpy": "{res_var}.numpy()"
            }
        }
    """
    templates = {}

    for fw in available_frameworks():
      adapter = get_adapter(fw)
      if not adapter:
        continue

      # Check if adapter implements the template provider protocol methods
      if (
        hasattr(adapter, "get_import_stmts")
        and hasattr(adapter, "get_creation_syntax")
        and hasattr(adapter, "get_numpy_conversion_syntax")
      ):
        try:
          tmpl = {
            "import": adapter.get_import_stmts(),
            "convert_input": adapter.get_creation_syntax("{np_var}"),
            "to_numpy": adapter.get_numpy_conversion_syntax("{res_var}"),
          }
          templates[fw] = tmpl
        except Exception:
          # Skip if adapter fails to generate (e.g. abstract methods not impl)
          pass

    return templates

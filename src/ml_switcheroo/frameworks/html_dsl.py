"""
HTML/SVG DSL Framework Adapter.

Simplified to only provide Metadata.
"""

from typing import Dict, List, Tuple, Any, Set, Optional

from ml_switcheroo.frameworks.base import (
  register_framework,
  FrameworkAdapter,
  StandardMap,
  StandardCategory,
  GhostRef,
  ImportConfig,
  InitMode,
)
from ml_switcheroo.semantics.schema import StructuralTraits, PluginTraits, OperationDef
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.loader import load_definitions
from ml_switcheroo.core.html.parser import HtmlParser


@register_framework("html")
class HtmlDSLAdapter(FrameworkAdapter):
  """Adapter for the HTML/SVG Visual DSL."""

  display_name: str = "HTML Grid DSL"
  ui_priority: int = 980
  inherits_from: Optional[str] = None

  def __init__(self) -> None:
    """TODO: Add docstring."""
    self._mode = InitMode.GHOST
    self._snapshot_data: Dict[str, Any] = {}

  def create_parser(self, code: str) -> HtmlParser:
    """Factory for the HTML Parser used by Ingestion."""
    return HtmlParser(code)

  @property
  def search_modules(self) -> List[str]:  # pragma: no cover
    """TODO: Add docstring."""
    return []

  @property  # pragma: no cover
  def unsafe_submodules(self) -> Set[str]:
    """TODO: Add docstring."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """TODO: Add docstring."""
    return ("html_dsl", "dsl")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """TODO: Add docstring."""
    return {"html_dsl": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="dsl")}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """TODO: Add docstring."""
    return {}

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """TODO: Add docstring."""
    return [SemanticTier.NEURAL]

  @property
  def structural_traits(self) -> StructuralTraits:
    """TODO: Add docstring."""
    return StructuralTraits(
      module_base="html_dsl.Module",  # pragma: no cover
      forward_method="forward",
      init_method_name="__init__",
      requires_super_init=True,
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """TODO: Add docstring."""
    return PluginTraits()  # pragma: no cover

  @property
  def test_config(self) -> Dict[str, str]:
    """TODO: Add docstring."""
    return {}

  @property
  def harness_imports(self) -> List[str]:
    """TODO: Add docstring."""
    return []

  def get_harness_init_code(self) -> str:
    """TODO: Add docstring."""
    return ""

  def get_to_numpy_code(self) -> str:
    """TODO: Add docstring."""
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> List[str]:
    """TODO: Add docstring."""
    return []

  @property
  def rng_seed_methods(self) -> List[str]:
    """TODO: Add docstring."""
    return []

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """TODO: Add docstring."""  # pragma: no cover
    defs = load_definitions("html_dsl")
    if "Module" not in defs:
      defs["Module"] = StandardMap(api="html_dsl.Module")  # pragma: no cover
    if "Conv2d" not in defs:
      defs["Conv2d"] = StandardMap(
        api="html_dsl.Conv2d",  # pragma: no cover
        args={"in_channels": "i", "out_channels": "o", "kernel_size": "k"},
      )
    return defs  # pragma: no cover

  @property
  def specifications(self) -> Dict[str, OperationDef]:  # pragma: no cover
    """TODO: Add docstring."""
    return {}

  # pragma: no cover
  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """TODO: Add docstring."""
    return []  # pragma: no cover

  def convert(self, data: Any) -> Any:
    """TODO: Add docstring."""  # pragma: no cover
    return str(data)  # pragma: no cover

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:  # pragma: no cover
    """TODO: Add docstring."""
    return ""

  # pragma: no cover
  def get_device_check_syntax(self) -> str:
    """TODO: Add docstring."""
    return "False"  # pragma: no cover

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """TODO: Add docstring."""
    return ""  # pragma: no cover

  def get_serialization_imports(self) -> List[str]:
    """TODO: Add docstring."""
    return []  # pragma: no cover

  # pragma: no cover
  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """TODO: Add docstring."""
    return ""  # pragma: no cover

  def get_weight_conversion_imports(self) -> List[str]:
    """TODO: Add docstring."""
    return []  # pragma: no cover

  def get_weight_load_code(self, path_var: str) -> str:
    """TODO: Add docstring."""
    return "# Weights not supported in HTML mode"  # pragma: no cover

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """TODO: Add docstring."""
    return tensor_var  # pragma: no cover

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """TODO: Add docstring."""
    return "# Weights not supported in HTML mode"  # pragma: no cover

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """TODO: Add docstring."""
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """TODO: Add docstring."""
    return None

  def get_tiered_examples(self) -> Dict[str, str]:
    """TODO: Add docstring."""
    return {  # pragma: no cover
      "tier2_neural": """ 
<div class="grid">
  <div class="box r">
    <span class="header-txt">conv: Conv2d</span>
    <code>i=1, o=32, k=3</code>
  </div>
</div>
"""
    }

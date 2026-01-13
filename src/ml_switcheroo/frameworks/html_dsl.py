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
    self._mode = InitMode.GHOST
    self._snapshot_data: Dict[str, Any] = {}

  def create_parser(self, code: str) -> HtmlParser:
    """Factory for the HTML Parser used by Ingestion."""
    return HtmlParser(code)

  @property
  def search_modules(self) -> List[str]:
    return []

  @property
  def unsafe_submodules(self) -> Set[str]:
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    return ("html_dsl", "dsl")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    return {"html_dsl": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="dsl")}

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    return {}

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    return [SemanticTier.NEURAL]

  @property
  def structural_traits(self) -> StructuralTraits:
    return StructuralTraits(
      module_base="html_dsl.Module",
      forward_method="forward",
      init_method_name="__init__",
      requires_super_init=True,
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    return PluginTraits()

  @property
  def test_config(self) -> Dict[str, str]:
    return {}

  @property
  def harness_imports(self) -> List[str]:
    return []

  def get_harness_init_code(self) -> str:
    return ""

  def get_to_numpy_code(self) -> str:
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> List[str]:
    return []

  @property
  def rng_seed_methods(self) -> List[str]:
    return []

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    defs = load_definitions("html_dsl")
    if "Module" not in defs:
      defs["Module"] = StandardMap(api="html_dsl.Module")
    if "Conv2d" not in defs:
      defs["Conv2d"] = StandardMap(
        api="html_dsl.Conv2d",
        args={"in_channels": "i", "out_channels": "o", "kernel_size": "k"},
      )
    return defs

  @property
  def specifications(self) -> Dict[str, OperationDef]:
    return {}

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    return []

  def convert(self, data: Any) -> Any:
    return str(data)

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    return ""

  def get_device_check_syntax(self) -> str:
    return "False"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    return ""

  def get_serialization_imports(self) -> List[str]:
    return []

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    return []

  def get_weight_load_code(self, path_var: str) -> str:
    return "# Weights not supported in HTML mode"

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    return tensor_var

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    return "# Weights not supported in HTML mode"

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    pass

  def get_doc_url(self, api_name: str) -> Optional[str]:
    return None

  def get_tiered_examples(self) -> Dict[str, str]:
    return {
      "tier2_neural": """ 
<div class="grid">
  <div class="box r">
    <span class="header-txt">conv: Conv2d</span>
    <code>i=1, o=32, k=3</code>
  </div>
</div>
"""
    }

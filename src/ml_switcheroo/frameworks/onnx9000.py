"""ONNX9000 Framework Adapter."""

import textwrap
from typing import List, Tuple, Dict, Any, Optional, Set

try:
  from onnx9000.core import ir
except ImportError:
  ir = None

from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardMap,
  ImportConfig,
  StandardCategory,
  InitMode,
  GhostRef,
)
from ml_switcheroo.enums import SemanticTier


@register_framework("onnx9000")
class Onnx9000Framework:
  """Adapter for ONNX9000 Graph representation.

  Generates syntax that utilizes `onnx9000.core.ir.Node` to build models.
  """

  display_name: str = "ONNX9000"
  ui_priority: int = 5
  inherits_from: Optional[str] = None

  def __init__(self) -> None:
    """Initialize ONNX9000 framework adapter."""
    self._mode = InitMode.LIVE if ir is not None else InitMode.GHOST
    self._snapshot_data: Dict[str, Any] = {}

  @property
  def search_modules(self) -> List[str]:
    """Modules to search for APIs."""
    return ["onnx9000.core.ir"] if self._mode == InitMode.LIVE else []

  @property
  def unsafe_submodules(self) -> Set[str]:
    """Submodules to avoid scanning."""
    return set()

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Canonical import alias for ONNX9000."""
    return ("onnx9000", "onnx9000")

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Namespace configurations."""
    return {
      "onnx9000": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="onnx9000"),
      "onnx9000.core.ir": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="ir"),
    }

  @property
  def discovery_heuristics(self) -> Dict[str, List[str]]:
    """Heuristics to identify APIs."""
    return {"extras": [r"onnx9000\.core\.ir\.", r"ir\."]}

  @property
  def test_config(self) -> Dict[str, str]:
    """Configuration for generated tests."""
    return {
      "import": "import onnx9000\nfrom onnx9000.core import ir\nimport numpy as np",
      "convert_input": "ir.Tensor(name='{np_var}', dtype=str({np_var}.dtype), shape=list({np_var}.shape))",
      "to_numpy": "{res_var}",
      "jit_template": "{fn}",
    }

  @property
  def harness_imports(self) -> List[str]:
    """Imports for the verification harness."""
    return ["import onnx9000", "from onnx9000.core import ir"]

  def get_harness_init_code(self) -> str:
    """Initialization code for the harness."""
    return ""

  @property
  def declared_magic_args(self) -> List[str]:
    """Magic arguments used by the framework."""
    return []

  @property
  def structural_traits(self) -> StructuralTraits:
    """Structural traits describing ONNX logic."""
    return StructuralTraits(
      module_base=None,
      forward_method="make_model",
      inject_magic_args=[],
      requires_super_init=False,
      lifecycle_strip_methods=[],
      lifecycle_warn_methods=[],
      jit_static_args=[],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Plugin traits."""
    return PluginTraits(
      has_numpy_compatible_arrays=False,
      requires_explicit_rng=False,
      requires_functional_control_flow=True,
      enforce_purity_analysis=False,
      strict_materialization_method=None,
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    """Seed methods."""
    return []

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """Semantic definitions loaded from onnx9000.json."""
    # For now, return empty or fallback since we don't have onnx9000.json yet
    return {}

  @property
  def specifications(self) -> Dict[str, Any]:
    """Operation specifications."""
    return {}

  def collect_api(self, category: StandardCategory) -> List[GhostRef]:
    """Collects APIs for a category."""
    return []

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Applies wiring to a snapshot."""
    pass

  def convert(self, data: Any) -> Any:
    """Converts data to an ONNX compatible format."""
    return data

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """ONNX models are device agnostic in definition."""
    return "None"

  def get_device_check_syntax(self) -> str:
    """Checks for device availability."""
    return "True"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Splits an RNG key."""
    return ""

  def get_serialization_imports(self) -> List[str]:
    """Imports for serialization."""
    return ["import onnx9000.core.exporter"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Syntax for serializing/deserializing."""
    if op == "save":
      return f"onnx9000.core.exporter.export({object_arg}, {file_arg})"
    elif op == "load":
      return f"onnx9000.core.parser.parse({file_arg})"
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Imports for weight conversion scripts."""
    return ["from onnx9000.core import parser"]

  def get_weight_load_code(self, path_var: str) -> str:
    """Code to load weights into raw_state dict."""
    return textwrap.dedent(f"""
            model = parser.parse({path_var})
            raw_state = model.get_weights()
        """)

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Expression to convert a tensor to numpy."""
    return f"{tensor_var}.to_numpy()"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Code to save a raw_state dict to a model file."""
    return textwrap.dedent(f"""
            # Requires exporter
            import onnx9000.core.exporter as exporter
            exporter.export_weights({state_var}, {path_var})
        """)

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """URL to documentation."""
    return "https://github.com/SamuelMarks/onnx9000"

  def get_tiered_examples(self) -> Dict[str, str]:
    """Examples for the demo UI."""
    return {
      "tier1_math": textwrap.dedent("""\
                from onnx9000.core import ir
                
                # Define simple Add node
                node = ir.Node('Add', inputs=['X', 'Y'], outputs=['Z'])
            """)
    }

  def get_to_numpy_code(self) -> str:
    """Code to extract a numpy array from an object."""
    return "if hasattr(obj, 'numpy'): return obj.numpy()"

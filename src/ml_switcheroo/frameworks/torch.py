"""PyTorch Framework Adapter.

This module implements the `FrameworkAdapter` protocol for PyTorch.
It provides:

1.  **Import Abstraction**: Self-declared namespace mappings (e.g., `torch.nn` is `NEURAL`).
2.  **Semantic Definitions**: Mappings loaded from `definitions/torch.json` via helper.
3.  **Discovery**: Heuristics and logic for scanning the installed `torch` library.
4.  **IO & Device Support**: Wires up serialization and device allocation.
5.  **Weight Migration**: Implements logic to generate scripts for converting .pth checkpoints
    to/from NumPy format for interoperability.
"""

import logging
import textwrap
from typing import List, Tuple, Dict, Any, Optional

try:
  import torch
  import torch.nn as nn
  import torch.optim as optim
except Exception:
  torch = None
  nn = None
  optim = None
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import (
  register_framework,
  StructuralTraits,
  PluginTraits,
  StandardMap,
  ImportConfig,
  InitMode,
  GhostRef,
  load_snapshot_for_adapter,
)
from ml_switcheroo.frameworks.loader import load_definitions


@register_framework("torch")
class TorchAdapter:
  """Adapter for PyTorch (Meta).

  Handles Source and Target translation rules for PyTorch, including
  `torch.nn`, `torch.optim`, and `torch.func` (vmap/grad).
  """

  display_name: str = "PyTorch"
  inherits_from: Optional[str] = None
  ui_priority: int = 0

  def __init__(self) -> None:
    """Initializes the adapter.

    Detects if PyTorch is installed. to switch between LIVE inspection
    and GHOST snapshot loading.
    """
    self._mode = InitMode.LIVE
    self._snapshot_data: Dict[str, Any] = {}
    if torch is None:
      self._mode = InitMode.GHOST
      self._snapshot_data = load_snapshot_for_adapter("torch")
      if not self._snapshot_data:
        logging.warning("PyTorch not installed and no snapshot found. Scanning unavailable.")

  @property
  def import_alias(self) -> Tuple[str, str]:
    """Returns the primary root import alias ('torch', 'torch').

    Returns:
        The module name and default alias.

    """
    return "torch", "torch"

  @property
  def import_namespaces(self) -> Dict[str, ImportConfig]:
    """Defines the semantic roles of PyTorch namespaces.

    Returns:
        Mapping of dot-path strings to configuration objects.

    """
    return {
      "torch": ImportConfig(tier=SemanticTier.ARRAY_API, recommended_alias="torch"),
      "torch.nn": ImportConfig(tier=SemanticTier.NEURAL, recommended_alias="nn"),
      "torch.nn.functional": ImportConfig(tier=SemanticTier.NEURAL_OPS, recommended_alias="F"),
      "torch.optim": ImportConfig(tier=SemanticTier.EXTRAS, recommended_alias="optim"),
      "torch.utils.data": ImportConfig(tier=SemanticTier.EXTRAS),
    }

  @property
  def supported_tiers(self) -> List[SemanticTier]:
    """Returns the semantic tiers fully supported by this adapter.

    Returns:
        List of supported tiers.

    """
    return [SemanticTier.NEURAL, SemanticTier.ARRAY_API, SemanticTier.EXTRAS]

  @property
  def test_config(self) -> Dict[str, str]:
    """Templates used by `gen-tests` to create physical verification files.

    Returns:
        Dictionary of code templates.

    """
    return {
      "import": "import torch",
      "convert_input": "torch.tensor({np_var})",
      "to_numpy": "{res_var}.detach().cpu().numpy()",
    }

  @property
  def harness_imports(self) -> List[str]:
    """Imports required for harness initialization.

    Returns:
        List of import statements.

    """
    return []

  def get_harness_init_code(self) -> str:
    """Returns helper code for initializing the harness.

    Returns:
        Python source code string.

    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Returns code to convert Torch tensors to NumPy (detach/cpu check).

    Returns:
        Python statement string.

    """
    return "if hasattr(obj, 'detach'): return obj.detach().cpu().numpy()"

  @property
  def structural_traits(self) -> StructuralTraits:
    """Defines how classes and functions are rewritten when targeting PyTorch.

    Returns:
        Configuration object for structural rewriting.

    """
    return StructuralTraits(
      module_base="torch.nn.Module",
      forward_method="forward",
      requires_super_init=True,
      auto_strip_magic_args=True,
      lifecycle_strip_methods=["to", "cpu", "cuda", "detach", "clone", "requires_grad_", "share_memory_"],
      lifecycle_warn_methods=["eval", "train"],
      impurity_methods=["add_", "sub_", "mul_", "div_", "pow_", "zero_", "copy_", "fill_"],
      jit_static_args=[],
      implicit_method_roots=["torch.Tensor"],
    )

  @property
  def plugin_traits(self) -> PluginTraits:
    """Capabilities flags. PyTorch uses imperative state and eager execution.

    Returns:
        Configuration object for plugin logic.

    """
    return PluginTraits(
      has_numpy_compatible_arrays=False,
      requires_explicit_rng=False,
      requires_functional_state=False,
      requires_functional_control_flow=False,
    )

  @property
  def rng_seed_methods(self) -> List[str]:
    """Global seed setting methods detected as impure side-effects.

    Returns:
        List of method names.

    """
    return ["manual_seed", "seed"]

  @property
  def declared_magic_args(self) -> List[str]:
    """Returns list of framework-specific magic arguments.
    Torch emits no magic args; all state is implicit.

    Returns:
        Empty list.

    """
    return []

  @property
  def definitions(self) -> Dict[str, StandardMap]:
    """The definitive mapping of Abstract Operations to PyTorch APIs.
    Loaded dynamically from `frameworks/definitions/torch.json`.

    Returns:
        Dictionary mapping operation abstract IDs to implementation details.

    """
    defs = load_definitions("torch")
    if "ReLU" not in defs:
      defs["ReLU"] = StandardMap(api="torch.nn.ReLU")
    if "relu" not in defs:
      defs["relu"] = StandardMap(api="torch.nn.functional.relu")
    if "Linear" not in defs:
      defs["Linear"] = StandardMap(
        api="torch.nn.Linear", args={"in_features": "in_features", "out_features": "out_features"}
      )
    if "Conv2d" not in defs:
      defs["Conv2d"] = StandardMap(
        api="torch.nn.Conv2d",
        args={"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size"},
      )
    return defs

  def get_device_syntax(self, device_type: str, device_index: Optional[str] = None) -> str:
    """Generates code for device creation.

    Args:
        device_type: The device type string (e.g. 'cuda', 'cpu').
        device_index: The optional device index.

    Returns:
        Code string for device creation.

    """
    args = [str(device_type)]
    if device_index:
      args.append(str(device_index))
    arg_str = ", ".join(args)
    return f"torch.device({arg_str})"

  def get_device_check_syntax(self) -> str:
    """Returns PyTorch syntax for checking CUDA availability.

    Returns:
        Python expression string.

    """
    return "torch.cuda.is_available()"

  def get_rng_split_syntax(self, rng_var: str, key_var: str) -> str:
    """Returns syntax for splitting RNG state.
    PyTorch uses global state-based randomness, so explicit splitting is a no-op.

    Args:
        rng_var: Variable name holding random state.
        key_var: Variable name for the new key.

    Returns:
        'pass' string (No-op).

    """
    return "pass"

  def get_serialization_imports(self) -> List[str]:
    """Returns imports required for IO operations.

    Returns:
        List of import statements.

    """
    return ["import torch"]

  def get_serialization_syntax(self, op: str, file_arg: str, object_arg: Optional[str] = None) -> str:
    """Generates save/load syntax.

    Args:
        op: Operation name ('save' or 'load').
        file_arg: Code string representing the file path.
        object_arg: Code string representing the object to save (optional).

    Returns:
        Python code string for the operation.

    """
    if op == "save" and object_arg:
      return f"torch.save({object_arg}, {file_arg})"
    elif op == "load":
      return f"torch.load({file_arg})"
    return ""

  def get_weight_conversion_imports(self) -> List[str]:
    """Returns imports required for the generated weight migration script logic.

    Returns:
        List of import strings.

    """
    return ["import torch"]

  def get_weight_load_code(self, path_var: str) -> str:
    """Returns Python code to load a .pth file into a raw state dictionary.
    Handles both bare state dicts and Lightning-style checkpoints.

    Args:
        path_var: Variable name containing the file path string.

    Returns:
        Block of python code setting 'raw_state'.

    """
    return textwrap.dedent(
      f""" 
            # Load PyTorch checkpoint to CPU to avoid CUDA dependency
            loaded = torch.load({path_var}, map_location='cpu') 
            
            # Unwrap common checkpoint formats
            if isinstance(loaded, dict) and 'state_dict' in loaded: 
                raw_state = loaded['state_dict'] 
            else: 
                raw_state = loaded
            
            if not isinstance(raw_state, dict): 
                raise ValueError(f"Expected dict-like checkpoint, got {{type(loaded)}}") 
            """
    )

  def get_tensor_to_numpy_expr(self, tensor_var: str) -> str:
    """Returns expression to convert a Torch tensor variable to a NumPy array.
    Includes detach and cpu calls for safety.

    Args:
        tensor_var: Name of variable holding the torch tensor.

    Returns:
        Expression string.

    """
    return f"{tensor_var}.detach().cpu().numpy()"

  def get_weight_save_code(self, state_var: str, path_var: str) -> str:
    """Returns Python code to save the converted state dictionary back to .pth format.
    Converts NumPy arrays back to Torch tensors before saving.

    Args:
        state_var: Variable name of the flat dictionary {key: numpy_array}.
        path_var: Variable name of the output path.

    Returns:
        Block of python code.

    """
    return textwrap.dedent(
      f""" 
            # Convert NumPy arrays back to Torch Tensors
            torch_state = {{k: torch.from_numpy(v) for k, v in {state_var}.items()}} 
            torch.save(torch_state, {path_var}) 
            """
    )

  def get_doc_url(self, api_name: str) -> Optional[str]:
    """Returns the official PyTorch documentation URL.

    Args:
        api_name: The fully qualified API name.

    Returns:
        URL string or None.

    """
    if "nn.init" in api_name:
      return f"https://pytorch.org/docs/stable/nn.init.html#{api_name}"
    return f"https://pytorch.org/docs/stable/generated/{api_name}.html"

  def get_tiered_examples(self) -> Dict[str, str]:
    """Provides code snippets for "Wizard" or "Demo" usage.

    Returns:
        Dictionary mapping tier IDs to code snippets.

    """
    return {
      "tier1_math": """import torch

def math_ops(x, y):
    # Tier 1: Core Tensor Operations
    a = torch.abs(x)
    b = torch.add(a, y)
    
    # Reduction
    return torch.mean(b)
""",
      "tier2_neural_simple": """import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        
    def forward(self, x):
        x = self.fc(x)
        return nn.functional.relu(x)
""",
      "tier2_neural_cnn": """import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
""",
      "tier3_extras_dataloader": """import torch
from torch.utils.data import DataLoader, TensorDataset

def create_loader(data, targets):
    # Tier 3: Data Loader
    ds = TensorDataset(data, targets)
    return DataLoader(ds, batch_size=32, num_workers=4)
""",
      "tier4_qwen3": """import torch
import torch.nn as nn

class QwenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard HF-style separate projections
        self.q_proj = nn.Linear(1024, 1024)
        self.k_proj = nn.Linear(1024, 1024)
        self.v_proj = nn.Linear(1024, 1024)
        
        self.gate_proj = nn.Linear(1024, 4096)
        self.up_proj = nn.Linear(1024, 4096)
        self.down_proj = nn.Linear(4096, 1024)
        
    def forward(self, x):
        # Attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # SwiGLU MLP
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Note: switcheroo handles the fusion with SwiGLU
        mlp_out = self.down_proj(gate * up) 
        
        return q, mlp_out
""",
      "tier4_qwen3-vl": """import torch
import torch.nn as nn

class Qwen3VLVisionConfig:
    in_channels: int = 3
    hidden_size: int = 1280
    temporal_patch_size: int = 2
    patch_size: int = 14

class Qwen3VLPatchEmbed(nn.Module):
    '''3D Convolutional patch embedding for vision input.'''
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.config = config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            padding=0,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        seq_len = hidden_states.shape[0]

        hidden_states = hidden_states.reshape(
            seq_len, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size
        )
        
        out = self.proj(hidden_states)
        
        return out.reshape(seq_len, cfg.hidden_size)
""",
    }

  def convert(self, data: Any) -> Any:
    """Converts input data (numpy, lists) into PyTorch Tensors for verification runners.

    Args:
        data: Input data structure.

    Returns:
        Converted PyTorch Tensor or original data if conversion fails.

    """
    try:
      import torch
      import numpy as np
    except Exception:
      return data
    if isinstance(data, (np.ndarray, np.generic)):
      try:
        return torch.from_numpy(data)
      except Exception:
        return torch.tensor(data)
    if isinstance(data, (list, tuple)):
      try:
        return torch.tensor(data)
      except Exception:
        pass
    return data

  def _collect_ghost(self, category: SemanticTier) -> List[GhostRef]:
    """Loads definitions from JSON snapshot.

    Args:
        category: The standard category to filter by.

    Returns:
        List of hydrated GhostRef objects.

    """
    if not self._snapshot_data:
      return []
    raw_list = self._snapshot_data.get("categories", {}).get(category.value, [])
    return [GhostRef.model_validate(item) for item in raw_list]

  def _collect_live(self, category: SemanticTier) -> List[GhostRef]:
    """Introspects live torch modules.

    Args:
        category: The standard category to filter by.

    Returns:
        List of discovered GhostRef objects.

    """
    results = []
    if category == SemanticTier.LOSS:
      results.extend(self._scan_losses())
    elif category == SemanticTier.OPTIMIZER:
      results.extend(self._scan_optimizers())
    elif category == SemanticTier.ACTIVATION:
      results.extend(self._scan_activations())
    elif category == SemanticTier.LAYER:
      results.extend(self._scan_layers())
    return results

  def apply_wiring(self, snapshot: Dict[str, Any]) -> None:
    """Apply manual patches to the standard mappings if necessary.
    Used to inject complex behaviors not captured by simple API scanning.

    Args:
        snapshot: The snapshot dictionary to update in-place.

    """
    snapshot.setdefault("mappings", {})

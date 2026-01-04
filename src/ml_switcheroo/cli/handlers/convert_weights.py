"""
Weight Migration Script Generation.

This module provides the `WeightScriptGenerator`, a utility that creates
standalone Python scripts for migrating model checkpoints (weights) between
deep learning frameworks.

It analyzes the source AST to identify model layers, queries the Semantic
Knowledge Base for parameter mapping rules (e.g., ``weight`` -> ``kernel``),
and determines tensor layout permutations (e.g., NCHW -> NHWC).

Supported Directions:
- PyTorch -> JAX (Flax)
- JAX (Flax) -> PyTorch

Unsupported combinations degrade gracefully by returning False.
"""

import textwrap
import pprint
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import libcst as cst
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.calls.utils import compute_permutation
from ml_switcheroo.utils.console import log_error, log_success, log_warning
from ml_switcheroo.frameworks import get_adapter


class WeightScriptGenerator:
  """
  Generates a Python script to migrate weights between frameworks.

  Attributes:
      semantics (SemanticsManager): The knowledge base manager.
      source_fw (str): The source framework key (e.g. 'torch').
      target_fw (str): The target framework key (e.g. 'jax').
  """

  def __init__(self, semantics: SemanticsManager, config: RuntimeConfig):
    """
    Initializes the generator.

    Args:
        semantics: The loaded semantics manager.
        config: Runtime configuration defining source and target frameworks.
    """
    self.semantics = semantics
    self.source_fw = config.effective_source
    self.target_fw = config.effective_target
    self.source_adapter = get_adapter(self.source_fw)
    self.target_adapter = get_adapter(self.target_fw)

  def generate(self, source_path: Path, output_script: Path) -> bool:
    """
    Main entry point to generate the migration script.

    Args:
        source_path: Path to the Python file containing the source model definition.
        output_script: Path where the generated script will be written.

    Returns:
        bool: True if generation was successful.
    """
    if not self.source_adapter or not self.target_adapter:
      log_error(f"Adapters not found for {self.source_fw} or {self.target_fw}")
      return False

    try:
      code = source_path.read_text(encoding="utf-8")
    except Exception as e:
      log_error(f"Failed to read source file: {e}")
      return False

    # 1. Extract Model Architecture
    try:
      tree = cst.parse_module(code)
      extractor = GraphExtractor()
      tree.visit(extractor)

      if not extractor.layer_registry:
        log_error("No layers detected in source file. Ensure it contains a class with layers in __init__.")
        return False

    except Exception as e:
      log_error(f"Failed to parse source AST: {e}")
      return False

    # 2. Build Mapping Rules
    rules = self._flatten_mapping_rules(extractor.layer_registry)

    # 3. Emit Script Code via Adapter delegation
    script_content = self._generate_script(rules)

    # 4. Write to file
    try:
      output_script.write_text(script_content, encoding="utf-8")
      log_success(f"Generated weight migration script at [path]{output_script}[/path]")
      return True
    except Exception as e:
      log_error(f"Failed to write output script: {e}")
      return False

  def _flatten_mapping_rules(self, layer_registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs mapping rules for each layer found in the AST.

    Arg:
        layer_registry: Dictionary of LogicalNodes extracted from source AST.

    Returns:
        List[Dict]: A list of rule dictionaries containing keys/permutations.
    """
    rules = []
    # Check source type for layout logic priority. Torch layout is NCHW. JAX/TF/MLX is often NHWC.
    # layout_map is defined relative to Hub.
    # If Source -> Hub (Permute A) -> Target (Permute B).
    # Typically layout_map is on the JAX/TF side (Target).

    # Assume Torch is Source (Standard Layout)
    is_torch_src = "torch" in self.source_fw

    abstract_params = ["weight", "bias", "running_mean", "running_var", "scale"]

    for layer_name, node in layer_registry.items():
      op_kind = node.kind  # e.g. "Conv2d"

      # Lookup Semantics
      definition = self.semantics.get_definition(op_kind)
      if not definition:
        definition = self.semantics.get_definition(f"{self.source_fw}.nn.{op_kind}")

      if not definition:
        continue

      _, details = definition
      variants = details.get("variants", {})

      src_variant = variants.get(self.source_fw) or {}
      tgt_variant = variants.get(self.target_fw) or {}

      src_args_map = src_variant.get("args", {}) or {}
      tgt_args_map = tgt_variant.get("args", {}) or {}
      tgt_layout = tgt_variant.get("layout_map", {}) or {}

      for p_name in abstract_params:
        # 1. Determine Source Key Suffix
        # If source defines mapping, use it. Else assume identity.
        src_suffix = src_args_map.get(p_name, p_name)

        # 2. Determine Target Key Suffix
        tgt_suffix = tgt_args_map.get(p_name, p_name)

        # 3. Compute Permutation
        perm = None

        # Simplified logic: If Target defines Layout Map relative to Standard, and Source IS Standard (Torch-like), use it.
        # If switching direction, invert it.
        if p_name in tgt_layout:
          rule = tgt_layout[p_name]  # e.g. "OIHW->HWIO"
          if "->" in rule:
            fmt_in, fmt_out = rule.split("->")
            if is_torch_src:
              perm = compute_permutation(fmt_in.strip(), fmt_out.strip())
            else:
              # Going HWIO -> OIHW
              perm = compute_permutation(fmt_out.strip(), fmt_in.strip())

        rules.append(
          {
            "layer": layer_name,
            "src_suffix": src_suffix,
            "tgt_suffix": tgt_suffix,
            # We use dot notation for intermediate representation
            "src_key": f"{layer_name}.{src_suffix}",
            "tgt_key": f"{layer_name}.{tgt_suffix}",
            "perm": perm,
          }
        )

    return rules

  def _generate_script(self, rules: List[Dict[str, Any]]) -> str:
    """
    Generates the migration script using Adapter primitives.
    """
    src_imports = "\n".join(self.source_adapter.get_weight_conversion_imports())
    tgt_imports = "\n".join(self.target_adapter.get_weight_conversion_imports())

    load_code = textwrap.indent(self.source_adapter.get_weight_load_code("input_path"), "    ")
    save_code = textwrap.indent(self.target_adapter.get_weight_save_code("converted_state", "output_path"), "    ")
    to_numpy_expr = self.source_adapter.get_tensor_to_numpy_expr("raw_val")

    rules_repr = pprint.pformat(rules, indent=4, width=100)

    return textwrap.dedent(f"""
import sys
import numpy as np
{src_imports}
{tgt_imports}

MAPPING_RULES = {rules_repr}

def permute(arr, p):
    if p: return np.transpose(arr, p)
    return arr

def migrate(input_path, output_path):
    print(f"Loading {{input_path}}...")
{load_code}

    converted_state = {{}}
    print("Transforming weights...")

    for rule in MAPPING_RULES:
        src_key = rule["src_key"]
        
        # Checking flat keys
        if src_key not in raw_state:
            print(f"⚠️  Missing source key: {{src_key}}")
            continue
            
        raw_val = raw_state[src_key]
        
        # Convert to Numpy
        np_val = {to_numpy_expr}
        
        # Permute if needed
        data = permute(np_val, rule["perm"])
        
        # Store for saving using target key string
        converted_state[rule["tgt_key"]] = data
        print(f"✔ {{src_key}} -> {{rule['tgt_key']}} (Shape: {{data.shape}})")

    print(f"Saving to {{output_path}}...")
{save_code}
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weight Migration Script")
    parser.add_argument("input", help="Input checkpoint path")
    parser.add_argument("output", help="Output checkpoint path")
    args = parser.parse_args()
    migrate(args.input, args.output)
""").strip()

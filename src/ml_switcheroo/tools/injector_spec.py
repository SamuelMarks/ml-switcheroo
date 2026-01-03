"""
LibCST Transformer for Injecting Specifications.

This tool modifies Python source files (specifically `standards_internal.py`)
by locating the `INTERNAL_OPS` dictionary and appending a new operation definition
derived from an ODL (Operation Definition Language) model.

Updates:
- Supports persistent semantic constraints (min, max, options).
- Checks for duplicate keys to prevent overwriting existing definitions.
- Uses robust string literal generation.
"""

import libcst as cst
from typing import Any
from ml_switcheroo.core.dsl import OperationDef, ParameterDef
from ml_switcheroo.tools.injector_fw.utils import convert_to_cst_literal


class StandardsInjector(cst.CSTTransformer):
  """
  Injects a new operation definition into the INTERNAL_OPS dict.

  It transforms the abstract `OperationDef` into a concrete LibCST Dictionary
  node structure, including rich parameter metadata (min, max, options) and appends
  it to the existing dictionary in the source.
  """

  def __init__(self, op_def: OperationDef):
    """
    Initializes the injector.

    Args:
        op_def: The definition model containing metadata and signatures.
    """
    self.op_def = op_def
    self.found = False

  def _build_param_dict(self, param: ParameterDef) -> cst.Dict:
    """
    Constructs a CST Dictionary node representing a parameter definition.

    Result format: `{'name': 'x', 'type': 'int', 'min': 0, 'options': [1, 2]}`
    """
    # We assume clean dictionary of properties
    data = {
      "name": param.name,
      "type": param.type,
      "default": param.default,
      "min": param.min,
      "max": param.max,
      "options": param.options,
      "rank": param.rank,
      "dtype": param.dtype,
      "kind": param.kind if param.kind != "positional_or_keyword" else None,
      "is_variadic": param.is_variadic if param.is_variadic else None,
    }

    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}

    # Use utility converter for the dict
    return convert_to_cst_literal(data)

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """
    Visits assignment statements to find `INTERNAL_OPS`.

    If found, modifies the dictionary value to include the new key-value pair,
    UNLESS the key already exists.

    Args:
        original_node: The node before transformation.
        updated_node: The node after transformation.

    Returns:
        cst.Assign: The modified assignment node.
    """
    # Ensure we are modifying "INTERNAL_OPS = { ... }"
    if len(original_node.targets) != 1:
      return updated_node

    target = original_node.targets[0].target
    if not isinstance(target, cst.Name) or target.value != "INTERNAL_OPS":
      return updated_node

    if not isinstance(updated_node.value, cst.Dict):
      return updated_node

    self.found = True

    # 1. Existence Check
    # Iterate existing elements to check keys
    for element in updated_node.value.elements:
      if isinstance(element, cst.DictElement):
        # Evaluate key literal if simple string
        if isinstance(element.key, cst.SimpleString):
          # Strip quotes roughly to check value
          key_val = element.key.value.strip("'\"")
          if key_val == self.op_def.operation:
            # Operation Exists: Skip injection to preserve manual edits/ordering
            return updated_node

    # 2. Build std_args list of Dict elements
    args_elements = []
    for arg in self.op_def.std_args:
      # If arg is simple string (legacy/user yaml), handle it
      if isinstance(arg, str):
        p_def = ParameterDef(name=arg)
        param_node = self._build_param_dict(p_def)
      elif isinstance(arg, (list, tuple)):
        p_def = ParameterDef(name=arg[0], type=arg[1] if len(arg) > 1 else "Any")
        param_node = self._build_param_dict(p_def)
      elif isinstance(arg, dict):
        # Assuming validated by Pydantic loading before
        p_model = ParameterDef(**arg)
        param_node = self._build_param_dict(p_model)
      else:
        # Fallback for ParameterDef objects directly
        param_node = self._build_param_dict(arg)

      args_elements.append(cst.Element(value=param_node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))

    # 3. Build the main operation Dictionary values
    # Structure: { 'description': ..., 'std_args': [...], 'variants': {} }
    dict_body = {
      "description": self.op_def.description,
      "std_args": cst.List(args_elements),  # Use pre-built list node for complex args
      "variants": {},
    }

    # We manually construct this dict to inject the special cst.List node
    dict_elements = []

    # Description
    dict_elements.append(
      cst.DictElement(
        key=convert_to_cst_literal("description"),
        value=convert_to_cst_literal(self.op_def.description),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

    # Std Args
    dict_elements.append(
      cst.DictElement(
        key=convert_to_cst_literal("std_args"),
        value=cst.List(args_elements),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

    # Variants
    dict_elements.append(
      cst.DictElement(
        key=convert_to_cst_literal("variants"),
        value=cst.Dict([]),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
    )

    new_entry_val = cst.Dict(dict_elements)
    new_entry_key = convert_to_cst_literal(self.op_def.operation)

    # 4. Append to existing dictionary elements
    new_elements = list(updated_node.value.elements)

    # Formatting
    new_dict_item = cst.DictElement(
      key=new_entry_key,
      value=new_entry_val,
      comma=cst.Comma(
        whitespace_after=cst.ParenthesizedWhitespace(
          first_line=cst.TrailingWhitespace(newline=cst.Newline()), indent=True
        )
      ),
    )

    new_elements.append(new_dict_item)

    new_dict = updated_node.value.with_changes(elements=new_elements)
    return updated_node.with_changes(value=new_dict)

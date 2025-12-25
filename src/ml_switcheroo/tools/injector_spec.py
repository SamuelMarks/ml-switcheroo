"""
LibCST Transformer for Injecting Specifications.

This tool modifies Python source files (specifically `standards_internal.py`)
by locating the `INTERNAL_OPS` dictionary and appending a new operation definition
derived from an ODL (Operation Definition Language) model.

Update: Now supports persisting semantic constraints (min, max, options) into
the dictionary structure.
"""

import libcst as cst
from typing import Any
from ml_switcheroo.core.dsl import OperationDef, ParameterDef


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

    Args:
        param: Validated parameter data.

    Returns:
        cst.Dict: A CST node structure representing the python dict.
    """
    elements = [
      cst.DictElement(
        cst.SimpleString("'name'"),
        cst.SimpleString(f"'{param.name}'"),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      )
    ]

    if param.type and param.type != "Any":
      elements.append(
        cst.DictElement(
          cst.SimpleString("'type'"),
          cst.SimpleString(f"'{param.type}'"),
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    if param.default is not None:
      # We store the default value as a string code representation in the dict.
      # e.g. default="-1" becomes 'default': '-1' in the python dict.
      elements.append(
        cst.DictElement(
          cst.SimpleString("'default'"),
          cst.SimpleString(f"'{param.default}'"),
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # --- Constraints Injection ---

    if param.min is not None:
      elements.append(
        cst.DictElement(
          cst.SimpleString("'min'"),
          cst.Float(str(param.min)),
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    if param.max is not None:
      elements.append(
        cst.DictElement(
          cst.SimpleString("'max'"),
          cst.Float(str(param.max)),
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    if param.options is not None:
      # Convert list of options to CST list node
      list_elements = []
      for opt in param.options:
        val_node = self._convert_literal(opt)
        list_elements.append(cst.Element(value=val_node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))

      # Clean last comma is optional in lists but clean style
      if list_elements:
        list_elements[-1] = list_elements[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

      elements.append(
        cst.DictElement(
          cst.SimpleString("'options'"),
          cst.List(list_elements),
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    return cst.Dict(elements)

  def _convert_literal(self, val: Any) -> cst.BaseExpression:
    if isinstance(val, bool):
      return cst.Name("True") if val else cst.Name("False")
    if isinstance(val, int):
      return cst.Integer(str(val))
    if isinstance(val, float):
      return cst.Float(str(val))
    if isinstance(val, str):
      return cst.SimpleString(f"'{val}'")
    return cst.SimpleString(f"'{str(val)}'")

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """
    Visits assignment statements to find `INTERNAL_OPS`.

    If found, modifies the dictionary value to include the new key-value pair.

    Args:
        original_node: The node before transformation.
        updated_node: The node after transformation (containing any recursive changes).

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

    # 1. Build std_args list of Dict elements
    args_elements = []
    for arg in self.op_def.std_args:
      param_node = self._build_param_dict(arg)
      args_elements.append(cst.Element(value=param_node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))

    # 2. Build the main operation Dictionary values
    # Structure: { 'description': ..., 'std_args': [...], 'variants': {} }
    # Note: Variants are empty in the Hub spec; they reside in frameworks (Spokes).
    dict_body = [
      cst.DictElement(
        cst.SimpleString("'description'"),
        cst.SimpleString(f"'{self.op_def.description}'"),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      ),
      cst.DictElement(
        cst.SimpleString("'std_args'"),
        cst.List(args_elements),
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      ),
      cst.DictElement(
        cst.SimpleString("'variants'"),
        cst.Dict([]),  # Variants are handled by the FW injector
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
      ),
    ]

    new_entry_val = cst.Dict(dict_body)
    new_entry_key = cst.SimpleString(f"'{self.op_def.operation}'")

    # 3. Append to existing dictionary elements
    new_elements = list(updated_node.value.elements)

    # Use formatting to ensure readable indentation on the new line
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

    # Reconstruct the dictionary
    new_dict = updated_node.value.with_changes(elements=new_elements)
    return updated_node.with_changes(value=new_dict)

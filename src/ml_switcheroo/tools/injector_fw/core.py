"""
Core Logic for Framework Adapter AST Transformer.

This module defines the `FrameworkInjector` class, which handles the localized
insertion of new Semantic Operations into existing Python framework adapter files.
"""

from typing import List, Optional, Set, Union

import libcst as cst

from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.tools.injector_fw.utils import (
  convert_to_cst_literal,
  get_import_root,
  is_docstring,
  is_future_import,
)


class FrameworkInjector(cst.CSTTransformer):
  """
  Injects a `StandardMap` entry into a Framework Adapter's definitions.

  It performs a targeted search for:
  1. A class decorated with `@register_framework("target_fw")`.
  2. A method named `definitions` decorated with `@property`.
  3. A `return` statement returning a `Dict`.

  Checks if key exists before injecting.
  """

  def __init__(self, target_fw: str, op_name: str, variant: FrameworkVariant):
    """
    Initializes the injector.

    Args:
        target_fw: Key string to identify the adapter class (e.g. "torch").
        op_name: Name of the operation key to insert.
        variant: Implementation details to construct the StandardMap.
    """
    self.target_fw = target_fw
    self.op_name = op_name
    self.variant = variant
    self.found = False

    self._in_target_class = False
    self._in_definitions_prop = False
    self._required_roots: Set[str] = set()
    self._existing_roots: Set[str] = set()

    # Collect explicit import requirements
    if self.variant.required_imports:
      for item in self.variant.required_imports:
        root = ""
        if hasattr(item, "module"):
          root = item.module.split(".")[0]
        elif isinstance(item, str):
          clean = item.strip()
          if clean.startswith("import "):
            # import numpy as np
            remainder = clean[7:].strip()
            if " as " in remainder:
              root = remainder.split(" as ")[0].split(".")[0]
            else:
              root = remainder.split(".")[0]
          elif clean.startswith("from "):
            # from X import Y
            remainder = clean[5:].strip()
            root = remainder.split(" ")[0].split(".")[0]

        if root:
          self._required_roots.add(root)

  def visit_Import(self, node: cst.Import) -> None:
    for alias in node.names:
      root = get_import_root(alias.name)
      self._existing_roots.add(root)

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    if node.module:
      root = get_import_root(node.module)
      self._existing_roots.add(root)

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    for decorator in node.decorators:
      if isinstance(decorator.decorator, cst.Call):
        func = decorator.decorator.func
        if isinstance(func, cst.Name) and func.value == "register_framework":
          args = decorator.decorator.args
          if args:
            first_arg = args[0].value
            if isinstance(first_arg, cst.SimpleString):
              val = first_arg.value.strip("'").strip('"')
              if val == self.target_fw:
                self._in_target_class = True
                return True
    return False

  def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
    self._in_target_class = False
    return updated_node

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    if self._in_target_class and node.name.value == "definitions":
      is_property = any(isinstance(d.decorator, cst.Name) and d.decorator.value == "property" for d in node.decorators)
      if is_property:
        self._in_definitions_prop = True
        return True
    return False

  def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
    self._in_definitions_prop = False
    return updated_node

  def leave_Return(self, original_node: cst.Return, updated_node: cst.Return) -> Union[cst.Return, cst.RemovalSentinel]:
    if self._in_definitions_prop and isinstance(updated_node.value, cst.Dict):
      # Existence check logic
      for element in updated_node.value.elements:
        if isinstance(element, cst.DictElement) and isinstance(element.key, cst.SimpleString):
          current_key = element.key.value.strip("'\"")
          if current_key == self.op_name:
            # Already exists, do not modify
            # Mark found so that we know logic was reached
            self.found = True
            return updated_node

      self.found = True
      new_key = convert_to_cst_literal(self.op_name)
      new_value = self._build_standard_map_call()

      new_element = cst.DictElement(
        key=new_key,
        value=new_value,
        comma=cst.Comma(
          whitespace_after=cst.ParenthesizedWhitespace(
            first_line=cst.TrailingWhitespace(newline=cst.Newline()),
            indent=True,
            last_line=cst.SimpleWhitespace("    " * 3),
          )
        ),
      )

      new_elements = list(updated_node.value.elements)
      new_elements.append(new_element)
      new_dict = updated_node.value.with_changes(elements=new_elements)
      return updated_node.with_changes(value=new_dict)

    return updated_node

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    missing_roots = self._required_roots - self._existing_roots
    if not missing_roots:
      return updated_node

    body = list(updated_node.body)
    insert_idx = 0
    for i, stmt in enumerate(body):
      if is_docstring(stmt, i) or is_future_import(stmt):
        insert_idx = i + 1
        continue
      break

    new_imports = []
    for root in sorted(missing_roots):
      if not root.isidentifier():
        continue
      imp_node = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name(root))])])
      new_imports.append(imp_node)

    new_body = body[:insert_idx] + new_imports + body[insert_idx:]
    return updated_node.with_changes(body=new_body)

  def _build_standard_map_call(self) -> cst.Call:
    tight_eq = cst.AssignEqual(
      whitespace_before=cst.SimpleWhitespace(""),
      whitespace_after=cst.SimpleWhitespace(""),
    )

    args_list: List[cst.Arg] = []

    # We construct the call arguments by inspecting all fields on the Pydantic model
    # excluding fields that are None or default empty

    # 1. API
    if self.variant.api:
      args_list.append(
        cst.Arg(
          keyword=cst.Name("api"),
          value=convert_to_cst_literal(self.variant.api),
          equal=tight_eq,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # 2. Dictionaries (args, inject_args, arg_values, casts, layout_map)
    for field in ["args", "inject_args", "arg_values", "casts", "layout_map"]:
      val = getattr(self.variant, field, None)
      if val:
        args_list.append(
          cst.Arg(
            keyword=cst.Name(field),
            value=convert_to_cst_literal(val),
            equal=tight_eq,
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
          )
        )

    # 3. Strings/Optionals
    for field in [
      "requires_plugin",
      "transformation_type",
      "operator",
      "macro_template",
      "output_cast",
      "output_adapter",
      "pack_to_tuple",
      "missing_message",
    ]:
      val = getattr(self.variant, field, None)
      if val:
        args_list.append(
          cst.Arg(
            keyword=cst.Name(field),
            value=convert_to_cst_literal(val),
            equal=tight_eq,
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
          )
        )

    # 4. Integers
    if self.variant.output_select_index is not None:
      args_list.append(
        cst.Arg(
          keyword=cst.Name("output_select_index"),
          value=convert_to_cst_literal(self.variant.output_select_index),
          equal=tight_eq,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # 5. Required Imports (List)
    if self.variant.required_imports:
      # Convert models to dicts if needed
      items = []
      for item in self.variant.required_imports:
        if hasattr(item, "model_dump"):
          items.append(item.model_dump())
        else:
          items.append(item)

      args_list.append(
        cst.Arg(
          keyword=cst.Name("required_imports"),
          value=convert_to_cst_literal(items),
          equal=tight_eq,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    if args_list:
      last = args_list[-1]
      args_list[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return cst.Call(func=cst.Name("StandardMap"), args=args_list)

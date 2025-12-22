"""
LibCST Transformer for Injecting Framework Mappings.

This module provides the logic to modify framework adapter files (e.g. `torch.py`)
by locating the specific class registered for a framework and injecting a new
`StandardMap` definition into its `definitions` property.

It handles:
1.  **Definitions Injection**: Appending the mapping to the definitions dictionary.
2.  **Smart Import Injection**: Analyzing the target API path (e.g. `scipy.special.erf`)
    and injecting necessary top-level imports (`import scipy`) if missing.
3.  **Variant Parameter Injection**: Supporting `inject_args` for adding fixed arguments.
4.  **Complex Literal Support**: Recursively converting Lists, Tuples, and Dicts to CST nodes.
"""

import libcst as cst
from typing import Optional, Union, List, Set, Sequence, Any, Dict
from ml_switcheroo.core.dsl import FrameworkVariant


class FrameworkInjector(cst.CSTTransformer):
  """
  Injects a `StandardMap` entry into a Framework Adapter's definitions.

  It performs a targeted search for:
  1. A class decorated with `@register_framework("target_fw")`.
  2. A method named `definitions` decorated with `@property`.
  3. A `return` statement returning a `Dict`.

  Additionally, it scans the module for existing imports and injects missing
  dependencies required by the new mapping (e.g. injecting `import scipy` if
  the API is `scipy.special.erf`).

  Attributes:
      target_fw (str): The framework key to look for (e.g. 'torch').
      op_name (str): The abstract operation name (e.g. 'LogSoftmax').
      variant (FrameworkVariant): The configuration to inject.
      found (bool): True if intrusion was successful.
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

    # State tracking vars for Dictionary Injection
    self._in_target_class = False
    self._in_definitions_prop = False

    # State tracking for Import Injection
    self._required_roots: Set[str] = set()
    self._existing_roots: Set[str] = set()

    # Analyze variant for potential import requirements
    # Logic: If API is 'scipy.special.erf', we likely need 'import scipy'.
    if self.variant.api:
      parts = self.variant.api.split(".")
      if len(parts) > 1:
        root = parts[0]
        # Heuristic: We only auto-inject if it looks like an external package.
        # We assume the user wants the root package imported.
        self._required_roots.add(root)

  def visit_Import(self, node: cst.Import) -> None:
    """Tracks existing top-level imports."""
    for alias in node.names:
      # "import scipy.special" -> tracks "scipy"
      root = _get_import_root(alias.name)
      self._existing_roots.add(root)

  def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
    """Tracks existing from-imports."""
    if node.module:
      # "from scipy import special" -> tracks "scipy"
      root = _get_import_root(node.module)
      self._existing_roots.add(root)

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """
    Checks if the class matches the target framework via decorator.
    """
    for decorator in node.decorators:
      # Look for @register_framework("self.target_fw")
      if isinstance(decorator.decorator, cst.Call):
        func = decorator.decorator.func
        if isinstance(func, cst.Name) and func.value == "register_framework":
          args = decorator.decorator.args
          if args:
            # Extract the first string argument
            first_arg = args[0].value
            if isinstance(first_arg, cst.SimpleString):
              # Quote stripping
              val = first_arg.value.strip("'").strip('"')
              if val == self.target_fw:
                self._in_target_class = True
                return True
    return False

  def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
    """Exit class scope."""
    self._in_target_class = False
    return updated_node

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """
    Enters functions. Only interested if inside target class and named 'definitions'.
    """
    if self._in_target_class and node.name.value == "definitions":
      # Optional: verify it has @property decorator
      is_property = any(isinstance(d.decorator, cst.Name) and d.decorator.value == "property" for d in node.decorators)
      if is_property:
        self._in_definitions_prop = True
        return True
    return False

  def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
    """Exit function scope."""
    self._in_definitions_prop = False
    return updated_node

  def leave_Return(self, original_node: cst.Return, updated_node: cst.Return) -> Union[cst.Return, cst.RemovalSentinel]:
    """
    Intercepts the return statement of the definitions property.
    Injects the new key-value into the dictionary.
    """
    if self._in_definitions_prop and isinstance(updated_node.value, cst.Dict):
      self.found = True

      # Construct key: "OpName"
      new_key = cst.SimpleString(f'"{self.op_name}"')

      # Construct value: StandardMap(...)
      new_value = self._build_standard_map_call()

      # Create Dict Element
      new_element = cst.DictElement(
        key=new_key,
        value=new_value,
        comma=cst.Comma(
          whitespace_after=cst.ParenthesizedWhitespace(
            first_line=cst.TrailingWhitespace(newline=cst.Newline()),
            indent=True,
            last_line=cst.SimpleWhitespace("    " * 3),  # Approx indent
          )
        ),
      )

      # Append to elements
      new_elements = list(updated_node.value.elements)
      new_elements.append(new_element)

      new_dict = updated_node.value.with_changes(elements=new_elements)
      return updated_node.with_changes(value=new_dict)

    return updated_node

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Post-process module to inject missing top-level imports.
    """
    missing_roots = self._required_roots - self._existing_roots
    if not missing_roots:
      return updated_node

    # Determine safe insertion point (after Docstrings and Futures)
    body = list(updated_node.body)
    insert_idx = 0

    for i, stmt in enumerate(body):
      if _is_docstring(stmt, i):
        insert_idx = i + 1
        continue
      if _is_future_import(stmt):
        insert_idx = i + 1
        continue
      break

    # Generate new import nodes
    new_imports = []
    for root in sorted(missing_roots):
      # Check if we should inject this root.
      # E.g. avoid injecting 'self' or garbage.
      if not root.isidentifier():
        continue

      # Create `import root`
      imp_node = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name(root))])])
      new_imports.append(imp_node)

    # Inject
    new_body = body[:insert_idx] + new_imports + body[insert_idx:]
    return updated_node.with_changes(body=new_body)

  def _build_standard_map_call(self) -> cst.Call:
    """
    Constructs `StandardMap(api='...', args={...}, inject_args={...}, ...)` CST node.
    Ensures strict formatting (no spaces around '=') to match test expectations.
    """
    # Tight assignment (keyword=value) without spaces
    tight_eq = cst.AssignEqual(
      whitespace_before=cst.SimpleWhitespace(""),
      whitespace_after=cst.SimpleWhitespace(""),
    )

    args_list: List[cst.Arg] = []

    # 1. api="string"
    if self.variant.api:
      args_list.append(
        cst.Arg(
          keyword=cst.Name("api"),
          value=cst.SimpleString(f'"{self.variant.api}"'),
          equal=tight_eq,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # 2. args={"std": "fw"}
    if self.variant.args:
      dict_elements = []
      for k, v in self.variant.args.items():
        dict_elements.append(
          cst.DictElement(
            key=cst.SimpleString(f'"{k}"'),
            value=cst.SimpleString(f'"{v}"'),
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
          )
        )

      # Clean trailing comma from last dict element
      if dict_elements:
        last_d = dict_elements[-1]
        dict_elements[-1] = last_d.with_changes(comma=cst.MaybeSentinel.DEFAULT)

      args_list.append(
        cst.Arg(
          keyword=cst.Name("args"),
          value=cst.Dict(elements=dict_elements),
          equal=tight_eq,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # 3. inject_args={"arg": val} -> Handles recursion now
    if self.variant.inject_args:
      dict_elements = []
      for k, v in self.variant.inject_args.items():
        # Handle types for the value (Recursive)
        val_node = _convert_to_cst_literal(v)

        dict_elements.append(
          cst.DictElement(
            key=cst.SimpleString(f'"{k}"'),
            value=val_node,
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
          )
        )

      if dict_elements:
        last_d = dict_elements[-1]
        dict_elements[-1] = last_d.with_changes(comma=cst.MaybeSentinel.DEFAULT)

      args_list.append(
        cst.Arg(
          keyword=cst.Name("inject_args"),
          value=cst.Dict(elements=dict_elements),
          equal=tight_eq,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # 4. casts={"arg": "type"}
    if self.variant.casts:
      dict_elements = []
      for k, v in self.variant.casts.items():
        dict_elements.append(
          cst.DictElement(
            key=cst.SimpleString(f'"{k}"'),
            value=cst.SimpleString(f'"{v}"'),
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
          )
        )

      if dict_elements:
        last_d = dict_elements[-1]
        dict_elements[-1] = last_d.with_changes(comma=cst.MaybeSentinel.DEFAULT)

      args_list.append(
        cst.Arg(
          keyword=cst.Name("casts"),
          value=cst.Dict(elements=dict_elements),
          equal=tight_eq,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # 5. optional fields
    for field in [
      "requires_plugin",
      "transformation_type",
      "output_adapter",
      "operator",
      "macro_template",
      "output_cast",
    ]:
      val = getattr(self.variant, field, None)
      if val:
        args_list.append(
          cst.Arg(
            keyword=cst.Name(field),
            value=cst.SimpleString(f'"{val}"'),
            equal=tight_eq,
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
          )
        )

    # 6. required_imports support
    if self.variant.required_imports:
      req_elements = []
      for item in self.variant.required_imports:
        # item can be str or ImportReq object
        # Convert to CST using our generic literal converter
        # If it's an object use model_dump
        if hasattr(item, "model_dump"):
          val_node = _convert_to_cst_literal(item.model_dump())
        else:
          val_node = _convert_to_cst_literal(item)

        req_elements.append(cst.Element(value=val_node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))

      # Clean trailing comma
      if req_elements:
        last = req_elements[-1]
        req_elements[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

      args_list.append(
        cst.Arg(
          keyword=cst.Name("required_imports"),
          value=cst.List(elements=req_elements),
          equal=tight_eq,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    # Remove comma from the last argument to ensure clean syntax: StandardMap(api="...")
    if args_list:
      last = args_list[-1]
      args_list[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return cst.Call(func=cst.Name("StandardMap"), args=args_list)


# --- Helpers ---


def _get_import_root(node: Union[cst.Name, cst.Attribute]) -> str:
  """Recursively extracts the root package name."""
  if isinstance(node, cst.Name):
    return node.value
  if isinstance(node, cst.Attribute):
    return _get_import_root(node.value)
  return ""


def _is_docstring(node: cst.CSTNode, idx: int) -> bool:
  """Checks if statement is a module docstring."""
  if idx != 0:
    return False
  if isinstance(node, cst.SimpleStatementLine) and len(node.body) == 1:
    expr = node.body[0]
    if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString)):
      return True
  return False


def _is_future_import(node: cst.CSTNode) -> bool:
  """Checks if statement is `from __future__ import ...`."""
  if isinstance(node, cst.SimpleStatementLine):
    for stmt in node.body:
      if isinstance(stmt, cst.ImportFrom):
        if stmt.module and isinstance(stmt.module, cst.Name) and stmt.module.value == "__future__":
          return True
  return False


def _convert_to_cst_literal(val: Any) -> cst.BaseExpression:
  """
  Recursively converts a python primitive or container to a CST node.
  Supports: int, float, bool, str, list, tuple, dict.
  """
  # 1. Container Recursion (List/Tuple)
  if isinstance(val, (list, tuple)):
    elements = []
    for item in val:
      node = _convert_to_cst_literal(item)
      elements.append(cst.Element(value=node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))))

    # Strip trailing comma from last element for cleanliness
    if elements:
      last = elements[-1]
      elements[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    if isinstance(val, list):
      return cst.List(elements=elements)
    else:
      return cst.Tuple(elements=elements)

  # 2. Key-Value Recursion (Dict)
  if isinstance(val, dict):
    elements = []
    for k, v in val.items():
      k_node = _convert_to_cst_literal(k)
      v_node = _convert_to_cst_literal(v)

      elements.append(
        cst.DictElement(
          key=k_node,
          value=v_node,
          comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
        )
      )

    if elements:
      last = elements[-1]
      elements[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return cst.Dict(elements=elements)

  # 3. Primitives
  if isinstance(val, bool):
    return cst.Name("True") if val else cst.Name("False")
  elif isinstance(val, int):
    return cst.Integer(str(val))
  elif isinstance(val, float):
    return cst.Float(str(val))
  elif isinstance(val, str):
    # Basic quote escaping
    s = val.replace('"', '\\"')
    return cst.SimpleString(f'"{s}"')
  elif val is None:
    return cst.Name("None")
  else:
    # Fallback
    s = str(val).replace('"', '\\"')
    return cst.SimpleString(f'"{s}"')

"""Integration Test Harness Generator.

Generates standalone verification scripts.
Bundles fuzzer logic (including Hypothesis strategies).
"""

from typing import Any

import json
import inspect
import textwrap
from pathlib import Path
from typing import Dict, Optional

import libcst as cst
from ml_switcheroo.testing.harness_generator_template import get_harness_skeleton
from ml_switcheroo.testing.fuzzer.core import InputFuzzer
from ml_switcheroo.utils.code_extractor import CodeExtractor
from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY, get_adapter

# Imports needed for bundling the fuzzer logic
import ml_switcheroo.testing.fuzzer.generators
import ml_switcheroo.testing.fuzzer.parser
import ml_switcheroo.testing.fuzzer.heuristics
import ml_switcheroo.testing.fuzzer.utils
import ml_switcheroo.testing.fuzzer.strategies
import ml_switcheroo.testing.fuzzer.type_parser
from ml_switcheroo.testing.signature_extractor import SignatureExtractor


class HarnessInjector(cst.CSTTransformer):
  """Injects dynamic blocks into the harness skeleton."""

  def __init__(
    self,
    fuzzer_block: str,
    imports_block: str,
    init_helpers_block: str,
    injection_block: str,
    to_numpy_block: str,
    source_path: str,
    target_path: str,
    source_fw: str,
    target_fw: str,
    hints_json: str,
  ) -> None:
    """Initialize the injector with code blocks and metadata.

    Args:
        fuzzer_block: Standalone code block containing fuzzer logic.
        imports_block: Import statements required for the target framework.
        init_helpers_block: Initialization helper code block.
        injection_block: Block containing parameter injection logic.
        to_numpy_block: Logic for converting framework outputs to NumPy arrays.
        source_path: File path of the source framework code.
        target_path: File path of the target framework code.
        source_fw: Name of the source framework (e.g., 'torch').
        target_fw: Name of the target framework (e.g., 'jax').
        hints_json: JSON string mapping ops to standard parameter types.
    """
    self.fuzzer_block = fuzzer_block
    self.imports_block = imports_block
    self.init_helpers_block = init_helpers_block
    self.injection_block = injection_block
    self.to_numpy_block = to_numpy_block
    self.source_path = source_path
    self.target_path = target_path
    self.source_fw = source_fw
    self.target_fw = target_fw
    self.hints_json = hints_json

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """Injects imports, fuzzer, and helpers at the module level.

    Args:
        original_node: The original LibCST Module node.
        updated_node: The updated LibCST Module node after visiting children.

    Returns:
        The modified LibCST Module with injected blocks.
    """
    # Find the index of the first function definition (def to_numpy) to insert before it
    insert_idx = 0
    for i, stmt in enumerate(updated_node.body):
      if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "to_numpy":
        insert_idx = i
        break

    nodes: list = []
    if self.imports_block.strip():
      nodes.extend(cst.parse_module(self.imports_block).body)
    if self.init_helpers_block.strip():
      nodes.extend(cst.parse_module(self.init_helpers_block).body)
    if self.fuzzer_block.strip():
      nodes.extend(cst.parse_module(self.fuzzer_block).body)

    new_body = list(updated_node.body[:insert_idx]) + nodes + list(updated_node.body[insert_idx:])
    return updated_node.with_changes(body=new_body)

  def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> Any:
    """Injects to_numpy logic into the to_numpy function definition.

    Args:
        original_node: The original LibCST FunctionDef node.
        updated_node: The updated LibCST FunctionDef node.

    Returns:
        The modified function node with injected normalization logic,
        or the updated node if no logic is injected.
    """
    if original_node.name.value == "to_numpy":
      if not self.to_numpy_block.strip():
        return updated_node
      parsed_logic = cst.parse_module(self.to_numpy_block).body
      # Find the try-except block
      insert_idx = 1
      new_body = (
        list(updated_node.body.body[:insert_idx]) + list(parsed_logic) + list(updated_node.body.body[insert_idx:])
      )
      return updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))
    return updated_node

  def leave_If(self, original_node: cst.If, updated_node: cst.If) -> Any:
    """Injects the param injection logic replacing the target `if tp not in tgt_inputs: pass`.

    Args:
        original_node: The original LibCST If node.
        updated_node: The updated LibCST If node.

    Returns:
        The modified If node containing the injected parameter logic,
        or the updated node unchanged.
    """
    if isinstance(original_node.test, cst.Comparison) and len(original_node.test.comparisons) == 1:
      comp = original_node.test.comparisons[0]
      if (
        isinstance(comp.operator, cst.NotIn)
        and isinstance(comp.comparator, cst.Name)
        and comp.comparator.value == "tgt_inputs"
      ):
        if self.injection_block.strip() != "pass":
          parsed = cst.parse_module(self.injection_block).body
          return updated_node.with_changes(body=updated_node.body.with_changes(body=parsed))
    return updated_node

  def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> Any:
    """Modifies the run_verification arguments to use dynamic paths and frameworks.

    Args:
        original_node: The original LibCST Call node.
        updated_node: The updated LibCST Call node.

    Returns:
        The modified Call node with updated keyword arguments,
        or the updated node unchanged.
    """
    if isinstance(original_node.func, cst.Name) and original_node.func.value == "run_verification":
      args = [
        cst.Arg(value=cst.SimpleString(f'r"{self.source_path}"'), keyword=cst.Name("source_path")),
        cst.Arg(value=cst.SimpleString(f'r"{self.target_path}"'), keyword=cst.Name("target_path")),
        cst.Arg(value=cst.SimpleString(f'"{self.source_fw}"'), keyword=cst.Name("source_fw")),
        cst.Arg(value=cst.SimpleString(f'"{self.target_fw}"'), keyword=cst.Name("target_fw")),
        cst.Arg(value=cst.SimpleString(f"r'{self.hints_json}'"), keyword=cst.Name("hints_json_str")),
      ]
      return updated_node.with_changes(args=args)
    return updated_node


class HarnessGenerator:
  """Generates standalone verification scripts tailored to the target framework."""

  def __init__(self) -> None:
    """Initializes the generator instance and its code extractor utility."""
    self.extractor = CodeExtractor()

  def generate(
    self,
    source_file: Path,
    target_file: Path,
    output_harness: Path,
    source_fw: str = "torch",
    target_fw: str = "jax",
    semantics: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Creates the verification harness file and writes it to disk.

    Args:
        source_file: Path to the source framework script.
        target_file: Path to the target framework script.
        output_harness: Target Path where the harness script will be saved.
        source_fw: Name of the source framework. Defaults to "torch".
        target_fw: Name of the target framework. Defaults to "jax".
        semantics: Optional dictionary of semantic operator details for parameter types.
    """
    fuzzer_code = self._bundle_fuzzer_dependencies()

    hints_map = {}
    if semantics:
      for op_name, details in semantics.items():
        args_data = details.get("std_args", [])
        func_hints = {}
        for arg in args_data:
          if isinstance(arg, (list, tuple)) and len(arg) == 2:
            func_hints[arg[0]] = arg[1]
          elif isinstance(arg, dict):  # pragma: no branch
            # Support rich ODL parameter definitions
            name = arg.get("name")
            typ = arg.get("type")
            if name and typ:
              func_hints[name] = typ
        if func_hints:  # pragma: no branch
          hints_map[op_name] = func_hints

    hints_json = json.dumps(hints_map).replace("'", '"')
    adapter_shim = self._generate_adapter_shim()

    fuzzer_block = f"{adapter_shim}\n\n{fuzzer_code}\n\nclass StandaloneFuzzer(InputFuzzer):\n    pass\n"

    imports_block, init_helpers_block, injection_logic_block = self._build_dynamic_init(target_fw)
    to_numpy_block = self._build_result_normalization(source_fw, target_fw)

    skeleton = get_harness_skeleton()
    module = cst.parse_module(skeleton)
    injector = HarnessInjector(
      fuzzer_block=fuzzer_block,
      imports_block=imports_block,
      init_helpers_block=init_helpers_block,
      injection_block=injection_logic_block,
      to_numpy_block=to_numpy_block,
      source_path=source_file.resolve().as_posix(),
      target_path=target_file.resolve().as_posix(),
      source_fw=source_fw,
      target_fw=target_fw,
      hints_json=hints_json,
    )

    modified_module = module.visit(injector)
    script_content = modified_module.code

    output_harness.parent.mkdir(parents=True, exist_ok=True)
    with open(output_harness, "wt", encoding="utf-8") as f:
      f.write(script_content)

  def _bundle_fuzzer_dependencies(self) -> str:
    """Extracts all helper functions required by InputFuzzer.

    Injects Hypothesis and typing imports globally for the bundle.

    Returns:
        A single combined string containing imports, classes, and helper
        functions needed to run the InputFuzzer standalone.
    """
    deps = []

    # Global imports required by the extracted code
    # We must ensure all imports used by strategies.py and core.py types/logic are present
    deps.append("import hypothesis.strategies as st")
    deps.append("import hypothesis.extra.numpy as npst")
    deps.append("import re")
    deps.append("import numpy as np")
    deps.append("from typing import Union, Any, Dict, List, Optional, Tuple, Callable")
    deps.append("from dataclasses import dataclass")
    deps.append("import typing")
    deps.append("""
@dataclass
class ParsedType:
    pass

@dataclass
class AnyType(ParsedType):
    pass

@dataclass
class NoneType(ParsedType):
    pass

@dataclass
class PrimitiveType(ParsedType):
    name: str

@dataclass
class UnionType(ParsedType):
    types: typing.List[ParsedType]

@dataclass
class OptionalType(ParsedType):
    inner: ParsedType

@dataclass
class TupleType(ParsedType):
    elements: typing.List[ParsedType]
    variadic: bool

@dataclass
class ListType(ParsedType):
    inner: ParsedType

@dataclass
class DictType(ParsedType):
    key_type: ParsedType
    value_type: ParsedType

@dataclass
class TensorType(ParsedType):
    dims: typing.Optional[typing.List[str]]

@dataclass
class CallableType(ParsedType):
    pass
""")

    def extract_module_functions(module: Any) -> Any:
      """Extracts and de-indents all top-level functions from the given module.

      Args:
          module: The Python module object to extract functions from.
      """
      funcs = inspect.getmembers(module, inspect.isfunction)
      for name, func in funcs:
        if func.__module__ == module.__name__:
          try:
            source = inspect.getsource(func)
            deps.append(textwrap.dedent(source))
          except OSError:
            pass

    # Order matters slightly for resolution order of helpers
    extract_module_functions(ml_switcheroo.testing.fuzzer.utils)
    extract_module_functions(ml_switcheroo.testing.fuzzer.strategies)
    extract_module_functions(ml_switcheroo.testing.fuzzer.generators)
    extract_module_functions(ml_switcheroo.testing.fuzzer.heuristics)

    deps.append("import ast")
    deps.append(
      "class TypeAnnotationParser(ast.NodeVisitor):\n    def parse(self, type_str: str) -> ParsedType:\n        if not type_str or type_str.strip() == '':\n            return AnyType()\n        try:\n            tree = ast.parse(type_str.strip(), mode='eval')\n            return self.visit(tree.body)\n        except SyntaxError:\n            return PrimitiveType(name=type_str)\n    def visit_Name(self, node: ast.Name) -> ParsedType:\n        name = node.id\n        if name == 'Any': return AnyType()\n        if name in ('None', 'NoneType'): return NoneType()\n        if name in ('int', 'integer', 'float', 'double', 'number', 'bool', 'boolean', 'str', 'string'): return PrimitiveType(name=name)\n        if name in ('Array', 'Tensor', 'ndarray'): return TensorType(dims=None)\n        if name in ('Callable', 'func', 'function'): return CallableType()\n        if name in ('List', 'Sequence'): return ListType(inner=AnyType())\n        if name in ('Dict', 'Mapping'): return DictType(key_type=AnyType(), value_type=AnyType())\n        if name == 'Tuple': return TupleType(elements=[AnyType()], variadic=True)\n        if name == 'Optional': return OptionalType(inner=AnyType())\n        return PrimitiveType(name=name)\n    def visit_Attribute(self, node: ast.Attribute) -> ParsedType:\n        return PrimitiveType(name=node.attr)\n    def visit_Constant(self, node: ast.Constant) -> ParsedType:\n        if node.value is None: return NoneType()\n        if node.value is Ellipsis: return PrimitiveType(name='Ellipsis')\n        return PrimitiveType(name=str(node.value))\n    def visit_Subscript(self, node: ast.Subscript) -> ParsedType:\n        base = self.visit(node.value)\n        slice_val = node.slice\n        args = []\n        is_variadic = False\n        raw_dims = []\n        def _process(elt):\n            nonlocal is_variadic\n            if isinstance(elt, ast.Constant) and elt.value is Ellipsis: is_variadic = True\n            elif isinstance(elt, ast.Constant) and isinstance(elt.value, str): raw_dims.append(elt.value)\n            elif isinstance(elt, ast.Name) and elt.id != 'Ellipsis': raw_dims.append(elt.id)\n            args.append(self.visit(elt))\n        if isinstance(slice_val, ast.Tuple):\n            for elt in slice_val.elts: _process(elt)\n        elif hasattr(ast, 'Index') and isinstance(slice_val, getattr(ast, 'Index')):\n            _process(getattr(slice_val, 'value'))\n        else:\n            _process(slice_val)\n        if isinstance(base, OptionalType): return OptionalType(inner=args[0] if args else AnyType())\n        if isinstance(base, ListType): return ListType(inner=args[0] if args else AnyType())\n        if isinstance(base, TupleType):\n            elements = [a for a in args if not (isinstance(a, PrimitiveType) and a.name == 'Ellipsis')]\n            return TupleType(elements=elements, variadic=is_variadic)\n        if isinstance(base, DictType): return DictType(key_type=args[0], value_type=args[1]) if len(args) == 2 else DictType(key_type=AnyType(), value_type=AnyType())\n        if isinstance(base, TensorType): return TensorType(dims=raw_dims if raw_dims else None)\n        if isinstance(base, PrimitiveType) and base.name == 'Union': return UnionType(types=args)\n        return base\n    def visit_BinOp(self, node: ast.BinOp) -> ParsedType:\n        if isinstance(node.op, ast.BitOr):\n            left = self.visit(node.left)\n            right = self.visit(node.right)\n            types = []\n            types.extend(left.types if isinstance(left, UnionType) else [left])\n            types.extend(right.types if isinstance(right, UnionType) else [right])\n            return UnionType(types=types)\n        return PrimitiveType(name='Unknown')\n    def generic_visit(self, node: ast.AST) -> ParsedType:\n        return PrimitiveType(name='Unknown')\n\ndef parse_type_annotation(type_str: str) -> ParsedType:\n    return TypeAnnotationParser().parse(type_str)"
    )

    extract_module_functions(ml_switcheroo.testing.fuzzer.parser)

    fuzzer_class = self.extractor.extract_class(InputFuzzer)

    return "\n\n".join(deps + [fuzzer_class])

  def _build_dynamic_init(self, target_fw: str) -> tuple[str, str, str]:
    """Builds initialization helpers and parameter injection logic for the target framework.

    Args:
        target_fw: Name of the target framework.

    Returns:
        A tuple containing:
            - The imports block string.
            - The helper initialization functions block string.
            - The injection logic block string.
    """
    adapter = get_adapter(target_fw)
    if not adapter:
      return "", "", "pass"

    imports = getattr(adapter, "harness_imports", [])
    imports_str = "\n".join(imports)
    init_code = getattr(adapter, "get_harness_init_code", lambda: "")()
    helper_name = SignatureExtractor.extract_first_function_name(init_code)
    magic_args = getattr(adapter, "declared_magic_args", [])

    injection_lines = []
    if magic_args and helper_name:
      quoted_args = [f'"{a}"' for a in magic_args]
      list_str = "[" + ", ".join(quoted_args) + "]"
      injection_lines.append("val = None")
      injection_lines.append(f"if tp in {list_str}:")
      injection_lines.append(f"    val = {helper_name}(seed=42)")
      injection_lines.append("if val is not None:")
      injection_lines.append("    tgt_inputs[tp] = val")
    else:
      injection_lines.append("pass")

    final_logic = injection_lines[0]
    for line in injection_lines[1:]:
      final_logic += "\n" + line

    return imports_str, init_code, final_logic

  def _build_result_normalization(self, source_fw: str, target_fw: str) -> str:
    """Aggregates NumPy normalization helper codes for the involved frameworks.

    Args:
        source_fw: Name of the source framework.
        target_fw: Name of the target framework.

    Returns:
        A combined string of normalization functions for converting outputs to NumPy.
    """
    blocks = []
    unique_fws = set([source_fw, target_fw])
    if "flax_nnx" in unique_fws:
      unique_fws.add("jax")

    for fw in unique_fws:
      adapter = get_adapter(fw)
      code = None
      if adapter and hasattr(adapter, "get_to_numpy_code"):
        try:
          code = adapter.get_to_numpy_code()
        except Exception:
          pass
      if code:
        blocks.append(f"# Framework: {fw}\n{code}")

    return "\n".join(blocks)

  def _generate_adapter_shim(self) -> str:
    """Generates a standalone mock/shim of get_adapter for the generated harness.

    Returns:
        A string containing a Python function definition that returns a
        generic adapter compatible with registered frameworks.
    """
    shim_lines = [
      "# Shim for missing get_adapter",
      "def get_adapter(framework):",
      "    class GenericAdapter:",
      "        def convert(self, data):",
      "            try:",
      "                import numpy as np",
      "                if not isinstance(data, (np.ndarray, np.generic)) and not isinstance(data, (list, tuple)):",
      "                    return data",
      "            except ImportError:",
      "                pass",
      "",
    ]
    frameworks = sorted(_ADAPTER_REGISTRY.keys())
    first = True
    for fw_name in frameworks:
      adapter_cls = _ADAPTER_REGISTRY[fw_name]
      if not hasattr(adapter_cls, "convert"):
        continue

      try:
        method_source = inspect.getsource(adapter_cls.convert)
      except OSError:
        continue

      clean_block = textwrap.dedent(method_source)
      lines = clean_block.splitlines()
      body_start = 0
      for i, line in enumerate(lines):  # pragma: no branch
        if line.strip().startswith("def convert"):  # pragma: no branch
          body_start = i + 1
          break

      body_lines = lines[body_start:]
      condition_kw = "if" if first else "elif"
      shim_lines.append(f"            {condition_kw} framework == '{fw_name}':")
      base_indent = " " * 16
      body_str = textwrap.dedent("\n".join(body_lines))
      indented_body = textwrap.indent(body_str, base_indent)
      shim_lines.append(indented_body)
      first = False

    shim_lines.append("")
    shim_lines.append("            return data")
    shim_lines.append("")
    shim_lines.append("    return GenericAdapter()")
    return "\n".join(shim_lines)

"""MLIR Printer.

Formats MLIR CST nodes into standardized MLIR text.
"""

from ml_switcheroo.core.mlir.cst import MlirNode, ModuleNode, OperationNode


class MlirPrinter:
  """Prints MLIR CST to textual representation.

  Ensures consistent formatting and layout, acting as an alternative or
  wrapper around `node.to_text()` that enforces file-level consistency.
  """

  def emit(self, node: MlirNode, header: str = "// Graph -> MLIR compilation output\n") -> str:
    """Emits the textual representation for an MLIR node.

    Args:
        node (~ml_switcheroo.core.mlir.cst.MlirNode): The root MLIR node to print.
        header (str): Optional header string for modules.

    Returns:
        str: The generated MLIR text.
    """
    if isinstance(node, ModuleNode):
      return self._emit_module(node, header)

    return node.to_text()

  def _emit_module(self, module: ModuleNode, header: str) -> str:
    """Emits a module with standard formatting.

    Args:
        module: The module to emit.
        header: The header string to add.

    Returns:
        str: The generated MLIR text.
    """
    # We can inject a file-level comment or simply use the existing to_text()
    # Currently, `ModuleNode.to_text` just delegates to its block.
    # We explicitly emit a module wrapper if the block isn't already inside one.

    out = [header] if header else []

    # Check if the module body block represents a module itself or just basic ops
    has_module_op = any(isinstance(op, OperationNode) and op.name == "module" for op in module.body.operations)

    if not has_module_op:
      out.append("module {\n")
      # Optional: In MLIR backend tests we wrap in a function if it's raw
      out.append("  func.func @main() {\n")

      # We process the operations directly to add indentation
      for op in module.body.operations:
        # basic indentation for the block
        op_text = op.to_text()
        # indent lines
        indented = "\n".join(f"    {line}" if line else "" for line in op_text.splitlines())
        out.append(f"{indented}\n")

      out.append("  }\n")
      out.append("}\n")
    else:
      out.append(module.to_text())

    return "".join(out)

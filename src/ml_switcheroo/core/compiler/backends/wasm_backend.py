"""WebAssembly (WAT) Compiler Backend.

This module provides the WasmBackend class, which compiles a LogicalGraph representing
computational steps into its WebAssembly Text (WAT) equivalent.
"""

from typing import Any, Optional
from ml_switcheroo.core.compiler.backend import CompilerBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.wasm.cst import (
  WatModule,
  WatFunc,
  WatParam,
  WatResult,
  WatLocal,
  WatInstr,
  WasmRegister,
  WasmOpcode,
  WasmArgument,
)


class WasmBackend(CompilerBackend):
  """Back-end for generating WebAssembly Text (WAT) from a LogicalGraph.

  This backend converts nodes of a LogicalGraph into corresponding stack-based WAT instructions,
  handling parameters, local variables, and outputs.
  """

  def __init__(self, semantics: Optional[Any] = None) -> None:
    """Initialize the WebAssembly backend.

    Args:
        semantics: Optional semantic information or rules used during translation of graph nodes.
    """
    self.semantics = semantics

  def compile(self, graph: LogicalGraph) -> str:
    """Compiles the LogicalGraph into a WAT string.

    This method translates a logical representation of operations (nodes and edges) into
    WebAssembly instructions, structures the inputs as function parameters, maps intermediate
    results to local variables, and produces the final WebAssembly Text (WAT) format.

    Args:
        graph: The logical graph containing nodes and edges representing operations.

    Returns:
        The generated WAT code representing the compiled module and functions.
    """
    module = WatModule()

    # Simple heuristic mapping for this backend
    func_name = graph.name or "main"
    func = WatFunc(name=func_name, export=True)

    # Determine inputs
    inputs = [n for n in graph.nodes if n.kind == "Input"]
    for i, inp in enumerate(inputs):
      func.params.append(WatParam(name=WasmRegister(f"arg{i}"), type_id="f32"))

    # Map nodes to instructions
    # Since WASM is stack-based, we map variables to locals
    for node in graph.nodes:
      if node.kind in ["Input", "Output"]:
        continue

      # Map common IR nodes to basic WASM instructions
      func.locals.append(WatLocal(name=WasmRegister(node.id), type_id="f32"))

      # Find incoming edges
      incoming = [e.source for e in graph.edges if e.target == node.id]
      for inc in incoming:
        # Map source to param if it's an input
        src_node = next((n for n in graph.nodes if n.id == inc), None)
        if src_node and src_node.kind == "Input":
          idx = inputs.index(src_node)
          func.body.append(WatInstr(opcode=WasmOpcode("local.get"), args=[WasmArgument(f"$arg{idx}")]))
        else:
          func.body.append(WatInstr(opcode=WasmOpcode("local.get"), args=[WasmArgument(f"${inc}")]))

      if node.kind == "Add":
        func.body.append(WatInstr(opcode=WasmOpcode("f32.add")))
      elif node.kind == "Mul":
        func.body.append(WatInstr(opcode=WasmOpcode("f32.mul")))
      elif node.kind == "Sub":
        func.body.append(WatInstr(opcode=WasmOpcode("f32.sub")))
      else:
        # Default to calling some imported function or just a comment
        func.body.append(WatInstr(opcode=WasmOpcode("call"), args=[WasmArgument(f"${node.kind}")]))

      func.body.append(WatInstr(opcode=WasmOpcode("local.set"), args=[WasmArgument(f"${node.id}")]))

    # Output
    outputs = [n for n in graph.nodes if n.kind == "Output"]
    if outputs:
      func.results.append(WatResult(type_id="f32"))
      incoming = [e.source for e in graph.edges if e.target == outputs[0].id]
      if incoming:
        func.body.append(WatInstr(opcode=WasmOpcode("local.get"), args=[WasmArgument(f"${incoming[0]}")]))

    module.functions.append(func)
    return module.to_text()

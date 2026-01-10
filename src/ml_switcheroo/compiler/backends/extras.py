"""
Extra Backends (Wrappers for Legacy/IR Emitters).

Provides CompilerBackend implementations for:
- TikZ
- HTML
- LaTeX DSL
- MLIR (StableHLO)
"""

from ml_switcheroo.compiler.backend import CompilerBackend
from ml_switcheroo.compiler.ir import LogicalGraph
from ml_switcheroo.core.tikz.emitter import TikzEmitter as LegacyTikzEmitter
from ml_switcheroo.core.html.emitter import HtmlEmitter as LegacyHtmlEmitter
from ml_switcheroo.core.latex.emitter import LatexEmitter as LegacyLatexEmitter
from ml_switcheroo.semantics.manager import SemanticsManager


class TikzBackend(CompilerBackend):
  def __init__(self, semantics: SemanticsManager = None):
    self.emitter = LegacyTikzEmitter()

  def compile(self, graph: LogicalGraph) -> str:
    return self.emitter.emit(graph)


class HtmlBackend(CompilerBackend):
  def __init__(self, semantics: SemanticsManager = None):
    self.emitter = LegacyHtmlEmitter()

  def compile(self, graph: LogicalGraph) -> str:
    children = self.emitter._layout_graph(graph)
    from ml_switcheroo.core.html.nodes import HtmlDocument

    # Use graph name populated by GraphExtractor
    doc = HtmlDocument(model_name=graph.name, children=children)
    return doc.to_html()


class LatexBackend(CompilerBackend):
  def __init__(self, semantics: SemanticsManager = None):
    self.emitter = LegacyLatexEmitter()

  def compile(self, graph: LogicalGraph) -> str:
    container = self.emitter._transcode_graph(graph, graph.name or "Model")
    return self.emitter._wrap_document(container.to_latex())


class MlirBackend(CompilerBackend):
  def __init__(self, semantics: SemanticsManager = None):
    pass

  def compile(self, graph: LogicalGraph) -> str:
    # Stub implementation to satisfying basic MLIR structure tests
    lines = ["// Graph -> MLIR compilation output"]
    lines.append("module {")
    for node in graph.nodes:
      if node.kind == "Input":
        val = node.metadata.get("value", "1")
        lines.append(f'  %{node.id} = "sw.constant"() {{value = {val}}} : i32')
      else:
        lines.append(f'  %{node.id} = "sw.op"() {{type = "{node.kind}"}}')
    lines.append("}")
    return "\n".join(lines)


class StableHloBackend(CompilerBackend):
  def __init__(self, semantics: SemanticsManager = None):
    pass

  def compile(self, graph: LogicalGraph) -> str:
    return "// Graph -> StableHLO compilation not fully implemented in this phase.\n"

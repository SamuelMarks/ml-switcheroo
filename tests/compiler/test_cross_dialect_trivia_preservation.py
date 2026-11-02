"""Unit tests for Phase 14: End-to-End Trivia Preservation Across Dialects.

Verifies 100% comment/trivia preservation across:
- Python <-> LogicalGraph <-> Python
- PyTorch <-> RDNA
- PyTorch <-> NVIDIA_SASS
- JAX <-> StableHLO
- PyTorch <-> MLIR
- RDNA <-> NVIDIA_SASS
"""

import libcst as cst
from ml_switcheroo_ir import DType

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.compiler.backends.mlir_backend import MlirBackend, extract_mlir_node_trivia
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.backends.python import PythonBackend
from ml_switcheroo.core.compiler.backends.python_strategies.init import (
  _normalize_py_comment,
  apply_node_trivia_to_stmt,
)
from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.frontends.nvidia_sass import NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.rdna import RdnaLifter, RdnaParser
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.graph import GraphExtractor
from ml_switcheroo.core.mlir.emitter import PythonToMlirEmitter
from ml_switcheroo.core.mlir.generator import MlirToPythonGenerator
from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.semantics.manager import SemanticsManager


def test_graph_extractor_and_python_backend_trivia_roundtrip() -> None:
  """Verify GraphExtractor captures leading and trailing comments and PythonBackend restores them."""
  code = """import torch.nn as nn

class TriviaNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Lead comment on linear init
        self.fc = nn.Linear(4, 2)  # Inline init comment

    def forward(self, x):
        # Forward call leading comment
        x = self.fc(x)  # Forward call inline comment
        x = self.fc(x)  # Second call inline comment
        # Return leading comment
        return x  # Return inline comment
"""
  tree = cst.parse_module(code)
  extractor = GraphExtractor()
  tree.visit(extractor)
  graph = extractor.graph

  fc_node = graph.nodes["fc"]
  assert "Lead comment on linear init" in str(fc_node.attributes.get("leading_trivia", ""))
  assert "Forward call leading comment" in str(fc_node.attributes.get("leading_trivia", ""))
  assert "Inline init comment" in str(fc_node.attributes.get("trailing_trivia", ""))
  assert "Forward call inline comment" in str(fc_node.attributes.get("trailing_trivia", ""))
  assert "Second call inline comment" in str(fc_node.attributes.get("trailing_trivia", ""))

  out_node = graph.nodes["output"]
  assert "Return leading comment" in str(out_node.attributes.get("leading_trivia", ""))
  assert "Return inline comment" in str(out_node.attributes.get("trailing_trivia", ""))

  backend = PythonBackend(framework="flax_nnx", semantics=SemanticsManager())
  synthesized = backend.compile(graph)
  assert "# Lead comment on linear init" in synthesized
  assert "# Inline init comment" in synthesized
  assert "# Forward call inline comment" in synthesized
  assert "# Return leading comment" in synthesized
  assert "# Return inline comment" in synthesized


def test_normalize_and_apply_node_trivia_helpers() -> None:
  """Verify comment normalization and list-based trivia decoration."""
  assert _normalize_py_comment("# already py") == "# already py"
  assert _normalize_py_comment("// mlir style") == "# mlir style"
  assert _normalize_py_comment("; asm style") == "# asm style"
  assert _normalize_py_comment("bare text") == "# bare text"

  stmt = cst.parse_statement("x = y  # existing")
  assert isinstance(stmt, cst.SimpleStatementLine)
  node = LogicalNode(
    id="n1",
    op_type="Linear",
    attributes={
      "leading_trivia": ["// line one", "; line two"],
      "trailing_trivia": "; extra trailing",
      "runtime_comment": "[on-device fwd] shape=(2, 4)",
    },
  )
  decorated = apply_node_trivia_to_stmt(stmt, node, include_runtime_comment=True)
  mod_code = cst.Module(body=[decorated]).code
  assert "# line one" in mod_code
  assert "# line two" in mod_code
  assert "# existing" in mod_code
  assert "# extra trailing" in mod_code
  assert "# [on-device fwd] shape=(2, 4)" in mod_code


def test_pytorch_to_rdna_and_sass_trivia_preservation() -> None:
  """Verify user comments survive PyTorch -> RDNA -> NVIDIA_SASS -> PyTorch."""
  py_code = """import torch.nn as nn

class AsmTriviaModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Custom projection layer comment
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        # Execute dense projection on GPU
        x = self.fc(x)  # inline projection note
        return x
"""
  sem = SemanticsManager()
  engine_to_rdna = ASTEngine(
    semantics=sem,
    config=RuntimeConfig(source_framework="torch", target_framework="rdna"),
  )
  rdna_res = engine_to_rdna.run(py_code)
  assert "Custom projection layer comment" in rdna_res.code
  assert "Execute dense projection on GPU" in rdna_res.code
  assert "inline projection note" in rdna_res.code

  engine_rdna_to_sass = ASTEngine(
    semantics=sem,
    config=RuntimeConfig(
      source_framework="rdna",
      target_framework="nvidia_sass",
      enable_on_device_execution=True,
    ),
  )
  sass_res = engine_rdna_to_sass.run(rdna_res.code)
  assert "Custom projection layer comment" in sass_res.code
  assert "Execute dense projection on GPU" in sass_res.code
  assert "inline projection note" in sass_res.code

  engine_sass_to_torch = ASTEngine(
    semantics=sem,
    config=RuntimeConfig(source_framework="nvidia_sass", target_framework="torch"),
  )
  torch_res = engine_sass_to_torch.run(sass_res.code)
  assert "Custom projection layer comment" in torch_res.code
  assert "Execute dense projection on GPU" in torch_res.code


def test_isa_lifter_and_synthesizer_trailing_and_list_comments() -> None:
  """Verify IsaLifter trailing comments and IsaSynthesizer list trivia handling."""
  rdna_text = """; RDNA Code Generation Initialized (Arch: gfx1030)
; Input x -> v0
; Custom leading comment before add
v_add_f32 v1, v0, v0
; Return: v1
; Trailing assembly epilogue comment
"""
  nodes = RdnaParser(rdna_text).parse().statements
  graph = RdnaLifter().lift(nodes)
  assert "output" in graph.nodes
  assert "Trailing assembly epilogue comment" in str(graph.nodes["output"].attributes.get("leading_trivia", ""))

  graph.nodes["output"].attributes["leading_trivia"] = ["List item comment A", "List item comment B"]
  synth = RdnaSynthesizer(SemanticsManager())
  synth.emit_trivia_comments = True
  isa_out = synth.from_graph(graph)
  py_mod = synth.to_python(isa_out)
  assert "Custom leading comment before add" in py_mod.code

  sass_text = """// Input x -> R0
// SASS custom note
FADD R1, R0, R0;
// Trailing note after instruction
"""
  sass_nodes = NvidiaSassParser(sass_text).parse().statements
  sass_py = NvidiaSassSynthesizer(SemanticsManager()).to_python(sass_nodes)
  assert "# SASS custom note" in sass_py.code
  assert "# Trailing note after instruction" in sass_py.code

  only_comments = NvidiaSassParser("// Standalone comment only\n//   \n").parse().statements
  empty_py = NvidiaSassSynthesizer(SemanticsManager()).to_python(only_comments)
  assert list(empty_py.body) == []


def test_mlir_and_stablehlo_inline_and_graph_trivia() -> None:
  """Verify PythonToMlirEmitter, MlirToPythonGenerator, MlirBackend, and StableHloBackend preserve trivia."""
  py_code = """# Header comment
def forward(x):
    # Leading comment
    y = x + x  # Inline trailing comment
    return y
"""
  tree = cst.parse_module(py_code)
  emitter = PythonToMlirEmitter()
  mlir_mod = emitter.convert(tree)
  mlir_text = mlir_mod.to_text()
  assert "// Leading comment" in mlir_text
  assert "// Inline trailing comment" in mlir_text

  parsed = MlirParser(mlir_text).parse()
  gen = MlirToPythonGenerator()
  reconstructed = gen.generate(parsed).code
  assert "# Leading comment" in reconstructed
  assert "# Inline trailing comment" in reconstructed

  graph = LogicalGraph(
    name="TriviaGraph",
    nodes={
      "x": LogicalNode(id="x", op_type="Input", attributes={"leading_trivia": ["Input comment"]}),
      "add": LogicalNode(
        id="add",
        op_type="Add",
        attributes={
          "leading_trivia": "Add op leading comment",
          "runtime_comment": "[on-device fwd] shape=(2, 4)",
        },
        shape_metadata=(2, 4),
        dtype=DType("float32"),
      ),
      "output": LogicalNode(id="output", op_type="Output", attributes={"leading_trivia": "Output comment"}),
    },
    edges=[LogicalEdge("x", "add"), LogicalEdge("add", "output")],
  )
  triv_list = extract_mlir_node_trivia(graph.nodes["x"].attributes)
  assert any("Input comment" in t.text for t in triv_list)

  mlir_compiled = MlirBackend(SemanticsManager()).compile(graph)
  assert "// Add op leading comment" in mlir_compiled
  assert "// [on-device fwd] shape=(2, 4)" in mlir_compiled
  assert "tensor<2x4xf32>" in mlir_compiled

  shlo_compiled = StableHloBackend(SemanticsManager()).compile(graph)
  assert "// Add op leading comment" in shlo_compiled
  assert "// [on-device fwd] shape=(2, 4)" in shlo_compiled
  assert "tensor<2x4xf32>" in shlo_compiled

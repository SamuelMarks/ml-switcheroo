"""Integration and unit tests for Hardware ISA Lifters and Multi-Branch DAG Extraction."""

import unittest

import libcst as cst

from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassInstruction,
  NvidiaSassMemory,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.frontends.nvidia_sass.lifter import NvidiaSassLifter
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.compiler.ir import LogicalGraph
from ml_switcheroo.core.compiler.isa.cst import IsaComment, IsaInstruction, IsaOperand
from ml_switcheroo.core.compiler.isa.lifter import (
  AnnotatedIsaRoundtripLifter,
  IsaLifter,
  RawIsaDataflowLifter,
  extract_instruction_reg_io,
  extract_on_device_kv_pairs,
)
from ml_switcheroo.core.graph import GraphExtractor


class TestHardwareLifters(unittest.TestCase):
  """Verify AnnotatedIsaRoundtripLifter, RawIsaDataflowLifter, and multi-branch DAG ordering."""

  def test_nvidia_sass_lifter_conv2d(self) -> None:
    """Verify raw NVIDIA SASS nested spatial loop lifts to Conv2d with loop bounds and SSA edges."""
    sass_code: str = "\x0a".join(
      [
        "MOV R5, 1;",
        "MOV R6, 2;",
        "L_KY_conv:",
        "  MOV R2, RZ;",
        "L_KX_conv:",
        "  FFMA R0, R5, R6, R0;",
        "  ISETP.LT.AND P0, PT, R2, 3, PT;",
        "  BRA L_KX_conv;",
        "FADD R7, R0, R5;",
      ]
    )
    ast_nodes = NvidiaSassParser(sass_code).parse().statements
    graph: LogicalGraph = NvidiaSassLifter().lift(ast_nodes)

    assert "conv2d_0" in graph.nodes
    conv_node = graph.nodes["conv2d_0"]
    assert conv_node.op_type == "Conv2d"
    assert conv_node.attributes["kernel_size"] == 3
    assert conv_node.attributes["arg_2"] == 3
    assert conv_node.attributes["dest_reg"] == "R0"
    assert conv_node.attributes["loop_labels"] == ["L_KY_conv", "L_KX_conv"]

    # Independent R5 and R6 MOVs must NOT have a fake sequential edge R5 -> R6!
    edges = [(e.source, e.target, e.target_idx) for e in graph.edges]
    assert ("R5", "R6", 0) not in edges
    # Conv2d receives R5 (port 0) and R6 (port 1) via SSA use-def
    assert ("R5", "conv2d_0", 0) in edges
    assert ("R6", "conv2d_0", 1) in edges
    # Post-loop residual FADD R7, R0, R5 receives conv2d_0 (port 0) and R5 (port 1)
    assert ("conv2d_0", "R7", 0) in edges
    assert ("R5", "R7", 1) in edges

  def test_rdna_lifter_conv2d(self) -> None:
    """Verify raw AMD RDNA spatial loop lifts to Conv2d with recovered kernel bound and SSA edges."""
    rdna_code: str = "\x0a".join(
      [
        "v_mov_b32 v2, 1",
        "v_mov_b32 v3, 2",
        "L_KY_conv:",
        "s_mov_b32 s0, 0",
        "L_KX_conv:",
        "v_fmac_f32 v1, v2, v3",
        "s_cmp_lt_i32 s0, 5",
        "s_cbranch_vccnz L_KX_conv",
        "v_add_f32 v4, v1, v2",
      ]
    )
    ast_nodes = RdnaParser(rdna_code).parse().statements
    graph: LogicalGraph = RdnaLifter().lift(ast_nodes)

    assert "conv2d_0" in graph.nodes
    conv_node = graph.nodes["conv2d_0"]
    assert conv_node.op_type == "Conv2d"
    assert conv_node.attributes["k"] == 5
    assert conv_node.attributes["arg_2"] == 5
    assert conv_node.attributes["dest_reg"] == "v1"

    edges = [(e.source, e.target, e.target_idx) for e in graph.edges]
    assert ("inst_0", "conv2d_0", 0) in edges
    assert ("inst_1", "conv2d_0", 1) in edges
    assert ("conv2d_0", "inst_2", 0) in edges
    assert ("inst_0", "inst_2", 1) in edges

  def test_raw_lifter_linear_sum_unanchored_and_ssa_redefinition(self) -> None:
    """Verify RawIsaDataflowLifter classifies Linear and Sum loops, memory loads, and SSA rewrites."""
    sass_linear_and_sum = "\x0a".join(
      [
        ".text",
        "// Zero Accumulator",
        "// Leading kernel note",
        "LDG.E R1, [R8 + 0x4];",
        "MOV R1, R8;",
        "FADD R1, R1, c[0x0][0x4];",
        "FADD R1, R1, R8;",
        "BRA L_AFTER_SKIP;",
        "L_AFTER_SKIP:",
        "BRA L_EXTERNAL_EXIT;",
        "// Loop header trivia",
        "L_GEMM:",
        "  FFMA R3, R1, R2, R3;",
        "  ISETP.LT.AND P0, PT, R4, 64, PT;",
        "  BRA L_GEMM;",
        "L_REDUCE:",
        "  FADD R5, R5, R3;",
        "  ISETP.LT.AND P0, PT, R6, 16, PT;",
        "  BRA L_REDUCE;",
        "// Trailing note",
      ]
    )
    nodes = NvidiaSassParser(sass_linear_and_sum).parse().statements
    raw_graph = NvidiaSassLifter(mode="raw").lift(nodes)

    assert raw_graph.nodes["linear_0"].op_type == "Linear"
    assert raw_graph.nodes["linear_0"].attributes["in_features"] == 64
    assert raw_graph.nodes["linear_0"].attributes["leading_trivia"] == "Loop header trivia"
    assert raw_graph.nodes["sum_0"].op_type == "Sum"
    assert raw_graph.nodes["sum_0"].attributes["elements"] == 16
    # SSA redefinition R1 -> R1_1 -> R1_2 -> linear_0 -> sum_0
    assert "R1" in raw_graph.nodes
    assert "R1_1" in raw_graph.nodes
    assert "R1_2" in raw_graph.nodes
    edge_pairs = [(e.source, e.target) for e in raw_graph.edges]
    assert ("R1", "R1_1") in edge_pairs
    assert ("R1_1", "R1_2") in edge_pairs
    assert ("R1_2", "linear_0") in edge_pairs
    assert ("linear_0", "sum_0") in edge_pairs
    assert raw_graph.nodes["sum_0"].attributes["leading_trivia"] == "Trailing note"

    # Nested overlapping loops starting at index 0 + spin loop with zero live-ins and zero acc_reg
    nested_spin = (
      NvidiaSassParser("L_OUTER:\x0aL_INNER:\x0a  NOP;\x0a  BRA L_INNER;\x0a  BRA L_OUTER;\x0a").parse().statements
    )
    spin_graph = NvidiaSassLifter(mode="raw").lift(nested_spin)
    assert "conv2d_0" in spin_graph.nodes
    assert "source_regs" not in spin_graph.nodes["conv2d_0"].attributes
    assert "dest_reg" not in spin_graph.nodes["conv2d_0"].attributes

    # Unanchored backward branch snippet without header label
    unanchored = RdnaParser("v_add_f32 v1, v2, v3\x0as_cbranch_vccnz L_KX_conv\x0a").parse().statements
    unanchored_graph = RdnaLifter(mode="raw").lift(unanchored)
    assert "conv2d_0" in unanchored_graph.nodes
    assert unanchored_graph.nodes["conv2d_0"].op_type == "Conv2d"

  def test_annotated_roundtrip_multi_input_and_base_lifter_fallbacks(self) -> None:
    """Verify AnnotatedIsaRoundtripLifter multi-input provenance, KV metadata, and base fallbacks."""
    annotated_nodes = [
      NvidiaSassComment(text="// Input x -> R0"),
      NvidiaSassComment(text="// Input skip -> R1"),
      NvidiaSassComment(text="// User trivia before block"),
      NvidiaSassComment(text="// BEGIN Add (res_add)"),
      NvidiaSassComment(text="// [on-device shape] in_features=8, out_features=8, dtype=f32"),
      NvidiaSassComment(text="// [on-device stride] in_channels=3, k=3"),
      NvidiaSassInstruction(
        opcode="FADD",
        operands=[NvidiaSassRegister(name="R2"), NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1")],
      ),
      NvidiaSassComment(text="// END Add (res_add)"),
      NvidiaSassInstruction(
        opcode="FMUL",
        operands=[NvidiaSassRegister(name="R2"), NvidiaSassRegister(name="R2"), NvidiaSassRegister(name="R0")],
      ),
      NvidiaSassComment(text="// Trailing trivia on last node"),
      NvidiaSassComment(text="// Return: R2"),
    ]
    graph = NvidiaSassLifter(mode="annotated").lift(annotated_nodes)
    # Input x and Input skip must have NO incoming edges
    assert graph.nodes["x"].inputs == []
    assert graph.nodes["skip"].inputs == []
    # res_add receives x (port 0) and skip (port 1)
    assert graph.nodes["res_add"].inputs == ["x", "skip"]
    assert graph.nodes["res_add"].attributes["dtype"] == "f32"
    # Disambiguated R2 instruction receives res_add (port 0) and x (port 1)
    assert graph.nodes["R2"].inputs == ["res_add", "x"]
    assert graph.nodes["output"].inputs == ["R2"]

    # Base IsaLifter and direct AnnotatedIsaRoundtripLifter / RawIsaDataflowLifter without custom resolver
    base_lifter: IsaLifter[IsaInstruction, IsaInstruction, IsaComment] = IsaLifter()
    inst = IsaInstruction(opcode="ADD", operands=[])
    base_graph = base_lifter.lift([IsaComment(text="; leading note"), inst, IsaComment(text="; trailing note")])
    assert "inst_0" in base_graph.nodes

    direct_ann: AnnotatedIsaRoundtripLifter[IsaInstruction, IsaInstruction, IsaComment] = AnnotatedIsaRoundtripLifter()
    ann_g = direct_ann.lift([IsaInstruction(opcode="NOP", operands=[]), IsaComment(text="; end note")])
    assert ann_g.nodes["inst_0"].attributes["leading_trivia"] == "end note"

    direct_raw: RawIsaDataflowLifter[IsaInstruction, IsaInstruction, IsaComment] = RawIsaDataflowLifter()
    raw_g = direct_raw.lift([IsaComment(text="; pre note"), IsaInstruction(opcode="MOV", operands=[])])
    assert raw_g.nodes["inst_0"].attributes["leading_trivia"] == "pre note"

    # Exercise extract_instruction_reg_io memory & non-named operand paths + extract_on_device_kv_pairs
    class BareOperand(IsaOperand):
      """Minimal operand without a ``name`` attribute."""

      def to_text(self) -> str:
        """Return literal operand text.

        Returns:
            str: Operand literal string.
        """
        return "0"

    mem_inst = NvidiaSassInstruction(
      opcode="STG.E",
      operands=[NvidiaSassMemory(base=NvidiaSassRegister(name="R4"), offset=0), BareOperand()],
    )
    dst, srcs = extract_instruction_reg_io(mem_inst)
    assert dst is None
    assert srcs == ["R4"]
    assert extract_on_device_kv_pairs("[on-device shape] bad_entry, =empty, k=3") == {"k": 3}

  def test_graph_extractor_multi_branch_residual_and_qkv_fanout(self) -> None:
    """Verify GraphExtractor preserves multi-input operand order for residuals, Q/K/V, and self-edges."""
    code = "\x0a".join(
      [
        "class ResidualAttentionBlock:",
        "    def __init__(self):",
        "        self.q_proj = nn.Linear(16, 16)",
        "        self.k_proj = nn.Linear(16, 16)",
        "        self.v_proj = nn.Linear(16, 16)",
        "        self.conv = nn.Conv2d(16, 16, 3)",
        "",
        "    def forward(self, x):",
        "        q = self.q_proj(x)",
        "        k = self.k_proj(x)",
        "        v = self.v_proj(x)",
        "        merged = torch.cat([q, k, v, *extra], 1)",
        "        attn = (q @ k) / v",
        "        res1 = self.conv(x) + x",
        "        res2 = res1 + x",
        "        diff = res2 - (attn * merged)",
        "        dup = torch.add(x, x)",
        "        _ = [q, k] + 1",
        "        _ = torch.abs(relu(x))",
        "        ignored_bit = x | x",
        "        return diff, dup",
      ]
    )
    extractor = GraphExtractor()
    cst.parse_module(code).visit(extractor)
    g = extractor.graph

    # torch.cat([q, k, v, *extra], 1) preserves exact operand order [q_proj, k_proj, v_proj]
    assert g.nodes["func_cat"].inputs == ["q_proj", "k_proj", "v_proj"]
    # Residual adds preserve [conv, Input_x] and [func_add, Input_x]
    assert g.nodes["func_add"].inputs == ["conv", "Input_x"]
    assert g.nodes["func_add_1"].inputs == ["func_add", "Input_x"]
    assert g.nodes["func_matmul"].inputs == ["q_proj", "k_proj"]
    assert g.nodes["func_div"].inputs == ["func_matmul", "v_proj"]
    assert g.nodes["func_mul"].inputs == ["func_div", "func_cat"]
    assert g.nodes["func_sub"].inputs == ["func_add_1", "func_mul"]
    # Duplicate self-edge torch.add(x, x) keeps BOTH ports (target_idx 0 and 1)!
    assert g.nodes["func_add_2"].inputs == ["Input_x", "Input_x"]
    # Tuple return (diff, dup) preserves [func_sub, func_add_2] on output
    assert g.nodes["output"].inputs == ["func_sub", "func_add_2"]

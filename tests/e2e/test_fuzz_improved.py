"""Property-based Hypothesis fuzzing suite for `ml_switcheroo` custom domain CST parsers."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml_switcheroo.core.compiler.backends.cpp.parser import CppParser
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction
from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassParser
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction
from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaParser
from ml_switcheroo.core.mlir.parser import MlirParser
from ml_switcheroo.core.tikz.parser import TikzParser

_NL = chr(10)
_BS = chr(92)


@given(
  dst_idx=st.integers(min_value=0, max_value=15),
  src_a=st.integers(min_value=0, max_value=15),
  src_b=st.integers(min_value=0, max_value=15),
  opcode=st.sampled_from(["FADD", "FMUL", "IADD3"]),
)
@settings(max_examples=25, deadline=None)
def test_nvidia_sass_parser_roundtrip_fuzz(dst_idx: int, src_a: int, src_b: int, opcode: str) -> None:
  """Verify `NvidiaSassParser` parses generated instructions and roundtrips `to_text()` losslessly.

  Args:
      dst_idx: Destination general-purpose register index.
      src_a: First source register index.
      src_b: Second source register index.
      opcode: Sampled SASS mnemonic.
  """
  line = f"{opcode} R{dst_idx}, R{src_a}, R{src_b};"
  mod = NvidiaSassParser(line).parse()
  assert len(mod.statements) == 1
  stmt = mod.statements[0]
  assert isinstance(stmt, NvidiaSassInstruction)
  assert stmt.opcode == opcode
  assert mod.to_text().strip() == line


@given(
  dst_idx=st.integers(min_value=0, max_value=15),
  src_a=st.integers(min_value=0, max_value=15),
  src_b=st.integers(min_value=0, max_value=15),
  opcode=st.sampled_from(["v_add_f32", "v_mul_f32", "v_sub_f32"]),
)
@settings(max_examples=25, deadline=None)
def test_rdna_parser_roundtrip_fuzz(dst_idx: int, src_a: int, src_b: int, opcode: str) -> None:
  """Verify `RdnaParser` parses generated VALU instructions and roundtrips `to_text()` losslessly.

  Args:
      dst_idx: Destination VGPR index.
      src_a: First source VGPR index.
      src_b: Second source VGPR index.
      opcode: Sampled RDNA VALU mnemonic.
  """
  line = f"{opcode} v{dst_idx}, v{src_a}, v{src_b}"
  mod = RdnaParser(line).parse()
  assert len(mod.statements) == 1
  stmt = mod.statements[0]
  assert isinstance(stmt, RdnaInstruction)
  assert stmt.opcode == opcode
  assert mod.to_text().strip() == line


@given(
  op_name=st.sampled_from(["stablehlo.abs", "stablehlo.exponential", "stablehlo.cosine"]),
  dim=st.integers(min_value=1, max_value=64),
)
@settings(max_examples=15, deadline=None)
def test_mlir_parser_roundtrip_fuzz(op_name: str, dim: int) -> None:
  """Verify `MlirParser` parses parameterized StableHLO modules and roundtrips `to_text()`.

  Args:
      op_name: Sampled StableHLO unary operation name.
      dim: Tensor dimension size.
  """
  mlir_src = _NL.join(
    [
      "module {",
      f"  func.func @main(%arg0: tensor<{dim}xf32>) -> tensor<{dim}xf32> {{",
      f'    %0 = "{op_name}"(%arg0) : (tensor<{dim}xf32>) -> tensor<{dim}xf32>',
      f"    return %0 : tensor<{dim}xf32>",
      "  }",
      "}",
      "",
    ]
  )
  parsed = MlirParser(mlir_src).parse()
  rendered = parsed.to_text()
  assert op_name in rendered
  assert f"tensor<{dim}xf32>" in rendered


@given(ret_val=st.integers(min_value=0, max_value=100))
@settings(max_examples=10, deadline=None)
def test_cpp_and_tikz_parser_invariants(ret_val: int) -> None:
  """Verify `CppParser` parses C++ functions and `TikzParser` rejects malformed input cleanly.

  Args:
      ret_val: Integer literal returned by generated C++ function.
  """
  cpp_code = _NL.join(["int compute(int x) {", f"  return {ret_val};", "}", ""])
  tu = CppParser(cpp_code).parse()
  assert str(ret_val) in tu.to_text()

  with pytest.raises(ValueError, match="Failed to parse TikZ"):
    TikzParser(f"{_BS}begin{{tikzpicture}} {_BS}node UNCLOSED").parse()


def test_isa_operands_transformer_ignores_non_operand_tokens() -> None:
  """Verify `NvidiaSassTransformer.operands` and `RdnaTransformer.operands` skip unexpected tokens safely."""
  from lark import Token

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.parser import NvidiaSassTransformer
  from ml_switcheroo.core.compiler.frontends.rdna.parser import RdnaTransformer

  assert NvidiaSassTransformer().operands([Token("SEMI", ";")]) == []
  assert RdnaTransformer().operands([Token("SEMI", ";")]) == []

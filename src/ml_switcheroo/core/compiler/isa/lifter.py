"""Unified ISA Lifters: Annotated Roundtrip Mode and Raw Dataflow Mode.

Provides two explicit lifting strategies and a unified ``IsaLifter`` facade:

1. ``AnnotatedIsaRoundtripLifter``: Reconstructs high-level ``LogicalGraph`` blocks
   when explicit semantic provenance comments (``BEGIN <Op>``, ``Input``, ``Return``)
   are present, while respecting register provenance for multi-input edges.
2. ``RawIsaDataflowLifter``: Lifts unannotated hardware assembly by constructing genuine
   SSA/register use-def dataflow edges (``dst_reg -> defining_node_id``) without fake
   sequential chaining and recognizing canonical ``Conv2d`` / ``Linear`` / ``Sum`` loop
   idioms via control-flow backward branches and ``IsaAnalyzer`` bounds recovery.
"""

import re
import typing
from dataclasses import dataclass, field

from ml_switcheroo.core.compiler.frontends.semantic_parser import (
  SemanticBegin,
  SemanticCommentParser,
  SemanticEnd,
  SemanticInput,
  SemanticReturn,
  SemanticUnmapped,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.isa.analysis import IsaAnalyzer
from ml_switcheroo.core.compiler.isa.cst import (
  IsaComment,
  IsaInstruction,
  IsaLabel,
  IsaMemory,
  IsaNode,
  IsaOperand,
  IsaRegister,
)

NodeT = typing.TypeVar("NodeT", bound=IsaNode)
InstructionT = typing.TypeVar("InstructionT", bound=IsaInstruction)
CommentT = typing.TypeVar("CommentT", bound=IsaComment)

IsaLiftMode: typing.TypeAlias = typing.Literal["auto", "annotated", "raw"]

_ZERO_OR_CONST_REGS: frozenset[str] = frozenset({"RZ", "RZS", "URZ", "PT", "UPT", "off", "glc", "slc"})

_INTERNAL_BOILERPLATE_COMMENTS: frozenset[str] = frozenset(
  {
    "Zero Accumulator",
    "Calc Address & Load Image",
    "Calc Address & Load Image Pixel",
    "Calc Address & Load Weight",
    "Load Weight",
    "Add Bias",
    "Q, K, V Projections",
    "Attention = Softmax(Q*K^T) * V",
    "Self Attention",
    "Feed Forward",
  }
)

_REG_TOKEN_RE = re.compile(r"^(?:[RU]?R\d+|P\d+|[sv](?:\[\d+:\d+\]|\d+))$")


@typing.runtime_checkable
class _NamedNodeProtocol(typing.Protocol):
  """Protocol for ISA CST operands exposing a ``name`` attribute.

  Attributes:
      name: Operand or label identifier (`str`).
  """

  name: str


@typing.runtime_checkable
class _MemoryBaseProtocol(typing.Protocol):
  """Protocol for ISA memory operands exposing a ``base`` register or identifier.

  Attributes:
      base: Base address register or string (`IsaNode | str`).
  """

  base: IsaNode | str


def _canonical_reg_name(op: IsaNode | IsaOperand | IsaRegister | str) -> str | None:
  """Extract a canonical physical register name from ``op`` if it represents a register.

  Args:
      op: Operand, register CST node, or raw string.

  Returns:
      str | None: Canonical register identifier (excluding zero/constant regs), or ``None``.
  """
  if isinstance(op, str):
    cand = op.strip()
  elif isinstance(op, IsaRegister):
    cand = op.name if isinstance(op, _NamedNodeProtocol) and op.name else op.to_text()
  elif isinstance(op, _NamedNodeProtocol):
    cand = op.name.strip()
  else:
    return None

  cand = cand.lstrip(", \t\r\n-|").rstrip(" \t\r\n|").strip()
  if not cand or cand in _ZERO_OR_CONST_REGS:
    return None
  if isinstance(op, IsaRegister) or _REG_TOKEN_RE.match(cand):
    return cand
  return None


def _is_branch_opcode(opcode: str) -> bool:
  """Check whether ``opcode`` is a control-flow branch instruction.

  Args:
      opcode: Assembly instruction mnemonic.

  Returns:
      bool: ``True`` if ``opcode`` is a conditional or unconditional branch.
  """
  up = opcode.upper()
  return up == "BRA" or up.startswith(("BRA.", "S_CBRANCH")) or up == "S_BRANCH"


def _is_non_dest_opcode(opcode: str) -> bool:
  """Check whether ``opcode`` does not write a GPR destination in ``operands[0]``.

  Args:
      opcode: Assembly instruction mnemonic.

  Returns:
      bool: ``True`` for branches, stores, barriers, waits, compares-without-dest, and terminators.
  """
  up = opcode.upper()
  if _is_branch_opcode(opcode):
    return True
  non_dest_prefixes = (
    "ST",
    "GLOBAL_STORE",
    "BUFFER_STORE",
    "DS_WRITE",
    "S_CMP",
    "V_CMP",
    "S_ENDPGM",
    "S_WAITCNT",
    "S_NOP",
    "S_BARRIER",
    "EXIT",
    "RET",
    "NOP",
    "BAR",
    "MEMBAR",
  )
  return up.startswith(non_dest_prefixes)


def extract_instruction_reg_io(inst: IsaInstruction) -> tuple[str | None, list[str]]:
  """Extract ``(dst_reg, src_regs)`` in operand order from an ``IsaInstruction``.

  Args:
      inst: Hardware ISA instruction CST node.

  Returns:
      tuple[str | None, list[str]]: Optional written destination register and ordered source registers.
  """
  if not inst.operands:
    return None, []

  up = inst.opcode.upper()
  if up.startswith(("ISETP", "FSETP")):
    dst_reg = _canonical_reg_name(inst.operands[0])
    src_operands = inst.operands[2:] if len(inst.operands) > 2 else []
  elif _is_non_dest_opcode(inst.opcode):
    dst_reg = None
    src_operands = list(inst.operands)
  else:
    dst_reg = _canonical_reg_name(inst.operands[0])
    src_operands = list(inst.operands[1:])

  src_regs: list[str] = []
  for op in src_operands:
    if isinstance(op, IsaMemory) and isinstance(op, _MemoryBaseProtocol):
      base_reg = _canonical_reg_name(op.base)
      if base_reg is not None:
        src_regs.append(base_reg)
    else:
      reg_name = _canonical_reg_name(op)
      if reg_name is not None:
        src_regs.append(reg_name)
  return dst_reg, src_regs


def _extract_branch_target(inst: IsaInstruction) -> str | None:
  """Extract the target label identifier from a branch instruction.

  Args:
      inst: Hardware ISA instruction CST node.

  Returns:
      str | None: Target label name if ``inst`` is a branch to a named label, else ``None``.
  """
  if not _is_branch_opcode(inst.opcode):
    return None
  for op in reversed(inst.operands):
    if isinstance(op, IsaLabel) and op.name:
      return op.name
    if isinstance(op, _NamedNodeProtocol) and op.name and not _REG_TOKEN_RE.match(op.name):
      return op.name
  return None


def _parse_tail_register(tail_text: str | None) -> str | None:
  """Extract a register name from a semantic marker tail string (e.g., ``' R0'``).

  Args:
      tail_text: Raw marker tail text.

  Returns:
      str | None: Parsed register name if valid, otherwise ``None``.
  """
  if not tail_text:
    return None
  token = tail_text.strip().split()[0] if tail_text.strip() else ""
  return _canonical_reg_name(token)


def extract_on_device_kv_pairs(text: str) -> dict[str, str | int | float]:
  """Extract ``key=value`` pairs from ``[on-device shape]`` or ``[on-device stride]`` comments.

  Args:
      text: Stripped comment text.

  Returns:
      dict[str, str | int | float]: Parsed integer or string attribute pairs.
  """
  extracted: dict[str, str | int | float] = {}
  for prefix in ("[on-device shape]", "[on-device stride]"):
    if text.startswith(prefix):
      payload = text[len(prefix) :].strip()
      for part in payload.split(","):
        if "=" in part:
          k_raw, v_raw = part.split("=", 1)
          key = k_raw.strip()
          val_str = v_raw.strip()
          if key and val_str:
            if val_str.lstrip("-").isdigit():
              extracted[key] = int(val_str)
            else:
              extracted[key] = val_str
  return extracted


def _is_trivia_comment(text: str) -> bool:
  """Determine whether a non-marker comment should be preserved as leading trivia.

  Args:
      text: Stripped comment body.

  Returns:
      bool: ``True`` if ``text`` is user/source trivia rather than synthesis boilerplate.
  """
  return (
    bool(text)
    and not text.startswith("RDNA Code Generation Initialized")
    and not text.startswith("NVIDIA_SASS Code Generation")
    and text not in _INTERNAL_BOILERPLATE_COMMENTS
  )


@dataclass(frozen=True)
class _DetectedLoopSpan:
  """Contiguous statement span ``[start_idx, end_idx]`` representing a natural backward loop.

  Attributes:
      start_idx: Inclusive statement start index.
      end_idx: Inclusive statement end index (the backward branch).
      labels: Tuple of loop header label names covered by this span.
  """

  start_idx: int
  end_idx: int
  labels: tuple[str, ...]


OrphanResolverCallable: typing.TypeAlias = typing.Callable[
  [InstructionT, int],
  tuple[str, str, dict[str, str | int | float]],
]


@dataclass
class _AnnotatedLiftState:
  """Mutable graph-construction state for ``AnnotatedIsaRoundtripLifter``.

  Attributes:
      graph: Target ``LogicalGraph`` under construction.
      seen_ids: Set of already-inserted node identifiers.
      input_ids: Set of declared ``Input`` node identifiers.
      pending_trivia: Buffer of leading trivia comments awaiting attachment.
      previous_node_id: Preceding node ID for linear fallback wiring.
      reg_defs: Active register-to-producer-node map.
  """

  graph: LogicalGraph
  seen_ids: set[str] = field(default_factory=set)
  input_ids: set[str] = field(default_factory=set)
  pending_trivia: list[str] = field(default_factory=list)
  previous_node_id: str | None = None
  reg_defs: dict[str, str] = field(default_factory=dict)


class AnnotatedIsaRoundtripLifter(typing.Generic[NodeT, InstructionT, CommentT]):
  """Lift ISA nodes carrying explicit ``; BEGIN <Op>`` / ``Input`` / ``Return`` markers.

  Preserves multi-input operand ordering when register provenance is available in
  ``Input <var> -> <reg>`` and instruction operands, falling back to block-order edges
  for macro blocks whose internal registers are self-contained.

  Attributes:
      default_graph_name: Default name assigned to lifted ``LogicalGraph`` instances.
      comment_type: Dialect comment CST class.
      instruction_type: Dialect instruction CST class.
      analyzer_cls: Dialect instruction analyzer class.
      comment_parser: Parser for structured semantic comments.
  """

  def __init__(
    self,
    *,
    default_graph_name: str = "DecompiledModel",
    comment_type: type[IsaComment] = IsaComment,
    instruction_type: type[IsaInstruction] = IsaInstruction,
    analyzer_cls: type[IsaAnalyzer] = IsaAnalyzer,
    orphan_resolver: OrphanResolverCallable[InstructionT] | None = None,
    comment_parser: SemanticCommentParser | None = None,
  ) -> None:
    """Initialize ``AnnotatedIsaRoundtripLifter``.

    Args:
        default_graph_name: Default graph name.
        comment_type: Dialect comment CST node type.
        instruction_type: Dialect instruction CST node type.
        analyzer_cls: Dialect ``IsaAnalyzer`` subclass.
        orphan_resolver: Callback resolving unenclosed instructions to ``(id, kind, meta)``.
        comment_parser: Shared ``SemanticCommentParser`` instance.
    """
    self.default_graph_name: str = default_graph_name
    self.comment_type: type[IsaComment] = comment_type
    self.instruction_type: type[IsaInstruction] = instruction_type
    self.analyzer_cls: type[IsaAnalyzer] = analyzer_cls
    self._orphan_resolver: OrphanResolverCallable[InstructionT] | None = orphan_resolver
    self.comment_parser: SemanticCommentParser = comment_parser or SemanticCommentParser()

  def _is_instruction(self, node: IsaNode) -> typing.TypeGuard[InstructionT]:
    """Narrow ``node`` to ``InstructionT``.

    Args:
        node: Candidate ISA CST node.

    Returns:
        bool: ``True`` when ``node`` matches ``self.instruction_type``.
    """
    return isinstance(node, self.instruction_type)

  def _resolve_orphan(
    self,
    inst: InstructionT,
    counter: int,
  ) -> tuple[str, str, dict[str, str | int | float]]:
    """Resolve an unenclosed instruction via ``_orphan_resolver`` or default naming.

    Args:
        inst: Unenclosed instruction node.
        counter: Monotonic instruction counter.

    Returns:
        tuple[str, str, dict[str, str | int | float]]: ``(node_id, kind, meta)``.
    """
    if self._orphan_resolver is not None:
      return self._orphan_resolver(inst, counter)
    return (f"inst_{counter}", f"asm.{inst.opcode or 'unknown'}", {})

  def _finalize_annotated_block(
    self,
    block_kind: str,
    block_instructions: list[InstructionT],
    block_meta: dict[str, str | int | float],
    reg_defs: dict[str, str],
    input_ids: set[str],
  ) -> tuple[dict[str, str | int | float | list[str]], list[str] | None, str | None]:
    """Analyze an annotated instruction block and resolve register use-def sources.

    Args:
        block_kind: Semantic operation kind from ``BEGIN <kind>``.
        block_instructions: Instructions enclosed inside the block.
        block_meta: Accumulated on-device ``key=value`` attributes.
        reg_defs: Active register-to-producer-node map.
        input_ids: Set of declared ``SemanticInput`` node IDs.

    Returns:
        Tuple of ``(merged_metadata, explicit_sources_or_none, last_written_reg)``.
    """
    meta: dict[str, str | int | float | list[str]] = dict(self.analyzer_cls.analyze_block(block_kind, block_instructions))
    if block_meta:
      meta.update(block_meta)
      if "in_features" in meta and "out_features" in meta:
        meta.pop("arg_0", None)
      if ("in_channels" in meta or "out_channels" in meta) and ("k" in meta or "kernel_size" in meta):
        meta.pop("arg_2", None)

    block_defined: set[str] = set()
    block_sources: list[str] = []
    last_written_reg: str | None = None
    for b_inst in block_instructions:
      b_dst, b_srcs = extract_instruction_reg_io(b_inst)
      for s_reg in b_srcs:
        if s_reg not in block_defined and s_reg in reg_defs:
          prod = reg_defs[s_reg]
          if prod not in block_sources:
            block_sources.append(prod)
      if b_dst is not None:
        block_defined.add(b_dst)
        last_written_reg = b_dst

    explicit_sources = block_sources if (len(block_sources) > 1 and all(s in input_ids for s in block_sources)) else None
    return meta, explicit_sources, last_written_reg

  @staticmethod
  def _commit_annotated_node(
    state: _AnnotatedLiftState,
    node_id: str,
    kind: str,
    meta: typing.Mapping[str, str | int | float | list[str]] | None = None,
    *,
    explicit_sources: list[str] | None = None,
    is_input: bool = False,
  ) -> None:
    """Insert a ``LogicalNode`` into ``state.graph`` and wire incoming edges in operand order.

    Args:
        state: Active ``_AnnotatedLiftState`` holding graph, seen IDs, trivia, and previous node.
        node_id: Unique node identifier.
        kind: Logical operation type.
        meta: Optional attribute dictionary.
        explicit_sources: Ordered upstream producer IDs resolved from register use-def.
        is_input: Whether this node is a graph ``Input``.
    """
    if node_id in state.seen_ids:
      return

    attrs: dict[str, str | int | float | list[str]] = dict(meta) if meta else {}
    if state.pending_trivia and not is_input:
      attrs["leading_trivia"] = "\x0a".join(state.pending_trivia)
      state.pending_trivia.clear()

    state.graph.nodes[node_id] = LogicalNode(id=node_id, op_type=kind, attributes=attrs)
    state.seen_ids.add(node_id)

    if not is_input:
      if explicit_sources:
        for target_idx, src_id in enumerate(explicit_sources):
          state.graph.add_edge(src_id, node_id, target_idx=target_idx)
      elif state.previous_node_id is not None:
        state.graph.add_edge(state.previous_node_id, node_id, target_idx=0)

    state.previous_node_id = node_id

  def _lift_orphan_annotated_instruction(
    self,
    state: _AnnotatedLiftState,
    node: InstructionT,
    instruction_counter: int,
  ) -> None:
    """Lift a standalone unenclosed instruction inside an annotated ISA stream.

    Args:
        state: Active ``_AnnotatedLiftState`` under construction.
        node: Standalone instruction CST node.
        instruction_counter: Monotonic counter for unnamed instructions.
    """
    dest_name, kind, orphan_meta = self._resolve_orphan(node, instruction_counter)
    next_count = instruction_counter + 1
    actual_id = dest_name if dest_name not in state.seen_ids else f"{dest_name}_{next_count}"
    dst_reg, src_regs = extract_instruction_reg_io(node)
    resolved_srcs = [state.reg_defs[r] for r in src_regs if r in state.reg_defs]
    self._commit_annotated_node(
      state,
      actual_id,
      kind,
      orphan_meta,
      explicit_sources=resolved_srcs if resolved_srcs else None,
    )
    if dst_reg is not None:
      state.reg_defs[dst_reg] = actual_id

  def _handle_leaf_semantic_marker(
    self,
    state: _AnnotatedLiftState,
    marker: SemanticInput | SemanticUnmapped | SemanticReturn,
  ) -> None:
    """Commit ``SemanticInput``, ``SemanticUnmapped``, or ``SemanticReturn`` markers into ``state.graph``.

    Args:
        state: Active ``_AnnotatedLiftState`` under construction.
        marker: Parsed leaf semantic marker comment.
    """
    if isinstance(marker, SemanticInput):
      self._commit_annotated_node(state, marker.name, "Input", {"name": marker.name}, is_input=True)
      state.input_ids.add(marker.name)
      in_reg = _parse_tail_register(marker.tail.text if marker.tail else None)
      if in_reg is not None:
        state.reg_defs[in_reg] = marker.name
      return
    if isinstance(marker, SemanticUnmapped):
      unmapped_meta: dict[str, str | int | float] = {"arg_1": 1} if "flatten" in marker.api else {}
      self._commit_annotated_node(state, marker.id, marker.api, unmapped_meta)
      return
    if "output" not in state.seen_ids:
      ret_reg = _parse_tail_register(marker.tail.text if marker.tail else None)
      ret_sources = [state.reg_defs[ret_reg]] if (ret_reg is not None and ret_reg in state.reg_defs) else None
      self._commit_annotated_node(state, "output", "Output", explicit_sources=ret_sources)

  def lift(self, nodes: typing.Sequence[IsaNode]) -> LogicalGraph:
    """Reconstruct a ``LogicalGraph`` from semantically annotated ISA CST nodes.

    Args:
        nodes: Sequence of ISA comments, labels, directives, and instructions.

    Returns:
        LogicalGraph: Reconstructed computational graph.
    """
    state = _AnnotatedLiftState(graph=LogicalGraph(name=self.default_graph_name))

    current_block_id: str | None = None
    current_block_kind: str | None = None
    current_instructions: list[InstructionT] = []
    current_block_meta: dict[str, str | int | float] = {}
    instruction_counter = 0

    for node in nodes:
      if isinstance(node, self.comment_type):
        text = node.text.lstrip(";/ ").strip()
        marker = self.comment_parser.parse(text)
        if not marker:
          if current_block_id is not None:
            current_block_meta.update(extract_on_device_kv_pairs(text))
          if _is_trivia_comment(text):
            state.pending_trivia.append(text)
          continue

        if isinstance(marker, (SemanticInput, SemanticUnmapped, SemanticReturn)):
          self._handle_leaf_semantic_marker(state, marker)
        elif isinstance(marker, SemanticBegin):
          current_block_kind, current_block_id = marker.kind, marker.id
          current_instructions, current_block_meta = [], {}
        elif isinstance(marker, SemanticEnd) and marker.id == current_block_id and current_block_kind:
          meta, explicit_sources, last_written_reg = self._finalize_annotated_block(
            current_block_kind, current_instructions, current_block_meta, state.reg_defs, state.input_ids
          )
          self._commit_annotated_node(
            state,
            current_block_id,
            current_block_kind,
            meta,
            explicit_sources=explicit_sources,
          )
          if last_written_reg is not None:
            state.reg_defs[last_written_reg] = current_block_id
          current_block_id, current_block_kind = None, None
          current_instructions, current_block_meta = [], {}

      elif current_block_id is not None and self._is_instruction(node):
        current_instructions.append(node)
      elif self._is_instruction(node):
        self._lift_orphan_annotated_instruction(state, node, instruction_counter)
        instruction_counter += 1

    if state.pending_trivia and state.graph.nodes:
      list(state.graph.nodes.values())[-1].attributes["leading_trivia"] = "\x0a".join(state.pending_trivia)
      state.pending_trivia.clear()

    return state.graph


class RawIsaDataflowLifter(typing.Generic[NodeT, InstructionT, CommentT]):
  """Lift unannotated hardware ISA via SSA register use-def edges and loop idiom recovery.

  Unlike ``AnnotatedIsaRoundtripLifter``, ``RawIsaDataflowLifter`` NEVER wires sequential
  ``previous_node_id`` chains: directed ``LogicalEdge`` connections are created strictly
  from true register def-use dependencies (preserving multi-input operand order and DAG
  fan-out), while natural backward-branch loops are recognized as ``Conv2d``, ``Linear``,
  or ``Sum`` idioms analyzed by ``IsaAnalyzer``.

  Attributes:
      default_graph_name: Default name assigned to lifted ``LogicalGraph`` instances.
      comment_type: Dialect comment CST class.
      instruction_type: Dialect instruction CST class.
      analyzer_cls: Dialect instruction analyzer class.
  """

  def __init__(
    self,
    *,
    default_graph_name: str = "DecompiledModel",
    comment_type: type[IsaComment] = IsaComment,
    instruction_type: type[IsaInstruction] = IsaInstruction,
    analyzer_cls: type[IsaAnalyzer] = IsaAnalyzer,
    orphan_resolver: OrphanResolverCallable[InstructionT] | None = None,
  ) -> None:
    """Initialize ``RawIsaDataflowLifter``.

    Args:
        default_graph_name: Default graph name.
        comment_type: Dialect comment CST node type.
        instruction_type: Dialect instruction CST node type.
        analyzer_cls: Dialect ``IsaAnalyzer`` subclass.
        orphan_resolver: Callback resolving standalone instructions to ``(id, kind, meta)``.
    """
    self.default_graph_name: str = default_graph_name
    self.comment_type: type[IsaComment] = comment_type
    self.instruction_type: type[IsaInstruction] = instruction_type
    self.analyzer_cls: type[IsaAnalyzer] = analyzer_cls
    self._orphan_resolver: OrphanResolverCallable[InstructionT] | None = orphan_resolver

  def _is_instruction(self, node: IsaNode) -> typing.TypeGuard[InstructionT]:
    """Narrow ``node`` to ``InstructionT``.

    Args:
        node: Candidate ISA CST node.

    Returns:
        bool: ``True`` when ``node`` matches ``self.instruction_type``.
    """
    return isinstance(node, self.instruction_type)

  def _resolve_orphan(
    self,
    inst: InstructionT,
    counter: int,
  ) -> tuple[str, str, dict[str, str | int | float]]:
    """Resolve a standalone instruction via ``_orphan_resolver`` or default naming.

    Args:
        inst: Standalone instruction node.
        counter: Monotonic instruction counter.

    Returns:
        tuple[str, str, dict[str, str | int | float]]: ``(node_id, kind, meta)``.
    """
    if self._orphan_resolver is not None:
      return self._orphan_resolver(inst, counter)
    return (f"inst_{counter}", f"asm.{inst.opcode or 'unknown'}", {})

  def _detect_loop_spans(self, nodes: typing.Sequence[IsaNode]) -> dict[int, _DetectedLoopSpan]:
    """Identify natural backward-branch loop spans indexed by start position.

    Args:
        nodes: Sequence of ISA CST nodes.

    Returns:
        dict[int, _DetectedLoopSpan]: Non-overlapping outer loop spans keyed by ``start_idx``.
    """
    label_positions: dict[str, int] = {}
    for idx, node in enumerate(nodes):
      if isinstance(node, IsaLabel) and node.name:
        label_positions[node.name] = idx

    raw_spans: list[_DetectedLoopSpan] = []
    for idx, node in enumerate(nodes):
      if not self._is_instruction(node):
        continue
      target_lbl = _extract_branch_target(node)
      if target_lbl is None:
        continue
      if target_lbl in label_positions and label_positions[target_lbl] <= idx:
        start_idx = label_positions[target_lbl]
        labels: list[str] = [target_lbl]
        # Absorb immediately preceding spatial loop header (e.g. L_KY_conv: MOV R2, RZ; L_KX_conv:)
        scan = start_idx - 1
        while scan >= 0:
          prev = nodes[scan]
          if isinstance(prev, IsaLabel) and prev.name:
            start_idx = scan
            labels.insert(0, prev.name)
            break
          if self._is_instruction(prev) and not _is_branch_opcode(prev.opcode):
            scan -= 1
            continue
          break
        raw_spans.append(_DetectedLoopSpan(start_idx=start_idx, end_idx=idx, labels=tuple(labels)))
      elif "conv" in target_lbl.lower() or "gemm" in target_lbl.lower() or "dot" in target_lbl.lower():
        # Unanchored backward branch snippet (e.g. partial basic-block test without header label)
        raw_spans.append(_DetectedLoopSpan(start_idx=0, end_idx=idx, labels=(target_lbl,)))

    # Merge overlapping/nested loop spans into outermost canonical spans
    raw_spans.sort(key=lambda sp: (sp.start_idx, -sp.end_idx))
    merged: list[_DetectedLoopSpan] = []
    for sp in raw_spans:
      if not merged or sp.start_idx > merged[-1].end_idx:
        merged.append(sp)
      else:
        prev_sp = merged[-1]
        combined_labels = tuple(dict.fromkeys((*prev_sp.labels, *sp.labels)))
        merged[-1] = _DetectedLoopSpan(
          start_idx=prev_sp.start_idx,
          end_idx=max(prev_sp.end_idx, sp.end_idx),
          labels=combined_labels,
        )
    return {sp.start_idx: sp for sp in merged}

  @staticmethod
  def _classify_loop_idiom(
    labels: tuple[str, ...],
    insts: typing.Sequence[IsaInstruction],
  ) -> str:
    """Classify a loop's high-level neural idiom from loop structure and ALU opcodes.

    Args:
        labels: Loop header/target label names.
        insts: Instructions inside the loop body.

    Returns:
        str: Canonical operation kind (``"Conv2d"``, ``"Linear"``, or ``"Sum"``).
    """
    joined_labels = " ".join(labels).lower()
    has_2d_spatial = len(labels) >= 2 or "conv" in joined_labels or ("ky" in joined_labels and "kx" in joined_labels)
    if has_2d_spatial:
      return "Conv2d"

    opcodes_upper = [inst.opcode.upper() for inst in insts]
    fma_tokens = ("FFMA", "FMA", "IMAD", "HMMA", "WMMA", "FMAC", "MAC", "DOT")
    if any(any(tok in op for tok in fma_tokens) for op in opcodes_upper) or any(
      kw in joined_labels for kw in ("gemm", "linear", "matmul", "fc", "dot")
    ):
      return "Linear"
    return "Sum"

  def _lift_raw_loop_span(
    self,
    graph: LogicalGraph,
    nodes: typing.Sequence[IsaNode],
    span: _DetectedLoopSpan,
    reg_defs: dict[str, str],
    kind_counters: dict[str, int],
    pending_trivia: list[str],
  ) -> str:
    """Lift a natural backward-branch loop span into a high-level ``LogicalNode`` and wire SSA edges.

    Args:
        graph: Target ``LogicalGraph`` under construction.
        nodes: Full sequence of ISA CST nodes.
        span: Detected ``_DetectedLoopSpan`` bounds and labels.
        reg_defs: Active SSA register definition table (mutated with loop accumulator output).
        kind_counters: Per-idiom monotonic counter map.
        pending_trivia: Mutable buffer of leading trivia comments.

    Returns:
        Lifted loop node identifier string.
    """
    loop_insts: list[InstructionT] = [n for n in nodes[span.start_idx : span.end_idx + 1] if self._is_instruction(n)]
    kind = self._classify_loop_idiom(span.labels, loop_insts)
    k_idx = kind_counters.get(kind, 0)
    kind_counters[kind] = k_idx + 1
    loop_id = f"{kind.lower()}_{k_idx}"

    meta: dict[str, str | int | float | list[str]] = dict(self.analyzer_cls.analyze_block(kind, loop_insts))
    meta["loop_labels"] = list(span.labels)

    defined_in_loop: set[str] = set()
    live_in_regs: list[str] = []
    acc_reg: str | None = None
    for l_inst in loop_insts:
      l_dst, l_srcs = extract_instruction_reg_io(l_inst)
      for s_reg in l_srcs:
        if s_reg not in defined_in_loop and s_reg not in live_in_regs:
          live_in_regs.append(s_reg)
      if l_dst is not None and not l_inst.opcode.upper().startswith(("ISETP", "FSETP")):
        defined_in_loop.add(l_dst)
        acc_reg = l_dst

    if live_in_regs:
      meta["source_regs"] = live_in_regs
    if acc_reg is not None:
      meta["dest_reg"] = acc_reg
    if pending_trivia:
      meta["leading_trivia"] = "\x0a".join(pending_trivia)
      pending_trivia.clear()

    graph.nodes[loop_id] = LogicalNode(id=loop_id, op_type=kind, attributes=meta)
    port_idx = 0
    for s_reg in live_in_regs:
      if s_reg in reg_defs:
        graph.add_edge(reg_defs[s_reg], loop_id, target_idx=port_idx, value_name=s_reg)
        port_idx += 1
    if acc_reg is not None:
      reg_defs[acc_reg] = loop_id
    return loop_id

  def _lift_raw_instruction(
    self,
    graph: LogicalGraph,
    node: InstructionT,
    instruction_counter: int,
    reg_defs: dict[str, str],
    pending_trivia: list[str],
    last_node_id: str | None,
  ) -> str:
    """Lift a standalone unannotated instruction node and wire register use-def edges.

    Args:
        graph: Target ``LogicalGraph`` under construction.
        node: Standalone instruction CST node.
        instruction_counter: Monotonic instruction counter.
        reg_defs: Active SSA register definition table (mutated if ``node`` writes a register).
        pending_trivia: Mutable buffer of leading trivia comments.
        last_node_id: Preceding node ID for zero-register control/barrier instructions.

    Returns:
        Inserted ``LogicalNode`` identifier string.
    """
    base_id, kind, inst_meta = self._resolve_orphan(node, instruction_counter)
    node_id = base_id
    suffix = 1
    while node_id in graph.nodes:
      node_id = f"{base_id}_{suffix}"
      suffix += 1

    attrs: dict[str, str | int | float | list[str]] = dict(inst_meta)
    if pending_trivia:
      attrs["leading_trivia"] = "\x0a".join(pending_trivia)
      pending_trivia.clear()

    dst_reg, src_regs = extract_instruction_reg_io(node)
    graph.nodes[node_id] = LogicalNode(id=node_id, op_type=kind, attributes=attrs)
    if dst_reg is None and not src_regs and not _is_branch_opcode(node.opcode) and last_node_id is not None:
      graph.add_edge(last_node_id, node_id, target_idx=0)
    else:
      for port_idx, s_reg in enumerate(src_regs):
        if s_reg in reg_defs:
          graph.add_edge(reg_defs[s_reg], node_id, target_idx=port_idx, value_name=s_reg)
    if dst_reg is not None:
      reg_defs[dst_reg] = node_id
    return node_id

  def lift(self, nodes: typing.Sequence[IsaNode]) -> LogicalGraph:
    """Lift raw unannotated ISA CST nodes into a register-dataflow ``LogicalGraph``.

    Args:
        nodes: Sequence of ISA comments, labels, directives, and instructions.

    Returns:
        LogicalGraph: Reconstructed dataflow graph with SSA edges and lifted loop idioms.
    """
    graph = LogicalGraph(name=self.default_graph_name)
    reg_defs: dict[str, str] = {}
    loop_spans = self._detect_loop_spans(nodes)
    pending_trivia: list[str] = []
    instruction_counter = 0
    kind_counters: dict[str, int] = {}
    last_node_id: str | None = None

    idx = 0
    total = len(nodes)
    while idx < total:
      if idx in loop_spans:
        span = loop_spans[idx]
        last_node_id = self._lift_raw_loop_span(graph, nodes, span, reg_defs, kind_counters, pending_trivia)
        idx = span.end_idx + 1
        continue

      node = nodes[idx]
      if isinstance(node, self.comment_type):
        text = node.text.lstrip(";/ ").strip()
        if _is_trivia_comment(text):
          pending_trivia.append(text)
      elif self._is_instruction(node):
        last_node_id = self._lift_raw_instruction(
          graph, node, instruction_counter, reg_defs, pending_trivia, last_node_id
        )
        instruction_counter += 1

      idx += 1

    if pending_trivia and graph.nodes:
      list(graph.nodes.values())[-1].attributes["leading_trivia"] = "\x0a".join(pending_trivia)
      pending_trivia.clear()

    return graph


class IsaLifter(typing.Generic[NodeT, InstructionT, CommentT]):
  """Unified ISA lifter facade dispatching between ``AnnotatedIsaRoundtripLifter`` and ``RawIsaDataflowLifter``.

  Attributes:
      DEFAULT_GRAPH_NAME: Default graph name (`str`).
      COMMENT_TYPE: Dialect comment CST type (`type[IsaComment]`).
      INSTRUCTION_TYPE: Dialect instruction CST type (`type[IsaInstruction]`).
      ANALYZER_CLS: Dialect analyzer class (`type[IsaAnalyzer]`).
      ORPHAN_PREFIX: Dialect opcode prefix for standalone instructions (`str`).
      mode: Lifting mode (``"auto"``, ``"annotated"``, or ``"raw"``).
      comment_parser: Semantic comment parser instance (`SemanticCommentParser`).
  """

  DEFAULT_GRAPH_NAME: str = "DecompiledModel"
  COMMENT_TYPE: type[IsaComment] = IsaComment
  INSTRUCTION_TYPE: type[IsaInstruction] = IsaInstruction
  ANALYZER_CLS: type[IsaAnalyzer] = IsaAnalyzer
  ORPHAN_PREFIX: str = "asm"

  def __init__(self, mode: IsaLiftMode = "auto") -> None:
    """Initialize ``IsaLifter`` in ``auto``, ``annotated``, or ``raw`` lifting mode.

    Args:
        mode: Lifting mode selector (defaults to ``"auto"``).
    """
    self.mode: IsaLiftMode = mode
    self.comment_parser: SemanticCommentParser = SemanticCommentParser()

  @staticmethod
  def _extract_on_device_kv_pairs(text: str) -> dict[str, str | int | float]:
    """Extract ``key=value`` pairs from ``[on-device shape]`` or ``[on-device stride]`` comments.

    Args:
        text: Stripped comment text.

    Returns:
        dict[str, str | int | float]: Parsed integer or string attribute pairs.
    """
    return extract_on_device_kv_pairs(text)

  def _resolve_orphan_instruction(
    self,
    node: InstructionT,
    instruction_counter: int,
  ) -> tuple[str, str, dict[str, str | int | float]]:
    """Resolve an unenclosed instruction into ``(node_id, op_kind, metadata)``.

    Args:
        node: Raw ISA instruction node.
        instruction_counter: Monotonic counter for unnamed instructions.

    Returns:
        tuple[str, str, dict[str, str | int | float]]: ``(node_id, op_kind, metadata_dict)``.
    """
    opcode = node.opcode or "unknown"
    return (f"inst_{instruction_counter}", f"{self.ORPHAN_PREFIX}.{opcode}", {})

  def _has_semantic_annotations(self, nodes: typing.Sequence[IsaNode]) -> bool:
    """Check whether ``nodes`` contain any structured semantic provenance comments.

    Args:
        nodes: Sequence of ISA CST nodes.

    Returns:
        bool: ``True`` if at least one comment parses as or begins with a semantic marker.
    """
    for node in nodes:
      if isinstance(node, self.COMMENT_TYPE):
        text = node.text.lstrip(";/ ").strip()
        if self.comment_parser.parse(text) is not None or text.startswith(("BEGIN ", "END ")):
          return True
    return False

  def build_annotated_lifter(self) -> AnnotatedIsaRoundtripLifter[NodeT, InstructionT, CommentT]:
    """Construct an ``AnnotatedIsaRoundtripLifter`` configured for this dialect.

    Returns:
        AnnotatedIsaRoundtripLifter[NodeT, InstructionT, CommentT]: Configured roundtrip lifter.
    """
    return AnnotatedIsaRoundtripLifter(
      default_graph_name=self.DEFAULT_GRAPH_NAME,
      comment_type=self.COMMENT_TYPE,
      instruction_type=self.INSTRUCTION_TYPE,
      analyzer_cls=self.ANALYZER_CLS,
      orphan_resolver=self._resolve_orphan_instruction,
      comment_parser=self.comment_parser,
    )

  def build_raw_lifter(self) -> RawIsaDataflowLifter[NodeT, InstructionT, CommentT]:
    """Construct a ``RawIsaDataflowLifter`` configured for this dialect.

    Returns:
        RawIsaDataflowLifter[NodeT, InstructionT, CommentT]: Configured raw dataflow lifter.
    """
    return RawIsaDataflowLifter(
      default_graph_name=self.DEFAULT_GRAPH_NAME,
      comment_type=self.COMMENT_TYPE,
      instruction_type=self.INSTRUCTION_TYPE,
      analyzer_cls=self.ANALYZER_CLS,
      orphan_resolver=self._resolve_orphan_instruction,
    )

  def lift(self, nodes: typing.Sequence[IsaNode]) -> LogicalGraph:
    """Lift ``nodes`` into a ``LogicalGraph`` using the configured or auto-detected mode.

    Args:
        nodes: Sequence of low-level ISA CST nodes.

    Returns:
        LogicalGraph: Reconstructed high-level computational graph.
    """
    if self.mode == "annotated" or (self.mode == "auto" and self._has_semantic_annotations(nodes)):
      return self.build_annotated_lifter().lift(nodes)
    return self.build_raw_lifter().lift(nodes)


__all__ = [
  "AnnotatedIsaRoundtripLifter",
  "IsaLiftMode",
  "IsaLifter",
  "RawIsaDataflowLifter",
  "extract_instruction_reg_io",
  "extract_on_device_kv_pairs",
]

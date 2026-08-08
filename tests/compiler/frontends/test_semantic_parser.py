"""Unit tests for the semantic comment parser within the compiler frontends.

This module validates the functionality of the `SemanticCommentParser` class,
which is responsible for analyzing inline code comments containing metadata
about model inputs, layer/block structure, return statements, and unmapped
framework operations. This parsing allows the compiler to rebuild or match
logical sections during model translation or execution tracing.
"""

import pytest

from ml_switcheroo.core.compiler.frontends.semantic_parser import (
  SemanticCommentParser,
  SemanticInput,
  SemanticBegin,
  SemanticEnd,
  SemanticUnmapped,
  SemanticReturn,
  SemanticMarker,
  Trivia,
)


def test_semantic_parser_input():
  """Verifies that input annotations are correctly parsed into SemanticInput markers.

  This test checks that comments matching the 'Input <name> ->' pattern are parsed,
  extracting the variable name and preserving whitespace variations.

  Args:
      None

  Returns:
      None
  """
  parser = SemanticCommentParser()
  marker = parser.parse("Input x ->")
  assert isinstance(marker, SemanticInput)
  assert marker.name == "x"
  assert marker.to_text() == "Input x ->"

  marker2 = parser.parse("  Input my_var -> ")
  assert isinstance(marker2, SemanticInput)
  assert marker2.name == "my_var"
  assert marker2.to_text() == "  Input my_var -> "


def test_semantic_parser_begin():
  """Verifies that 'BEGIN' annotations are correctly parsed into SemanticBegin markers.

  This test validates that block start annotations (e.g. 'BEGIN Conv2d(block1)')
  are parsed into SemanticBegin objects with their designated operation type
  and block identifier properly extracted.

  Args:
      None

  Returns:
      None
  """
  parser = SemanticCommentParser()
  marker = parser.parse("BEGIN Conv2d(block1)")
  assert isinstance(marker, SemanticBegin)
  assert marker.kind == "Conv2d"
  assert marker.id == "block1"
  assert marker.to_text() == "BEGIN Conv2d(block1)"

  marker2 = parser.parse("BEGIN Conv2d ( block1 ) ")
  assert isinstance(marker2, SemanticBegin)
  assert marker2.to_text() == "BEGIN Conv2d ( block1 ) "


def test_semantic_parser_end():
  """Verifies that 'END' annotations are correctly parsed into SemanticEnd markers.

  This test checks that block termination annotations (e.g. 'END Conv2d(block1)')
  are correctly processed, preserving the operation type and block identifier.

  Args:
      None

  Returns:
      None
  """
  parser = SemanticCommentParser()
  marker = parser.parse("END Conv2d(block1)")
  assert isinstance(marker, SemanticEnd)
  assert marker.kind == "Conv2d"
  assert marker.id == "block1"
  assert marker.to_text() == "END Conv2d(block1)"

  marker2 = parser.parse("END Conv2d ( block1 ) ")
  assert isinstance(marker2, SemanticEnd)
  assert marker2.to_text() == "END Conv2d ( block1 ) "


def test_semantic_parser_unmapped():
  """Verifies that unmapped operations are correctly parsed into SemanticUnmapped markers.

  This test checks that custom/unmapped library annotations (e.g. 'Unmapped Op: Linear(node1)'
  or fully qualified Python function paths) are parsed, extracting both the API name
  and the node identifier.

  Args:
      None

  Returns:
      None
  """
  parser = SemanticCommentParser()
  marker = parser.parse("Unmapped Op: Linear(node1)")
  assert isinstance(marker, SemanticUnmapped)
  assert marker.api == "Linear"
  assert marker.id == "node1"
  assert marker.to_text() == "Unmapped Op: Linear(node1)"

  marker2 = parser.parse(" Unmapped Op: torch.nn.functional.relu(node2) ")
  assert isinstance(marker2, SemanticUnmapped)
  assert marker2.api == "torch.nn.functional.relu"
  assert marker2.id == "node2"
  assert marker2.to_text() == " Unmapped Op: torch.nn.functional.relu(node2) "

  marker3 = parser.parse("Unmapped Op: Linear ( node1 ) ")
  assert isinstance(marker3, SemanticUnmapped)
  assert marker3.to_text() == "Unmapped Op: Linear ( node1 ) "


def test_semantic_parser_return():
  """Verifies that return annotations are correctly parsed into SemanticReturn markers.

  This test validates that comments marking function outputs (e.g. 'Return:' or
  'Return: output_var') are recognized and parsed as SemanticReturn instances.

  Args:
      None

  Returns:
      None
  """
  parser = SemanticCommentParser()
  marker = parser.parse("Return:")
  assert isinstance(marker, SemanticReturn)
  assert marker.to_text() == "Return:"

  marker2 = parser.parse(" Return: output_var ")
  assert isinstance(marker2, SemanticReturn)
  assert marker2.to_text() == " Return: output_var "


def test_semantic_parser_trivia():
  """Verifies the behavior of the Trivia semantic marker class.

  This test checks that Trivia markers represent arbitrary text comment blocks
  and return their content faithfully when converted back to a string representation.

  Args:
      None

  Returns:
      None
  """
  trivia = Trivia("test")
  assert trivia.to_text() == "test"


def test_semantic_parser_invalid():
  """Verifies that invalid or malformed comments return None during parsing.

  This test ensures that comments not matching any recognized semantic syntax,
  such as raw plain-text comments or poorly formatted directives (e.g., missing
  closing parenthesis), are safely rejected by the parser.

  Args:
      None

  Returns:
      None
  """
  parser = SemanticCommentParser()
  assert parser.parse("Invalid comment") is None
  assert parser.parse("BEGIN Conv2d block1)") is None  # missing lparen


def test_semantic_marker_base():
  """Verifies the base class NotImplementedError for to_text."""
  marker = SemanticMarker()
  with pytest.raises(NotImplementedError):
    marker.to_text()

"""
Static Analysis Package.

This package contains visitors and utilities for inspecting Python ASTs (Abstract Syntax Trees)
to extract metadata, ensuring safety, and building symbol tables prior to transformation.

Modules:
    - ``audit``: Coverage analysis to find missing API mappings.
    - ``dependencies``: Scanning for external package usage.
    - ``lifecycle``: Verifying class member initialization correctness.
    - ``purity``: Detecting side effects (I/O, Globals) violating functional purity.
    - ``symbol_table``: Inferring variable types and scopes.
"""

# TODO Plan: README Audit Fixes & Quality Metrics

> **STRICT CONSTRAINT:** Do not introduce ANY new dependencies in `pyproject.toml`, `requirements.txt`, or any other environment file. All implementations must rely on the standard library or existing dependencies.

## 1. README Documentation Corrections
- [x] **Fix Flax NNX vs Linen Confusion:**
  - [x] Correct the "Functional Unwrapping" section in `README.md`.
  - [x] Accurately describe that **Flax Linen** uses the functional `layer.apply(params, x)` pattern, whereas **Flax NNX** provides an Object-Oriented API (`nnx.Module`).
  - [x] Ensure the description of unwrapping correctly states that it converts functional paradigms (like Linen) into OO paradigms (like Torch or NNX), or vice versa.
- [x] **Address MaxText Claims:**
  - [x] Review the "Auto-Sharding & Distributed Semantics" section in `README.md`.
  - [x] Decide whether to remove the mention of `MaxText` OR implement a complete `maxtext` framework adapter.
  - [x] If removing, adjust the copy to focus solely on `PaxML`.
- [x] **Clarify Safetensors Support:**
  - [x] Update the "Weight Migration (Checkpointing)" section in `README.md`.
  - [x] Ensure the documentation accurately reflects the current state of `safetensors` support (currently only exposed directly via Apple MLX and some plugin traits, but missing from core `convert_weights.py` PyTorch/JAX pipelines).

## 2. Codebase Alignment (Fulfilling README Claims)
- [x] **Flax NNX / Linen Refactoring:**
  - [x] Audit `src/ml_switcheroo/core/rewriter/calls/pre.py` and ensure the `functional_execution_method` unwrapping logic strictly targets functional frameworks (like Flax Linen or JAX standard), not Flax NNX.
  - [x] Verify `FlaxNnxAdapter` (`src/ml_switcheroo/frameworks/flax_nnx.py`) correctly defines structural traits representing its OO nature (`forward_method="__call__"`) without functional trait leakage.
- [x] **Implement MaxText Framework Adapter (If kept in README):**
  - [x] Create `src/ml_switcheroo/frameworks/maxtext.py`.
  - [x] Implement `MaxTextAdapter` inheriting from `JAXStackMixin` or `FrameworkAdapter`.
  - [x] Register the adapter with `@register_framework("maxtext")`.
  - [x] Provide required standard imports, syntax mappings, and hardware abstractions.
  - [x] Wire it into the `SemanticsManager` registry.
- [x] **Expand `WeightScriptGenerator` for Safetensors:**
  - [x] Modify `src/ml_switcheroo/cli/handlers/convert_weights.py` to natively support `safetensors` as a primary weight migration format.
  - [x] Integrate `safetensors` support into PyTorch (`torch_io.py`) and JAX (`jax_stack.py`) mixins.
  - [x] Update the docstrings in `convert_weights.py` to list `safetensors` alongside `PyTorch -> JAX (Flax)`.
  - [x] Ensure layout permutations (NCHW <-> NHWC) correctly apply when writing to/reading from `safetensors`.

## 3. Quality Metrics & Constraints
- [x] **100% Documentation Coverage:**
  - [x] Every modified/new module must have a complete docstring.
  - [x] Every modified/new file must have a docstring.
  - [x] Every modified/new class must have a docstring.
  - [x] Every modified/new function/method must have a docstring.
  - [x] Every argument/parameter and return value must be documented.
- [x] **100% Test Coverage:**
  - [x] Implement tests to cover 100% of functions added or modified.
  - [x] Implement tests to cover 100% of lines added or modified.
  - [x] Implement tests to cover 100% of branches added or modified.
  - [x] Add unit tests for `convert_weights.py` specifically targeting `safetensors` logic.
  - [x] Add unit tests for `MaxTextAdapter` (if implemented).
- [x] **Strong Typing:**
  - [x] 100% Type hint coverage across all modified and new files.
  - [x] Ensure strict compliance (no `Any` used as a crutch, proper `Optional`/`Union` usage).
  - [x] No use of `# type: ignore` to bypass architectural flaws.

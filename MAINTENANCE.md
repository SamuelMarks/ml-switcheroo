Maintenance
===========

This project acts as a bridge between specifications. Maintenance involves ingesting upstream standards and syncing them
with downstream libraries.

## 1. Updating Standards (Ingestion)

We parse official repos to build the "Standard" columns of our knowledge base.

### Array API (Math)

Clone [data-apis/array-api](https://github.com/data-apis/array-api).

```bash
ml_switcheroo import-spec ./array-api/src/array_api_stubs/_2023_12
```

### ONNX (Neural)

Clone [onnx/onnx](https://github.com/onnx/onnx).

```bash
ml_switcheroo import-spec ./onnx/docs/Operators.md
```

---

## 2. Syncing Implementations

Once standards are ingested, link them to the installed specific versions of frameworks.

```bash
# 1. Sync PyTorch
ml_switcheroo sync torch

# 2. Sync JAX
ml_switcheroo sync jax

# 3. Sync Other Backends (if installed)
ml_switcheroo sync tensorflow
ml_switcheroo sync mlx
```

*Note: The `FrameworkSyncer` automatically checks function signatures (arity) to prevent mismatches.*

---

## 3. Verification & Publishing

### Verification Gating

We can "lock" specific APIs if they fail verification to prevent the transpiler from generating broken code using them.

```bash
# 1. Run full verification suite per API
ml_switcheroo ci 

# Output is a report.json. This can be fed back into config:
# validation_report = "verification.json"
```

### Release

To update the README table with the latest verification results:

```bash
ml_switcheroo ci --update-readme
```

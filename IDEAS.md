ml-switcheroo ideas
===================

## Advanced Features

### Dynamic Shape Tracing
Integrate with `torch.fx` or `jax.make_jaxpr` to extract graph information for cases where static AST analysis fails (e.g. data-dependent control flow that resolves to static graph).

### Documentation Generator
Generate "Migration Guides" automatically by diffing the semantics JSONs.
Example output: "In JAX, `torch.foo` is equivalent to `jax.bar`, but argument `dim` becomes `axis`. Warning: `keepdims` default differs."

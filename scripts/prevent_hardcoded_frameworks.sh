#!/usr/bin/env bash

# Greps for hardcoded framework routing logic in python files

FAILED=0
PATTERN="(if|elif|return)[[:space:]]+(self\.|ctx\.|engine\.)?(framework|target|source|target_framework|source_framework|target_fw|source_fw)[[:space:]]+(==|!=|in)[[:space:]]*(\{|\[|\(|str\()?[[:space:]]*['\"](jax|torch|mlx|tensorflow|keras|numpy|flax|flax_nnx|paxml)['\"]"
DICT_PATTERN="^[[:space:]]*['\"](torch|jax|flax_nnx|mlx|paxml|keras)['\"][[:space:]]*:"
REFLECTION_PATTERN="(getattr\([[:space:]]*\{['\"]v['\"]:|getattr\([^,]+,[[:space:]]*['\"](__getitem__|__setitem__|__call__)['\"]\))"

for file in "$@"; do
    if [[ "$file" == *.py ]]; then
        if grep -q -E "$PATTERN" "$file"; then
            echo "Error: Hardcoded framework routing found in $file"
            grep -n -E "$PATTERN" "$file"
            echo "Please use JSON variants or traits for dispatch logic."
            FAILED=1
        fi
        if [[ "$file" == *"ml_switcheroo/core/"* || "$file" == *"ml_switcheroo/plugins/"* ]]; then
            if grep -q -E "$DICT_PATTERN" "$file"; then
                echo "Error: Hardcoded framework dictionary key found in $file"
                grep -n -E "$DICT_PATTERN" "$file"
                echo "Please move framework-specific configuration into StructuralTraits, PluginTraits, or ODL."
                FAILED=1
            fi
        fi
        if grep -q -E "$REFLECTION_PATTERN" "$file"; then
            echo "Error: Reflection type-checker evasion found in $file"
            grep -n -E "$REFLECTION_PATTERN" "$file"
            echo "Please use explicit type narrowing instead of dynamic getattr dunder access."
            FAILED=1
        fi
    fi
done

exit $FAILED

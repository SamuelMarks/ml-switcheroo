# bash completion for ml_switcheroo
_ml_switcheroo_completion() {
    local cur prev opts commands frameworks
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    commands="convert convert-weights define import-onnx gen-weight-script matrix schema suggest scaffold harvest verified-pipeline ci gen-docs gen-tests completion"
    frameworks="torch jax flax_nnx mlx keras tensorflow numpy nvidia_sass rdna mlir stablehlo ir wasm cpp"

    case "$prev" in
        --source|--target|-s|-t)
            COMPREPLY=( $(compgen -W "$frameworks" -- "$cur") )
            return 0
            ;;
        --intermediate)
            COMPREPLY=( $(compgen -W "ir ml_switcheroo_ir mlir tikz" -- "$cur") )
            return 0
            ;;
        completion)
            COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") )
            return 0
            ;;
    esac

    if [[ "$COMP_CWORD" -eq 1 ]] ; then
        COMPREPLY=( $(compgen -W "$commands --help --version" -- "$cur") )
        return 0
    fi
}

complete -F _ml_switcheroo_completion ml_switcheroo
complete -F _ml_switcheroo_completion "🔄🦘"

#compdef ml_switcheroo 🔄🦘

_ml_switcheroo() {
    local -a commands frameworks
    commands=('convert:convert command' 'convert-weights:convert-weights command' 'define:define command' 'import-onnx:import-onnx command' 'gen-weight-script:gen-weight-script command' 'matrix:matrix command' 'schema:schema command' 'suggest:suggest command' 'scaffold:scaffold command' 'harvest:harvest command' 'verified-pipeline:verified-pipeline command' 'ci:ci command' 'gen-docs:gen-docs command' 'gen-tests:gen-tests command' 'completion:completion command')
    frameworks=('torch:torch framework' 'jax:jax framework' 'flax_nnx:flax_nnx framework' 'mlx:mlx framework' 'keras:keras framework' 'tensorflow:tensorflow framework' 'numpy:numpy framework' 'nvidia_sass:nvidia_sass framework' 'rdna:rdna framework' 'mlir:mlir framework' 'stablehlo:stablehlo framework' 'ir:ir framework' 'ml_switcheroo_ir:ml_switcheroo_ir framework')

    _arguments -C
        '1: :->command'
        '*: :->args'

    case $state in
        command)
            _describe -t commands 'ml_switcheroo command' commands
            ;;
        args)
            case $words[2] in
                convert)
                    _arguments
                        '--source[Source framework]:framework:($frameworks)'
                        '--target[Target framework]:framework:($frameworks)'
                        '--intermediate[Intermediate dialect]:dialect:(ir ml_switcheroo_ir mlir tikz)'
                    ;;
                completion)
                    _values 'shell' bash zsh fish
                    ;;
            esac
            ;;
    esac
}

_ml_switcheroo "$@"

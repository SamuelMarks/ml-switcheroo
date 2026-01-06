# StableHLO Specification

### `abs`

Computes the absolute value. 
Returns a tensor of the same type. 

#### Syntax

```mlir
%result = stablehlo.abs %operand : tensor<4xf32> 
```

--- 

### `add`

Performs element-wise addition. 
This is a very long description that should continually go on and on to verify
that the truncation logic works correctly because nobody wants a single line
description in the generated json file that spans five hundred characters and
breaks the terminal wrapping logic when printing summaries to the console log output. 

#### Syntax

```mlir
%result = stablehlo.add %lhs, %rhs : tensor<2xi32> 
```

### `log_plus_one`

Computes log(x + 1). 

#### Inputs

No syntax block provided here. 

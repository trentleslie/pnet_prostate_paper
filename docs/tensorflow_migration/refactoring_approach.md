# Refactoring Approach for TensorFlow 2.x Migration

This document outlines the systematic approach for refactoring the P-NET codebase from TensorFlow 1.x to TensorFlow 2.x.

## Refactoring Methodology

The refactoring process follows these key principles:

1. **Maintain Functional Equivalence**: Ensure that refactored code produces numerically equivalent results to the original implementation.

2. **Incremental Approach**: Refactor one component at a time, with careful testing at each step.

3. **Test-Driven Refactoring**: Create tests that validate the behavior of both original and refactored code.

4. **Documentation**: Document changes, patterns, and potential issues to guide future refactoring efforts.

## Refactoring Workflow

For each component being refactored, follow these steps:

1. **Analysis**: Identify TF1.x patterns and required changes
2. **Plan**: Create a detailed refactoring plan
3. **Implementation**: Apply the changes
4. **Testing**: Verify that the refactored code works as expected
5. **Integration**: Update dependent code

## Model Definition Functions

The model definition functions (`build_pnet`, `build_pnet2`, etc.) are being refactored in the following steps:

1. **Import Updates**: Replace keras imports with tensorflow.keras
2. **Parameter Renaming**: Update deprecated parameter names
3. **Model Creation Syntax**: Update Model creation calls
4. **Layer Operations**: Replace merge operations with layer-based operations
5. **Python 3 Compatibility**: Update print statements and other Python 2 syntax

## Custom Layers

Custom layers (`Diagonal`, `SparseTF`, etc.) are being refactored to:

1. **Update Base Class**: Use tensorflow.keras.layers.Layer
2. **Update Constructor**: Rename parameters following TF2.x conventions
3. **Update build Method**: Change weight creation approach
4. **Update call Method**: Ensure compatibility with eager execution
5. **Add get_config**: Ensure proper serialization

## Gradient Calculation

Gradient calculation functions are being refactored to:

1. **Replace K.function**: Use tf.GradientTape for gradient computation
2. **Update Backend Operations**: Replace K.* operations with tf.* equivalents
3. **Remove Session Handling**: Ensure compatibility with eager execution
4. **Ensure Numerical Stability**: Maintain normalization and other stabilizing factors

## Testing Strategy

Testing refactored code involves:

1. **Unit Tests**: Test individual functions in isolation
2. **Integration Tests**: Test interactions between components
3. **Comparison Tests**: Compare outputs of original and refactored implementations
4. **Edge Cases**: Test with various input shapes, types, and values
5. **Performance Tests**: Validate performance characteristics

## Implementation Status

Current implementation status:

- ✅ Gradient Calculation Functions (`get_gradient_layer`, `get_gradient`)
- 🔄 Model Definition Functions (`build_pnet2`, `get_pnet`)
- 🔄 Custom Layers (`Diagonal`, `SparseTF`)
- ⏱️ Tests
- ⏱️ Integration

## Best Practices

When refactoring, follow these best practices:

1. **Use TF2.x Idioms**: Embrace eager execution and TF2's functional style
2. **Leverage @tf.function**: Use for performance-critical operations
3. **Maintain API Compatibility**: Keep function signatures consistent
4. **Document Changes**: Add comments explaining significant changes
5. **Consider Backward Compatibility**: Add fallbacks if needed for loading old models
# Component Refactoring Plans

This document consolidates the refactoring plans for key components of the P-NET codebase for TensorFlow 2.x compatibility.

## Table of Contents

1. [Model Building Functions](#model-building-functions)
2. [Network Construction Functions](#network-construction-functions)
3. [Custom Layers](#custom-layers)
4. [Testing Strategy](#testing-strategy)

## Model Building Functions

The `build_pnet2` function in `prostate_models.py` constructs a Pathway Network (P-NET) model using Keras. This function needs to be refactored for TensorFlow 2.x compatibility.

### TF1.x Patterns Identified

1. **Import Statements**:
   - Using `keras` direct imports instead of `tensorflow.keras`
   - Missing `tensorflow` import

2. **Print Statements**:
   - Python 2.x style print statements without parentheses

3. **Model Creation and Layer Definitions**:
   - `Model(input=[ins], output=outcome)` syntax (square brackets for input)
   - Older regularizer syntax: `W_regularizer=l2(w_reg)`

4. **Layer Configuration**:
   - Older parameter names: `W_regularizer` instead of `kernel_regularizer`

5. **Model Compilation**:
   - Old-style metrics definition with custom functions

### Step-by-Step Refactoring Plan

1. **Update Imports**:
   - Replace all `keras.*` imports with `tensorflow.keras.*`
   - Add `import tensorflow as tf`

2. **Fix Print Statements**:
   - Update all print statements to use Python 3 syntax with parentheses
   - Format print statements to use f-strings where appropriate

3. **Update Layer Definitions and Parameters**:
   - Replace `W_regularizer` → `kernel_regularizer`
   - Update any other deprecated parameter names

4. **Update Model Creation**:
   - `Model(input=[ins], output=outcome)` → `Model(inputs=ins, outputs=outcome)`

5. **Update Model Compilation**:
   - Ensure custom metrics are TF2.x compatible
   - Update loss specification for multi-output models if needed

## Network Construction Functions

The `get_pnet` function in `builders_utils.py` creates the core network structure used by model building functions. It needs to be refactored for TensorFlow 2.x compatibility.

### TF1.x Patterns Identified

1. **Layer Creation**:
   - `W_regularizer=l2(w_reg)` syntax
   - Other parameter naming issues

2. **Merge Operations**:
   - `multiply([outcome, attention_probs], name='attention_mul')` 
   - Direct usage of `merge` functions

3. **Python 2.x Style**:
   - Print statements without parentheses
   - Old-style string formatting

### Step-by-Step Refactoring Plan

1. **Update Layer Creation**:
   - Replace `W_regularizer=l2(w_reg)` with `kernel_regularizer=l2(w_reg)`
   - Update other parameter names

2. **Replace Merge Operations**:
   - Replace `multiply([a, b], name='c')` with `Multiply(name='c')([a, b])`
   - Update other merge operations

3. **Fix Python Style**:
   - Update print statements to use parentheses
   - Use f-strings for string formatting

4. **Update Type Checking**:
   - Replace `type(var) == list` with `isinstance(var, list)`
   - Update other type checking patterns

## Custom Layers

Custom layers like `Diagonal` and `SparseTF` need to be updated to be compatible with TensorFlow 2.x.

### TF1.x Patterns to Update

1. **Base Class and Imports**:
   - `from keras import backend as K`
   - `from keras.layers import Layer`

2. **Constructor Parameters**:
   - `W_regularizer`, `init`, etc.

3. **Weight Creation**:
   - Parameter naming in `add_weight`
   - Weight creation approach

4. **Call Method**:
   - `call(self, x, mask=None)` 
   - Backend operations

### Step-by-Step Refactoring Plan

1. **Update Base Class and Imports**:
   - `from tensorflow.keras import backend as K`
   - `from tensorflow.keras.layers import Layer`

2. **Update Constructor Parameters**:
   - `W_regularizer` → `kernel_regularizer`
   - `init` → `kernel_initializer`

3. **Update Weight Creation**:
   - Update parameter names
   - Use TF2.x-style weight initialization

4. **Update Call Method**:
   - `call(self, inputs, training=None)`
   - Replace backend operations with TF operations
   - Handle the `training` parameter

5. **Add Config Method**:
   - Implement proper `get_config` for serialization
   - Ensure proper parameter handling

## Testing Strategy

A comprehensive testing strategy is essential to ensure the refactored code works correctly.

### Testing Approach

1. **Create Test Models**:
   - Simple models with well-defined behavior
   - Models using each custom layer

2. **Test Forward Pass**:
   - Verify output shapes
   - Verify output values for known inputs

3. **Test Gradients**:
   - Compare gradient values with original implementation
   - Test with different input shapes and sizes

4. **Test Edge Cases**:
   - Zero inputs
   - Single sample
   - Very large inputs
   - Multi-output models

### Implementation Plan

1. **Create Test Script**:
   - Define test models and data
   - Implement comparison functions

2. **Test Metrics**:
   - Correlation between original and refactored outputs
   - Absolute differences
   - Relative differences

3. **Visualization**:
   - Plot gradient distributions
   - Compare original vs. refactored values

4. **Integration Testing**:
   - Test the refactored components together
   - Verify end-to-end behavior
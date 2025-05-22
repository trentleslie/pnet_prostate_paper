# TensorFlow 2.x Custom Layers

This directory contains the refactored TensorFlow 2.x compatible versions of custom layers used in the P-NET architecture, along with their unit tests.

## Overview

The original implementation of P-NET used TensorFlow 1.x, which had a different API and execution model compared to TensorFlow 2.x. This refactoring effort updates the custom layers to be compatible with TensorFlow 2.x while maintaining the same functionality.

The refactored layers are:
- `Diagonal`: A layer where each unit is connected to a specific subset of input features.
- `SparseTF`: A layer with sparse connectivity defined by a connection map or indices.

## Implementation Details

### Diagonal Layer

The Diagonal layer creates a block-diagonal connectivity pattern, where each output unit is connected to a specific subset of input features. This is achieved by:
1. Dividing the input features evenly among output units
2. Using element-wise multiplication, reshaping, and summing operations

### SparseTF Layer

The SparseTF layer implements arbitrary sparse connectivity patterns, where connections between inputs and outputs are defined by either:
1. A binary connection map (2D numpy array)
2. A list of non-zero indices (pairs of input and output indices)

This is achieved using TensorFlow's scatter_nd operation to create a sparse weight matrix.

## Usage

### Importing

```python
# Import the TF2.x layers directly
from model.layers_custom_tf2 import Diagonal, SparseTF

# Or use the compatibility aliases from layers_custom
from model.layers_custom import DiagonalTF2, SparseTFTF2
```

### Example Usage

```python
import tensorflow as tf
from model.layers_custom_tf2 import Diagonal, SparseTF

# Create a simple model with a Diagonal layer
inputs = tf.keras.layers.Input(shape=(12,))
x = Diagonal(units=4, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Create a model with a SparseTF layer
# Define connectivity pattern
import numpy as np
input_dim = 8
output_dim = 4
conn_map = np.zeros((input_dim, output_dim))
for i in range(input_dim):
    j = i % output_dim
    conn_map[i, j] = 1

# Create model
inputs = tf.keras.layers.Input(shape=(input_dim,))
x = SparseTF(units=output_dim, map=conn_map, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

## Running the Tests

To run the unit tests for the refactored layers:

```bash
cd /procedure/pnet_prostate_paper
python -m model.testing.test_layers_custom_tf2
```

The tests verify:
- Correct initialization and building of layers
- Forward pass calculations
- Activation function application
- Bias functionality
- Serialization and deserialization
- Integration with Keras models

## Backward Compatibility

The original layers (`Diagonal` and `SparseTF` in `layers_custom.py`) have been kept for backward compatibility, but they emit deprecation warnings. New code should use the TensorFlow 2.x compatible versions.

## Future Improvements

Possible future improvements include:
- Performance optimizations using TensorFlow's XLA compiler
- Additional unit tests for edge cases
- Integration with other TensorFlow 2.x features like distributed training
# TensorFlow 2.x Migration Patterns

This document outlines the key patterns and transformations required when migrating code from TensorFlow 1.x to TensorFlow 2.x. These patterns are being applied throughout the P-NET codebase.

## Import Changes

### TF1.x Pattern
```python
from keras import Input
from keras.engine import Model
from keras.layers import Dense, Dropout, Lambda, Concatenate
from keras.regularizers import l2
import keras.backend as K
```

### TF2.x Pattern
```python
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Lambda, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K  # Still needed for some operations
```

## Parameter Name Changes

### TF1.x Pattern
```python
layer = Dense(units=64, W_regularizer=l2(0.01), init='glorot_uniform')
```

### TF2.x Pattern
```python
layer = Dense(units=64, kernel_regularizer=l2(0.01), kernel_initializer='glorot_uniform')
```

## Model Creation

### TF1.x Pattern
```python
model = Model(input=[inputs], output=outputs)
```

### TF2.x Pattern
```python
model = Model(inputs=inputs, outputs=outputs)
```

## Backend Operations

### TF1.x Pattern
```python
dot = lambda a, b: K.batch_dot(a, b, axes=1)
l2_norm = lambda a, b: K.sqrt(K.sum(K.square(a - b), axis=1, keepdims=True))
```

### TF2.x Pattern
```python
dot = lambda a, b: tf.matmul(a, b, transpose_b=True)
l2_norm = lambda a, b: tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True))
```

## Gradient Calculation

### TF1.x Pattern
```python
grad = model.optimizer.get_gradients(model.total_loss, layer)
gradients = layer * grad
get_gradients = K.function(inputs=input_tensors, outputs=[gradients])
gradients = get_gradients(inputs)[0]
```

### TF2.x Pattern
```python
with tf.GradientTape() as tape:
    outputs = model(x, training=False)
    loss = model.loss(y, outputs)
    
grads = tape.gradient(loss, layer)
gradients = layer * grads
```

## Layer Merging Operations

### TF1.x Pattern
```python
merged = merge([layer1, layer2], mode='concat', concat_axis=1)
multiplied = merge([layer1, layer2], mode='mul')
```

### TF2.x Pattern
```python
merged = Concatenate(axis=1)([layer1, layer2])
multiplied = Multiply()([layer1, layer2])
```

## Custom Layer Implementation

### TF1.x Pattern
```python
class CustomLayer(Layer):
    def __init__(self, output_dim, W_regularizer=None):
        self.output_dim = output_dim
        self.W_regularizer = W_regularizer
        super(CustomLayer, self).__init__()
        
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
                                 initializer='glorot_uniform',
                                 regularizer=self.W_regularizer,
                                 name='W')
        self.built = True
        
    def call(self, x, mask=None):
        return K.dot(x, self.W)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

### TF2.x Pattern
```python
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel_regularizer=None):
        super(CustomLayer, self).__init__()
        self.output_dim = output_dim
        self.kernel_regularizer = kernel_regularizer
        
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                     initializer='glorot_uniform',
                                     regularizer=self.kernel_regularizer,
                                     name='kernel')
        super(CustomLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        return tf.matmul(inputs, self.kernel)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
        
    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        }
        base_config = super(CustomLayer, self).get_config()
        return {**base_config, **config}
```

## Layer Output Extraction

### TF1.x Pattern
```python
inp = model.input
out = layer.output
func = K.function([inp, K.learning_phase()], [out])
layer_output = func([X, 0])[0]  # 0 = test phase
```

### TF2.x Pattern
```python
# Create a temporary model
temp_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
# Get layer output with eager execution
layer_output = temp_model(X, training=False).numpy()
```

## Session Handling

### TF1.x Pattern
```python
with tf.Session() as sess:
    K.set_session(sess)
    # Perform operations
    result = sess.run(tensor, feed_dict={...})
```

### TF2.x Pattern
```python
# Direct tensor operations in eager mode
result = tensor.numpy()  # For tf.Tensor objects
# or
result = model(inputs, training=False)  # For model calls
```

These patterns form the foundation for refactoring the P-NET codebase to be compatible with TensorFlow 2.x while maintaining functional equivalence.
"""
Test script for validating the TF2.x implementation of custom layers.
This script provides unit tests for the Diagonal and SparseTF layers to ensure
they behave correctly with TensorFlow 2.x.
"""

import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import the refactored layers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers_custom_tf2 import Diagonal, SparseTF, SparseTFConstraint


class TestDiagonalLayer(unittest.TestCase):
    """Test cases for the Diagonal layer."""
    
    def setUp(self):
        """Setup for each test case."""
        # Create a simple data set for testing
        np.random.seed(42)
        self.batch_size = 10
        self.input_dim = 12
        self.units = 4
        self.X = np.random.random((self.batch_size, self.input_dim))
        
        # Initialize weights for deterministic testing
        self.initial_kernel = np.ones(self.input_dim)
        self.initial_bias = np.zeros(self.units)
        
        # Define the n_inputs_per_node (how many inputs connect to each output unit)
        self.n_inputs_per_node = self.input_dim // self.units
    
    def test_layer_initialization(self):
        """Test that the layer initializes correctly."""
        layer = Diagonal(units=self.units, activation='relu')
        self.assertEqual(layer.units, self.units)
        self.assertEqual(layer.activation.__name__, 'relu')
        self.assertTrue(layer.use_bias)
    
    def test_build(self):
        """Test that the layer builds correctly."""
        layer = Diagonal(units=self.units)
        layer.build(input_shape=(None, self.input_dim))
        
        # Check that kernel and bias were created
        self.assertIsNotNone(layer.kernel)
        self.assertIsNotNone(layer.bias)
        
        # Check shapes
        self.assertEqual(layer.kernel.shape, (self.input_dim,))
        self.assertEqual(layer.bias.shape, (self.units,))
        
        # Check non-zero indices calculation
        self.assertEqual(layer.n_inputs_per_node, self.n_inputs_per_node)
        self.assertEqual(layer.nonzero_ind.shape, (self.input_dim, 2))
    
    def test_call(self):
        """Test the forward pass of the layer."""
        # Create the layer with known weights for deterministic testing
        layer = Diagonal(units=self.units, activation=None, use_bias=True)
        layer.build(input_shape=(None, self.input_dim))
        
        # Set weights to known values
        layer.set_weights([self.initial_kernel, self.initial_bias])
        
        # Compute output
        output = layer(self.X)
        
        # Compute expected output manually
        expected_output = np.zeros((self.batch_size, self.units))
        for i in range(self.batch_size):
            for j in range(self.units):
                # Compute the weighted sum for each unit
                start = j * self.n_inputs_per_node
                end = (j + 1) * self.n_inputs_per_node
                
                # Each unit gets inputs from a specific range
                unit_inputs = self.X[i, start:end]
                
                # Weights for this range
                weights = self.initial_kernel[start:end]
                
                # Weighted sum
                expected_output[i, j] = np.sum(unit_inputs * weights) + self.initial_bias[j]
        
        # Check that the outputs match
        self.assertTrue(np.allclose(output.numpy(), expected_output))
    
    def test_activation(self):
        """Test that activations are applied correctly."""
        # Create the layer with ReLU activation
        layer = Diagonal(units=self.units, activation='relu')
        layer.build(input_shape=(None, self.input_dim))
        
        # Set some weights to produce negative outputs
        kernel = np.ones(self.input_dim) * -1.0
        bias = np.zeros(self.units)
        layer.set_weights([kernel, bias])
        
        # Compute output
        X_ones = np.ones((self.batch_size, self.input_dim))
        output = layer(X_ones)
        
        # With negative weights and ReLU, all outputs should be 0
        self.assertTrue(np.all(output.numpy() == 0))
    
    def test_no_bias(self):
        """Test that the layer works without bias."""
        layer = Diagonal(units=self.units, use_bias=False)
        layer.build(input_shape=(None, self.input_dim))
        
        # Check that bias is None
        self.assertIsNone(layer.bias)
        
        # Set weights
        layer.set_weights([self.initial_kernel])
        
        # Compute output
        output = layer(self.X)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.units))
    
    def test_serialization(self):
        """Test that the layer can be serialized and deserialized."""
        # Create and build the layer
        layer = Diagonal(units=self.units, activation='relu')
        layer.build(input_shape=(None, self.input_dim))
        
        # Get config
        config = layer.get_config()
        
        # Create a new layer from the config
        new_layer = Diagonal.from_config(config)
        
        # Check that the config was properly restored
        self.assertEqual(new_layer.units, self.units)
        self.assertEqual(new_layer.activation.__name__, 'relu')
        self.assertTrue(new_layer.use_bias)
    
    def test_in_model(self):
        """Test that the layer works in a Keras model."""
        # Create a model with the Diagonal layer
        inputs = Input(shape=(self.input_dim,))
        x = Diagonal(units=self.units, activation='relu')(inputs)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create some test data
        X = np.random.random((32, self.input_dim))
        y = np.random.randint(0, 2, (32, 1))
        
        # Train the model for one step
        model.fit(X, y, epochs=1, verbose=0)
        
        # Check that prediction works
        predictions = model.predict(X)
        self.assertEqual(predictions.shape, (32, 1))


class TestSparseTFLayer(unittest.TestCase):
    """Test cases for the SparseTF layer."""
    
    def setUp(self):
        """Setup for each test case."""
        # Create a simple data set for testing
        np.random.seed(42)
        self.batch_size = 10
        self.input_dim = 8
        self.units = 4
        self.X = np.random.random((self.batch_size, self.input_dim))
        
        # Create a connectivity map
        self.conn_map = np.zeros((self.input_dim, self.units))
        
        # Connect each input to exactly one output
        for i in range(self.input_dim):
            j = i % self.units
            self.conn_map[i, j] = 1
        
        # Create non-zero indices from the map
        self.nonzero_ind = np.array(np.nonzero(self.conn_map)).T
        
        # Initialize weights for deterministic testing
        self.initial_kernel = np.ones(self.nonzero_ind.shape[0])
        self.initial_bias = np.zeros(self.units)
    
    def test_layer_initialization(self):
        """Test that the layer initializes correctly."""
        # Test with connectivity map
        layer1 = SparseTF(units=self.units, map=self.conn_map, activation='relu')
        self.assertEqual(layer1.units, self.units)
        self.assertEqual(layer1.activation.__name__, 'relu')
        self.assertTrue(layer1.use_bias)
        
        # Test with non-zero indices
        layer2 = SparseTF(units=self.units, nonzero_ind=self.nonzero_ind, activation='relu')
        self.assertEqual(layer2.units, self.units)
        self.assertEqual(layer2.activation.__name__, 'relu')
        self.assertTrue(layer2.use_bias)
    
    def test_build(self):
        """Test that the layer builds correctly."""
        # Test with connectivity map
        layer = SparseTF(units=self.units, map=self.conn_map)
        layer.build(input_shape=(None, self.input_dim))
        
        # Check that kernel and bias were created
        self.assertIsNotNone(layer.kernel_vector)
        self.assertIsNotNone(layer.bias)
        
        # Check shapes
        self.assertEqual(layer.kernel_vector.shape, (self.nonzero_ind.shape[0],))
        self.assertEqual(layer.bias.shape, (self.units,))
        
        # Check that nonzero_ind was created from the map
        self.assertEqual(layer.nonzero_ind.shape, self.nonzero_ind.shape)
    
    def test_call(self):
        """Test the forward pass of the layer."""
        # Create the layer with known weights for deterministic testing
        layer = SparseTF(units=self.units, nonzero_ind=self.nonzero_ind, activation=None, use_bias=True)
        layer.build(input_shape=(None, self.input_dim))
        
        # Set weights to known values
        layer.set_weights([self.initial_kernel, self.initial_bias])
        
        # Compute output
        output = layer(self.X)
        
        # Compute expected output manually
        expected_output = np.zeros((self.batch_size, self.units))
        for i in range(self.batch_size):
            for idx, (input_idx, output_idx) in enumerate(self.nonzero_ind):
                # Add the weighted input to the corresponding output unit
                expected_output[i, output_idx] += self.X[i, input_idx] * self.initial_kernel[idx]
        
        # Check that the outputs match
        self.assertTrue(np.allclose(output.numpy(), expected_output))
    
    def test_activation(self):
        """Test that activations are applied correctly."""
        # Create the layer with ReLU activation
        layer = SparseTF(units=self.units, nonzero_ind=self.nonzero_ind, activation='relu')
        layer.build(input_shape=(None, self.input_dim))
        
        # Set some weights to produce negative outputs
        kernel = np.ones(self.nonzero_ind.shape[0]) * -1.0
        bias = np.zeros(self.units)
        layer.set_weights([kernel, bias])
        
        # Compute output
        X_ones = np.ones((self.batch_size, self.input_dim))
        output = layer(X_ones)
        
        # With negative weights, negative inputs, and ReLU, all outputs should be 0
        self.assertTrue(np.all(output.numpy() == 0))
    
    def test_no_bias(self):
        """Test that the layer works without bias."""
        layer = SparseTF(units=self.units, nonzero_ind=self.nonzero_ind, use_bias=False)
        layer.build(input_shape=(None, self.input_dim))
        
        # Check that bias is None
        self.assertIsNone(layer.bias)
        
        # Set weights
        layer.set_weights([self.initial_kernel])
        
        # Compute output
        output = layer(self.X)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.units))
    
    def test_serialization(self):
        """Test that the layer can be serialized and deserialized."""
        # Create and build the layer
        layer = SparseTF(units=self.units, nonzero_ind=self.nonzero_ind, activation='relu')
        layer.build(input_shape=(None, self.input_dim))
        
        # Get config
        config = layer.get_config()
        
        # Create a new layer from the config
        new_layer = SparseTF.from_config(config)
        
        # Check that the config was properly restored
        self.assertEqual(new_layer.units, self.units)
        self.assertEqual(new_layer.activation.__name__, 'relu')
        self.assertTrue(new_layer.use_bias)
        self.assertTrue(np.array_equal(np.array(new_layer.nonzero_ind), np.array(layer.nonzero_ind)))
    
    def test_in_model(self):
        """Test that the layer works in a Keras model."""
        # Create a model with the SparseTF layer
        inputs = Input(shape=(self.input_dim,))
        x = SparseTF(units=self.units, nonzero_ind=self.nonzero_ind, activation='relu')(inputs)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create some test data
        X = np.random.random((32, self.input_dim))
        y = np.random.randint(0, 2, (32, 1))
        
        # Train the model for one step
        model.fit(X, y, epochs=1, verbose=0)
        
        # Check that prediction works
        predictions = model.predict(X)
        self.assertEqual(predictions.shape, (32, 1))
    
    def test_constraint(self):
        """Test that the SparseTFConstraint works correctly."""
        # Create a constraint
        constraint = SparseTFConstraint(nonzero_ind=self.nonzero_ind, shape=(self.input_dim, self.units))
        
        # Create a dense weight matrix with all ones
        weights = tf.ones((self.input_dim, self.units))
        
        # Apply the constraint
        constrained_weights = constraint(weights)
        
        # The constrained weights should have zeros everywhere except at the nonzero indices
        expected_weights = np.zeros((self.input_dim, self.units))
        for input_idx, output_idx in self.nonzero_ind:
            expected_weights[input_idx, output_idx] = 1.0
        
        # Check that the constrained weights match the expected weights
        self.assertTrue(np.allclose(constrained_weights.numpy(), expected_weights))


if __name__ == '__main__':
    # Run the tests
    unittest.main()
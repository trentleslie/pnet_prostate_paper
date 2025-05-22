"""
Simplified test suite for the TensorFlow 2.x compatible custom layers.

This script tests the basic functionality of the Diagonal and SparseTF layers
to ensure they work properly with TensorFlow 2.x.
"""

import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Import the TF2.x custom layers
from model.layers_custom_tf2 import Diagonal, SparseTF, SparseTFConstraint


class TestDiagonalLayer(unittest.TestCase):
    """Test cases for the TF2.x compatible Diagonal layer."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.input_dim = 10
        self.units = 5
        self.X = np.random.randn(self.batch_size, self.input_dim).astype(np.float32)
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = Diagonal(units=self.units, activation='relu')
        self.assertEqual(layer.units, self.units)
        self.assertEqual(layer.activation.__name__, 'relu')
    
    def test_build(self):
        """Test the build method."""
        layer = Diagonal(units=self.units)
        layer.build(input_shape=(None, self.input_dim))
        self.assertTrue(layer.built)
        self.assertEqual(layer.kernel.shape, (self.input_dim,))
        self.assertEqual(layer.bias.shape, (self.units,))
    
    def test_call(self):
        """Test the forward pass."""
        layer = Diagonal(units=self.units, activation=None)
        layer.build(input_shape=(None, self.input_dim))
        
        # Set weights to known values for deterministic testing
        kernel = np.ones(self.input_dim)
        bias = np.zeros(self.units)
        layer.set_weights([kernel, bias])
        
        # Compute output
        output = layer(self.X)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.units))
    
    def test_with_model(self):
        """Test integration with a Keras model."""
        inputs = Input(shape=(self.input_dim,))
        x = Diagonal(units=self.units, activation='relu')(inputs)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create dummy data
        X = np.random.randn(self.batch_size, self.input_dim).astype(np.float32)
        y = np.random.randint(0, 2, size=(self.batch_size, 1)).astype(np.float32)
        
        # Train for one step to verify everything works
        model.fit(X, y, epochs=1, verbose=0)
        
        # Test prediction
        preds = model.predict(X)
        self.assertEqual(preds.shape, (self.batch_size, 1))


class TestSparseTFLayer(unittest.TestCase):
    """Test cases for the TF2.x compatible SparseTF layer."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.input_dim = 10
        self.units = 5
        self.X = np.random.randn(self.batch_size, self.input_dim).astype(np.float32)
        
        # Create a test connectivity map
        self.map = np.zeros((self.input_dim, self.units))
        # Allow connections only for specific patterns
        for i in range(self.input_dim):
            self.map[i, i % self.units] = 1
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = SparseTF(units=self.units, map=self.map, activation='relu')
        self.assertEqual(layer.units, self.units)
        self.assertEqual(layer.activation.__name__, 'relu')
        np.testing.assert_array_equal(layer.map, self.map.astype(np.float32))
    
    def test_build(self):
        """Test the build method."""
        layer = SparseTF(units=self.units, map=self.map)
        layer.build(input_shape=(None, self.input_dim))
        self.assertTrue(layer.built)
        
        # Verify nonzero_ind is computed correctly
        expected_count = np.count_nonzero(self.map)
        self.assertEqual(layer.nonzero_ind.shape[0], expected_count)
    
    def test_call(self):
        """Test the forward pass."""
        layer = SparseTF(units=self.units, map=self.map, activation=None)
        layer.build(input_shape=(None, self.input_dim))
        
        # Compute output
        output = layer(self.X)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.units))
    
    def test_with_model(self):
        """Test integration with a Keras model."""
        inputs = Input(shape=(self.input_dim,))
        x = SparseTF(units=self.units, map=self.map, activation='relu')(inputs)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create dummy data
        X = np.random.randn(self.batch_size, self.input_dim).astype(np.float32)
        y = np.random.randint(0, 2, size=(self.batch_size, 1)).astype(np.float32)
        
        # Train for one step to verify everything works
        model.fit(X, y, epochs=1, verbose=0)
        
        # Test prediction
        preds = model.predict(X)
        self.assertEqual(preds.shape, (self.batch_size, 1))
    
    def test_constraint_enforces_sparsity(self):
        """Test that the SparseTFConstraint correctly enforces the connectivity pattern."""
        # Create a layer with the connectivity map
        layer = SparseTF(units=self.units, map=self.map)
        layer.build(input_shape=(None, self.input_dim))
        
        # Create a test input tensor
        x = tf.random.normal((1, self.input_dim))
        
        # Call the layer to ensure all variables are materialized
        _ = layer(x)
        
        # Directly create and apply a sparse constraint
        constraint = SparseTFConstraint(layer.nonzero_ind, (self.input_dim, self.units))
        
        # Create a test weight matrix filled with ones
        test_weights = tf.ones((self.input_dim, self.units))
        
        # Apply the constraint
        constrained_weights = constraint(test_weights)
        
        # Convert to numpy for easier testing
        constrained_np = constrained_weights.numpy()
        
        # Verify that weights are zero where connections should not exist
        for i in range(self.input_dim):
            for j in range(self.units):
                if self.map[i, j] == 0:
                    self.assertEqual(constrained_np[i, j], 0)
                else:
                    self.assertEqual(constrained_np[i, j], 1)


if __name__ == '__main__':
    unittest.main()
"""
Test script for validating the TensorFlow 2.x refactored model building functions.

This script tests the refactored implementations of build_pnet2 and get_pnet
to ensure they work correctly with TensorFlow 2.x.
"""

import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class MockData:
    """
    Mock data class to simulate the Data class used in model building functions.
    Now includes 'last_instance' for test inspection and 'include_histology_features' parameter.
    """
    last_instance = None  # Class variable to store the last created instance

    def __init__(self, id="test", type="prostate_paper", params=None, 
                 test_size=0.3, stratify=True, include_histology_features=False): # Default changed to False
        """Initialize mock data, tracking include_histology_features and storing instance."""
        MockData.last_instance = self # Store this instance for inspection
        self.id = id
        self.type = type
        self.params_arg = params or {} # This 'params' is the inner dict for feature selection etc.
        self.test_size = test_size
        self.stratify = stratify
        self.include_histology_features_received = include_histology_features # Store it
        
        logging.info(f"MockData initialized. include_histology_features_received: {self.include_histology_features_received}")

        effective_params_for_dims = self.params_arg

        if effective_params_for_dims and 'n_samples' in effective_params_for_dims:
            self.n_samples = effective_params_for_dims['n_samples']
        else:
            self.n_samples = 100
            
        if effective_params_for_dims and 'n_features' in effective_params_for_dims:
            self.n_features = effective_params_for_dims['n_features']
        else:
            self.n_features = 50
            
        if effective_params_for_dims and 'n_genes' in effective_params_for_dims:
            self.n_genes = effective_params_for_dims['n_genes']
        else:
            self.n_genes = 20
        
        self._generate_mock_data()
        
    def _generate_mock_data(self):
        """Generate mock data for testing."""
        # Create mock features (gene names)
        self.genes = [f'gene_{i}' for i in range(self.n_genes)]
        
        # Create mock feature columns - mimicking MultiIndex for genes->features
        columns = []
        for g in self.genes:
            for j in range(self.n_features // self.n_genes):
                columns.append(f"{g}_{j}")
        
        # Mock data matrix
        self.x = np.random.random((self.n_samples, self.n_features)).astype(np.float32)
        
        # Mock binary outcome
        self.y = np.random.randint(0, 2, size=(self.n_samples, 1)).astype(np.float32)
        
        # Mock sample info (sample IDs)
        self.info = np.array([f"sample_{i}" for i in range(self.n_samples)])
        
        # Create pandas Index for columns to match expected input
        self.columns = pd.Index(columns)
        
    def get_data(self):
        """Simulates the Data.get_data method by returning mock data"""
        return self.x, self.y, self.info, self.columns
        
    def get_train_validate_test(self):
        """Simulates the Data.get_train_validate_test method."""
        # For simplicity, just split the data into 60/20/20
        n_train = int(self.n_samples * 0.6)
        n_validate = int(self.n_samples * 0.2)
        
        x_train = self.x[:n_train]
        y_train = self.y[:n_train]
        info_train = self.info[:n_train]
        
        x_validate = self.x[n_train:n_train+n_validate]
        y_validate = self.y[n_train:n_train+n_validate]
        info_validate = self.info[n_train:n_train+n_validate]
        
        x_test = self.x[n_train+n_validate:]
        y_test = self.y[n_train+n_validate:]
        info_test = self.info[n_train+n_validate:]
        
        return x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, self.columns


# Import the refactored model building functions
from unittest.mock import patch # Added for per-test patching
from model.builders.builders_utils import get_pnet
from model.builders.prostate_models import build_pnet, build_pnet2 # Added build_pnet

# Create mock for ReactomeNetwork
class MockReactomeNetwork:
    def get_layers(self, n_levels, direction):
        """Mock for ReactomeNetwork.get_layers method."""
        # Generate a simple mock pathway hierarchy
        pathways = {}
        for level in range(n_levels):
            level_dict = {}
            for i in range(5):  # 5 pathways per level
                pathway_name = f"pathway_{level}_{i}"
                genes = [f"gene_{j}" for j in range(4)]  # 4 genes per pathway
                level_dict[pathway_name] = genes
            pathways[level] = level_dict
        return pathways

# We need to patch imports
import model.builders.prostate_models as pm
import model.builders.builders_utils as bu
# pm.Data = MockData # Global patch removed, will use @patch decorator
# Mock get_layer_maps function, to be used with @patch
def _mock_get_layer_maps_for_test(genes, n_hidden_layers, direction='root_to_leaf', add_unk_genes=True):
    """Mock for the get_layer_maps function that doesn't require Reactome data."""
    import pandas as pd
    import numpy as np
    
    maps = []
    
    # Start with the genes as provided (this represents the input dimension to first pathway layer)
    current_input_features = list(genes)
    
    # Create pathway structure for each layer
    for i in range(n_hidden_layers):
        # Calculate number of pathways for this layer
        num_pathways = max(1, len(current_input_features) // (i + 2))
        if num_pathways == 0:
            num_pathways = 1
        
        # Create pathway names for this layer
        pathway_names = [f'pathway_L{i}_{j}' for j in range(num_pathways)]
        
        # Add UNK pathway if requested (this creates an additional pathway, not input feature)
        if add_unk_genes:
            pathway_names.append('UNK')
        
        # Create mapping matrix: rows = current input features, columns = pathways
        n_input_features = len(current_input_features)
        n_pathways = len(pathway_names)
        
        # Initialize mapping matrix
        mapping_matrix = np.zeros((n_input_features, n_pathways))
        
        # Assign input features to pathways (excluding UNK for now)
        regular_pathways = pathway_names[:-1] if add_unk_genes else pathway_names
        genes_per_pathway = max(1, n_input_features // len(regular_pathways)) if regular_pathways else n_input_features
        
        for j, pathway_name in enumerate(regular_pathways):
            start_idx = j * genes_per_pathway
            end_idx = min((j + 1) * genes_per_pathway, n_input_features)
            
            # Assign features to this pathway
            for feature_idx in range(start_idx, end_idx):
                if feature_idx < n_input_features:
                    mapping_matrix[feature_idx, j] = 1
        
        # Handle UNK pathway - assign unassigned features to UNK
        if add_unk_genes:
            # Find features not assigned to any regular pathway
            assigned_features = np.sum(mapping_matrix[:, :-1], axis=1)
            unassigned_features = assigned_features == 0
            mapping_matrix[unassigned_features, -1] = 1
        
        # Create DataFrame with input features as index and pathways as columns
        filtered_map = pd.DataFrame(mapping_matrix, 
                                   index=current_input_features, 
                                   columns=pathway_names)
        
        maps.append(filtered_map)
        
        # For next layer, the input features are the pathway names from this layer
        current_input_features = pathway_names
    
    return maps

# bu.get_layer_maps = mock_get_layer_maps # Manual patch removed


class TestModelBuilders(unittest.TestCase):
    # Constants for log messages (to avoid typos in multiple tests)
    LOG_MSG_GENOMIC_ONLY_IGNORED = "Building P-NET{} model with genomic data only (histology ignored)"
    LOG_MSG_GENOMIC_ONLY_NOT_IMPL = "Building P-NET{} model with genomic data only (histology pathway not implemented)"
    LOG_MSG_HISTOLOGY_NOT_IMPL_WARNING = "ignore_missing_histology=False specified, but histology pathway not yet implemented. Using genomic data only."

    """Test cases for the refactored model building functions."""
    
    def setUp(self):
        """Setup for each test case."""
        # Configure the mock data parameters
        self.n_samples = 32
        self.n_features = 100
        self.n_genes = 20
        
        # Create optimizer
        self.optimizer = optimizers.Adam(learning_rate=0.001)
        
        # Default params for build_pnet2
        self.default_params_pnet2 = {
            'optimizer': self.optimizer,
            'w_reg': 0.01,
            'w_reg_outcomes': 0.01,
            'dropout': 0.5,
            'activation': 'relu',
            'n_hidden_layers': 1,
            'loss_weights': 1.0, 
            'use_bias': False, 
            'loss': 'binary_crossentropy',
            'direction': 'root_to_leaf',
            'batch_normal': False, 
            'kernel_initializer': 'glorot_uniform',
            'shuffle_genes': False, 
            'attention': False, 
            'dropout_testing': False, 
            'non_neg': False, 
            'repeated_outcomes': True, 
            'sparse_first_layer': True,
            'data_params': {
                'id': 'test_pnet2',
                'type': 'prostate_paper',
                'params': {
                    'n_samples': self.n_samples, 
                    'n_features': self.n_features, 
                    'n_genes': self.n_genes
                }
            }
        }

        # Default params for build_pnet (original)
        self.default_params_pnet1 = {
            'optimizer': self.optimizer,
            'w_reg': 0.01,
            'dropout': 0.5,
            'activation': 'tanh', # Original build_pnet often used tanh
            'n_hidden_layers': 1,
            'use_bias': False,
            'loss': 'binary_crossentropy',
            'direction': 'root_to_leaf',
            'batch_normal': False,
            'kernel_initializer': 'glorot_uniform',
            'shuffle_genes': False,
            'reg_outcomes': False, # Specific to build_pnet
            'data_params': {
                'id': 'test_pnet1',
                'type': 'prostate_paper',
                'params': {
                    'n_samples': self.n_samples, 
                    'n_features': self.n_features, 
                    'n_genes': self.n_genes
                }
            }
        }
        
        # Create default params for build_pnet2
        self.default_params = { # Keep self.default_params for any existing tests that might use it, aliasing to pnet2 for now

            'optimizer': self.optimizer,
            'w_reg': 0.01,
            'w_reg_outcomes': 0.01,
            'dropout': 0.5,
            'activation': 'relu',
            'n_hidden_layers': 1,
            'data_params': {
                'id': 'test',
                'type': 'prostate_paper',
                'params': {
                    'n_samples': self.n_samples, 
                    'n_features': self.n_features, 
                    'n_genes': self.n_genes
                }
            }
        }
        
    def tearDown(self):
        """Cleanup after each test case."""
        # No need to restore get_layer_maps manually if using @patch on the class
        pass

    # --- Tests for build_pnet2 --- 
    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_ignore_histology_default(self):
        """Test build_pnet2 default behavior (ignore_missing_histology=True)."""
        params = self.default_params_pnet2.copy()
        # ignore_missing_histology is not in params, build_pnet2 defaults it to True

        with self.assertLogs(level='INFO') as cm:
            model, _ = build_pnet2(**params)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertFalse(MockData.last_instance.include_histology_features_received)
        self.assertTrue(any(self.LOG_MSG_GENOMIC_ONLY_IGNORED.format("2") in log_msg for log_msg in cm.output))

    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_ignore_histology_true(self):
        """Test build_pnet2 with ignore_missing_histology=True."""
        params = self.default_params_pnet2.copy()
        params['ignore_missing_histology'] = True

        with self.assertLogs(level='INFO') as cm:
            model, _ = build_pnet2(**params)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertFalse(MockData.last_instance.include_histology_features_received)
        self.assertTrue(any(self.LOG_MSG_GENOMIC_ONLY_IGNORED.format("2") in log_msg for log_msg in cm.output))

    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_ignore_histology_false(self):
        """Test build_pnet2 with ignore_missing_histology=False."""
        params = self.default_params_pnet2.copy()
        params['ignore_missing_histology'] = False

        with self.assertLogs(level='INFO') as cm: # Captures INFO and WARNING
            model, _ = build_pnet2(**params)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertTrue(MockData.last_instance.include_histology_features_received)
        # Check for specific warning and then the fallback info message
        self.assertTrue(any(self.LOG_MSG_HISTOLOGY_NOT_IMPL_WARNING in log_msg for log_msg in cm.output), "Warning for not implemented histology missing")
        self.assertTrue(any(self.LOG_MSG_GENOMIC_ONLY_NOT_IMPL.format("2") in log_msg for log_msg in cm.output), "Fallback to genomic only message missing")

    # --- Tests for build_pnet (original) --- 
    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet_ignore_histology_default(self):
        """Test build_pnet default behavior (ignore_missing_histology=True)."""
        params = self.default_params_pnet1.copy()
        # ignore_missing_histology is not in params, build_pnet defaults it to True

        with self.assertLogs(level='INFO') as cm:
            model, _ = build_pnet(**params)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertFalse(MockData.last_instance.include_histology_features_received)
        self.assertTrue(any(self.LOG_MSG_GENOMIC_ONLY_IGNORED.format("") in log_msg for log_msg in cm.output))

    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet_ignore_histology_true(self):
        """Test build_pnet with ignore_missing_histology=True."""
        params = self.default_params_pnet1.copy()
        params['ignore_missing_histology'] = True

        with self.assertLogs(level='INFO') as cm:
            model, _ = build_pnet(**params)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertFalse(MockData.last_instance.include_histology_features_received)
        self.assertTrue(any(self.LOG_MSG_GENOMIC_ONLY_IGNORED.format("") in log_msg for log_msg in cm.output))

    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet_ignore_histology_false(self):
        """Test build_pnet with ignore_missing_histology=False."""
        params = self.default_params_pnet1.copy()
        params['ignore_missing_histology'] = False

        with self.assertLogs(level='INFO') as cm:
            model, _ = build_pnet(**params)
        
        self.assertIsInstance(model, tf.keras.Model)
        self.assertTrue(MockData.last_instance.include_histology_features_received)
        self.assertTrue(any(self.LOG_MSG_HISTOLOGY_NOT_IMPL_WARNING in log_msg for log_msg in cm.output), "Warning for not implemented histology missing")
        self.assertTrue(any(self.LOG_MSG_GENOMIC_ONLY_NOT_IMPL.format("") in log_msg for log_msg in cm.output), "Fallback to genomic only message missing")

    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    def test_get_pnet_basic(self):
        """Test the get_pnet function with basic configuration."""
        # Create a mock data instance
        mock_data = MockData(
            id="test",
            type="prostate_paper",
            params={
                'n_samples': self.n_samples, 
                'n_features': self.n_features, 
                'n_genes': self.n_genes
            }
        )
        
        # Get the mock data
        X, y, info, cols = mock_data.get_data()
        
        # Create input tensor
        input_tensor = tf.keras.layers.Input(shape=(self.n_features,), dtype='float32', name='inputs')
        
        # Extract genes (in real code this would be cols.levels[0])
        genes = [f'gene_{i}' for i in range(self.n_genes)]
        
        # Call get_pnet with minimal parameters
        outcome, decision_outcomes, feature_names = get_pnet(
            inputs=input_tensor,
            features=cols,
            genes=genes,
            n_hidden_layers=1,
            direction='root_to_leaf',
            activation='relu',
            activation_decision='sigmoid',
            w_reg=0.01,
            w_reg_outcomes=0.01,
            dropout=0.5,
            sparse=True,
            add_unk_genes=True,
            batch_normal=False,
            kernel_initializer='glorot_uniform'
        )
        
        # Verify the outcome tensor
        self.assertIsNotNone(outcome)
        
        # Verify decision_outcomes is a list with expected length
        self.assertIsInstance(decision_outcomes, list)
        self.assertEqual(len(decision_outcomes), 2)  # Should have 2 outputs for n_hidden_layers=1
        
        # Verify feature_names dictionary
        self.assertIsInstance(feature_names, dict)
        self.assertIn('h0', feature_names)
    
    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_basic(self):
        """Test the build_pnet2 function with basic configuration."""
        # We've already monkeypatched the Data class in the imports
        
        try:
            # Call build_pnet2 with minimal parameters
            model, feature_names = build_pnet2(
                optimizer=self.optimizer,
                w_reg=0.01,
                w_reg_outcomes=0.01,
                n_hidden_layers=1,
                data_params={
                    'id': 'test',
                    'type': 'prostate_paper',
                    'params': {
                        'n_samples': self.n_samples, 
                        'n_features': self.n_features, 
                        'n_genes': self.n_genes
                    }
                }
            )
            
            # Verify model is a Keras Model
            self.assertIsInstance(model, Model)
            
            # Verify model has expected I/O shape
            self.assertEqual(model.input_shape, (None, self.n_features))
            
            # For repeated_outcomes=True (default), output should be a list
            self.assertIsInstance(model.output, list)
            self.assertEqual(len(model.output), 2)  # Should have 2 outputs for n_hidden_layers=1
            
            # Verify feature_names dictionary
            self.assertIsInstance(feature_names, dict)
            self.assertIn('inputs', feature_names)
            self.assertIn('h0', feature_names)
            
            # Create dummy data for a forward pass
            X = np.random.random((4, self.n_features))
            
            # Test forward pass
            outputs = model.predict(X)
            
            # Verify outputs shape
            self.assertEqual(len(outputs), 2)  # Should have 2 outputs for n_hidden_layers=1
            self.assertEqual(outputs[0].shape, (4, 1))  # Each output should be (batch_size, 1)
            self.assertEqual(outputs[1].shape, (4, 1))
            
        finally:
            # No need to restore Data class as we've patched it globally
            pass
    
    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_single_output(self):
        """Test the build_pnet2 function with single output configuration."""
        # We've already monkeypatched the Data class in the imports
        
        try:
            # Call build_pnet2 with repeated_outcomes=False
            model, feature_names = build_pnet2(
                optimizer=self.optimizer,
                w_reg=0.01,
                w_reg_outcomes=0.01,
                n_hidden_layers=1,
                repeated_outcomes=False,  # Only use the last output
                data_params={
                    'id': 'test',
                    'type': 'prostate_paper',
                    'params': {
                        'n_samples': self.n_samples, 
                        'n_features': self.n_features, 
                        'n_genes': self.n_genes
                    }
                }
            )
            
            # Verify model is a Keras Model
            self.assertIsInstance(model, Model)
            
            # Verify model has expected I/O shape
            self.assertEqual(model.input_shape, (None, self.n_features))
            
            # For repeated_outcomes=False, output should be a single tensor
            self.assertEqual(model.output_shape, (None, 1))
            
            # Create dummy data for a forward pass
            X = np.random.random((4, self.n_features))
            
            # Test forward pass
            outputs = model.predict(X)
            
            # Verify outputs shape
            self.assertEqual(outputs.shape, (4, 1))  # Output should be (batch_size, 1)
            
        finally:
            # No need to restore Data class as we've patched it globally
            pass
    
    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_with_sparse_options(self):
        """Test the build_pnet2 function with different sparse layer options."""
        # We've already monkeypatched the Data class in the imports
        
        try:
            # Configuration combinations to test
            configs = [
                {'sparse': True, 'sparse_first_layer': True},
                {'sparse': True, 'sparse_first_layer': False},
                {'sparse': False, 'sparse_first_layer': True},
                {'sparse': False, 'sparse_first_layer': False}
            ]
            
            for config in configs:
                # Create params with the current config
                params = {
                    **self.default_params,
                    **config,
                    'n_hidden_layers': 1
                }
                
                # Call build_pnet2 with the current config
                model, _ = build_pnet2(**params)
                
                # Verify model is a Keras Model
                self.assertIsInstance(model, Model)
                
                # Create dummy data for a forward pass
                X = np.random.random((4, self.n_features))
                
                # Test forward pass
                outputs = model.predict(X)
                
                # Verify outputs - just checking it runs without error
                self.assertEqual(len(outputs), 2)  # Should have 2 outputs for n_hidden_layers=1
                
        finally:
            # No need to restore Data class as we've patched it globally
            pass
    
    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_with_batch_norm(self):
        """Test the build_pnet2 function with batch normalization."""
        # We've already monkeypatched the Data class in the imports
        
        try:
            # Call build_pnet2 with batch_normal=True
            model, _ = build_pnet2(
                **self.default_params,
                batch_normal=True
            )
            
            # Verify model is a Keras Model
            self.assertIsInstance(model, Model)
            
            # Create dummy data for a forward pass
            X = np.random.random((4, self.n_features))
            
            # Test forward pass
            outputs = model.predict(X)
            
            # Verify outputs - just checking it runs without error
            self.assertEqual(len(outputs), 2)  # Should have 2 outputs for n_hidden_layers=1
            
            # Check that the model contains BatchNormalization layers
            batch_norm_layers = [layer for layer in model.layers if 'batch_normalization' in layer.name.lower()]
            self.assertGreater(len(batch_norm_layers), 0)
            
        finally:
            # No need to restore Data class as we've patched it globally
            pass
    
    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_with_attention(self):
        """Test the build_pnet2 function with attention mechanism."""
        # We've already monkeypatched the Data class in the imports
        
        try:
            # Call build_pnet2 with attention=True
            model, _ = build_pnet2(
                **self.default_params,
                attention=True
            )
            
            # Verify model is a Keras Model
            self.assertIsInstance(model, Model)
            
            # Create dummy data for a forward pass
            X = np.random.random((4, self.n_features))
            
            # Test forward pass
            outputs = model.predict(X)
            
            # Verify outputs - just checking it runs without error
            self.assertEqual(len(outputs), 2)  # Should have 2 outputs for n_hidden_layers=1
            
            # Check that the model contains attention-related layers
            attention_layers = [layer for layer in model.layers if 'attention' in layer.name.lower()]
            self.assertGreater(len(attention_layers), 0)
            
        finally:
            # No need to restore Data class as we've patched it globally
            pass
    
    @patch('model.builders.builders_utils.get_layer_maps', new=_mock_get_layer_maps_for_test)
    @patch('model.builders.prostate_models.Data', new=MockData)
    def test_build_pnet2_with_multiple_hidden_layers(self):
        """Test the build_pnet2 function with multiple hidden layers."""
        # We've already monkeypatched the Data class in the imports
        
        try:
            # Call build_pnet2 with n_hidden_layers=2
            # Create a custom params dictionary without n_hidden_layers
            params = self.default_params.copy()
            params.pop('n_hidden_layers', None)
            
            model, feature_names = build_pnet2(
                **params,
                n_hidden_layers=2
            )
            
            # Verify model is a Keras Model
            self.assertIsInstance(model, Model)
            
            # Create dummy data for a forward pass
            X = np.random.random((4, self.n_features))
            
            # Test forward pass
            outputs = model.predict(X)
            
            # Verify outputs
            self.assertEqual(len(outputs), 3)  # Should have 3 outputs for n_hidden_layers=2
            
            # Verify feature_names dictionary has entries for each layer
            self.assertIn('h0', feature_names)
            # h1 may or may not be present depending on how pathways are defined
            
        finally:
            # No need to restore Data class as we've patched it globally
            pass


if __name__ == '__main__':
    unittest.main()
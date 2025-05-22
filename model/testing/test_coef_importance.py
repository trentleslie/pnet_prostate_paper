"""
Test script for validating the TF2.x implementation of get_coef_importance.
This script tests the refactored implementation of get_coef_importance which now
relies on activation gradients instead of deepexplain.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# Import the required functions
from model.model_utils import get_coef_importance
from model.coef_weights_utils import get_activation_gradients_importance
from model.nn import Model

def create_test_model():
    """Create a simple model for testing coefficient importance"""
    inputs = tf.keras.layers.Input(shape=(10,))
    hidden1 = tf.keras.layers.Dense(8, activation='relu', name='hidden1')(inputs)
    hidden2 = tf.keras.layers.Dense(5, activation='relu', name='hidden2')(hidden1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(hidden2)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def create_multi_output_test_model():
    """Create a model with multiple outputs for testing"""
    inputs = tf.keras.layers.Input(shape=(10,))
    hidden = tf.keras.layers.Dense(8, activation='relu', name='hidden')(inputs)
    output1 = tf.keras.layers.Dense(1, activation='sigmoid', name='output1')(hidden)
    output2 = tf.keras.layers.Dense(1, activation='sigmoid', name='output2')(hidden)
    model = tf.keras.models.Model(inputs=inputs, outputs=[output1, output2])
    model.compile(
        optimizer='adam',
        loss={'output1': 'binary_crossentropy', 'output2': 'binary_crossentropy'}
    )
    return model


def generate_test_data(n_samples=100):
    """Generate test data for importance calculation"""
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n_samples, 10)
    y = (np.random.randn(n_samples) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


def generate_multi_output_test_data(n_samples=100):
    """Generate test data for multi-output importance calculation"""
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n_samples, 10)
    y1 = (np.random.randn(n_samples) > 0).astype(np.float32).reshape(-1, 1)
    y2 = (np.random.randn(n_samples) > 0).astype(np.float32).reshape(-1, 1)
    return X, [y1, y2]


def visualize_importance(importance_dict, title="Feature Importance Analysis"):
    """Create visual analysis of feature importance scores"""
    plt.figure(figsize=(12, 8))
    
    # Plot importance scores for each layer
    n_layers = len(importance_dict)
    rows = max(1, (n_layers + 1) // 2)
    cols = min(2, n_layers)
    
    for i, (layer_name, importance) in enumerate(importance_dict.items()):
        plt.subplot(rows, cols, i+1)
        
        # If the importance is 2D or more, take sum or mean along appropriate axis
        if importance.ndim > 1:
            importance_flat = np.sum(importance, axis=tuple(range(importance.ndim-1)))
        else:
            importance_flat = importance
            
        # Create bar plot
        plt.bar(range(len(importance_flat)), np.abs(importance_flat))
        plt.title(f"Layer: {layer_name}")
        plt.xlabel("Feature Index")
        plt.ylabel("Absolute Importance")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"importance_{title.replace(' ', '_').lower()}.png")
    plt.close()


def test_get_coef_importance_methods():
    """Test different feature importance methods"""
    print("\n=== Testing Feature Importance Methods ===")
    model = create_test_model()
    X, y = generate_test_data()
    
    methods = [
        'deepexplain_grad*input',  # The default method
        'deepexplain_grad',
        'gradient_outcome',
        'gradient_outcome*input'
    ]
    
    results = {}
    
    for method in methods:
        print(f"\nTesting method: {method}")
        importance = get_coef_importance(model, X, y, target=-1, feature_importance=method)
        
        # Basic validation
        print(f"  Number of layers with importance scores: {len(importance)}")
        for layer_name, imp in importance.items():
            print(f"  Layer '{layer_name}' importance shape: {imp.shape}")
            print(f"  Min/Max/Mean importance: {imp.min():.4f}/{imp.max():.4f}/{imp.mean():.4f}")
        
        results[method] = importance
        
        # Visualize the importance scores
        visualize_importance(importance, f"Method: {method}")
    
    return results


def test_activation_gradients_importance():
    """Test get_activation_gradients_importance directly"""
    print("\n=== Testing Activation Gradients Importance ===")
    model = create_test_model()
    X, y = generate_test_data()
    
    methods = ['input*grad', 'grad', 'input', 'grad_abs', 'input*grad_abs']
    
    results = {}
    
    for method in methods:
        print(f"\nTesting method: {method}")
        importance = get_activation_gradients_importance(model, X, y, method=method)
        
        # Basic validation
        print(f"  Number of layers with importance scores: {len(importance)}")
        for layer_name, imp in importance.items():
            print(f"  Layer '{layer_name}' importance shape: {imp.shape}")
            print(f"  Min/Max/Mean importance: {imp.min():.4f}/{imp.max():.4f}/{imp.mean():.4f}")
        
        results[method] = importance
        
        # Visualize the importance scores
        visualize_importance(importance, f"Activation Method: {method}")
    
    return results


def test_multi_output_model():
    """Test importance calculation with a multi-output model"""
    print("\n=== Testing Multi-Output Model ===")
    model = create_multi_output_test_model()
    X, y = generate_multi_output_test_data()
    
    # Test with default method
    importance = get_coef_importance(model, X, y, target=-1, feature_importance='deepexplain_grad*input')
    
    # Basic validation
    print(f"Number of layers with importance scores: {len(importance)}")
    for layer_name, imp in importance.items():
        print(f"Layer '{layer_name}' importance shape: {imp.shape}")
        print(f"Min/Max/Mean importance: {imp.min():.4f}/{imp.max():.4f}/{imp.mean():.4f}")
    
    # Visualize the importance scores
    visualize_importance(importance, "Multi-Output Model")
    
    return importance


def test_target_specification():
    """Test specifying different targets for importance calculation"""
    print("\n=== Testing Target Specification ===")
    model = create_multi_output_test_model()
    X, y = generate_multi_output_test_data()
    
    targets = [-1, 0, 1, 'output1', 'output2']
    method = 'deepexplain_grad*input'
    
    results = {}
    
    for target in targets:
        print(f"\nTesting target: {target}")
        try:
            importance = get_coef_importance(model, X, y, target=target, feature_importance=method)
            
            # Basic validation
            print(f"  Number of layers with importance scores: {len(importance)}")
            for layer_name, imp in importance.items():
                print(f"  Layer '{layer_name}' importance shape: {imp.shape}")
                print(f"  Min/Max/Mean importance: {imp.min():.4f}/{imp.max():.4f}/{imp.mean():.4f}")
            
            results[str(target)] = importance
            
            # Visualize the importance scores
            visualize_importance(importance, f"Target: {target}")
        except Exception as e:
            print(f"  Error with target {target}: {str(e)}")
    
    return results


def compare_methods_correlation():
    """Compare correlation between different importance calculation methods"""
    print("\n=== Comparing Method Correlations ===")
    model = create_test_model()
    X, y = generate_test_data()
    
    methods = [
        'deepexplain_grad*input',
        'deepexplain_grad',
        'gradient_outcome*input',
        'gradient_outcome'
    ]
    
    # Calculate importance for each method
    importance_results = {}
    for method in methods:
        importance_results[method] = get_coef_importance(model, X, y, target=-1, feature_importance=method)
    
    # For each layer, calculate correlation between methods
    for layer_name in importance_results[methods[0]].keys():
        print(f"\nLayer: {layer_name}")
        
        # Create correlation matrix
        method_values = {}
        for method in methods:
            if layer_name in importance_results[method]:
                # Flatten importance values if multi-dimensional
                imp = importance_results[method][layer_name]
                if imp.ndim > 1:
                    imp = np.sum(imp, axis=tuple(range(imp.ndim-1)))
                method_values[method] = imp
        
        # Skip if any method doesn't have this layer
        if len(method_values) != len(methods):
            print("  Skipping - not all methods have this layer")
            continue
        
        # Calculate correlation matrix
        n_methods = len(methods)
        corr_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Calculate correlation coefficient
                    values1 = method_values[method1]
                    values2 = method_values[method2]
                    
                    # Handle potential shape mismatches
                    min_len = min(len(values1), len(values2))
                    values1 = values1[:min_len]
                    values2 = values2[:min_len]
                    
                    corr = np.corrcoef(values1, values2)[0, 1]
                    corr_matrix[i, j] = corr
                    print(f"  Correlation between {method1} and {method2}: {corr:.4f}")
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(n_methods), methods, rotation=45)
        plt.yticks(range(n_methods), methods)
        plt.title(f"Method Correlation for Layer '{layer_name}'")
        plt.tight_layout()
        plt.savefig(f"correlation_matrix_{layer_name}.png")
        plt.close()
    
    return importance_results


if __name__ == "__main__":
    print("Testing TensorFlow 2.x implementation of get_coef_importance")
    
    # Run the tests
    test_get_coef_importance_methods()
    test_activation_gradients_importance()
    test_multi_output_model()
    test_target_specification()
    compare_methods_correlation()
    
    print("\nAll tests completed. Check the output images for visualizations.")
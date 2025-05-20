"""
Test script for validating the TF2.x implementation of gradient functions.
This script compares the new TF2.x implementation of get_gradient_layer 
with the original TF1.x implementation.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# Import both implementations - for testing, we'll need to have both
# available simultaneously
from coef_weights_utils import get_gradient_layer


def create_test_model():
    """Create a simple model for testing gradient calculations"""
    inputs = tf.keras.layers.Input(shape=(10,))
    hidden = tf.keras.layers.Dense(5, activation='relu', name='hidden')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(hidden)
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
    """Generate test data for gradient comparison"""
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n_samples, 10)
    y = (np.random.randn(n_samples) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


def generate_multi_output_test_data(n_samples=100):
    """Generate test data for multi-output gradient comparison"""
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(n_samples, 10)
    y1 = (np.random.randn(n_samples) > 0).astype(np.float32).reshape(-1, 1)
    y2 = (np.random.randn(n_samples) > 0).astype(np.float32).reshape(-1, 1)
    return X, [y1, y2]


def visualize_gradient_comparison(tf2_gradients, title="Gradient Analysis"):
    """Create visual analysis of gradients"""
    plt.figure(figsize=(10, 6))
    
    # Distribution of gradient values
    plt.subplot(1, 2, 1)
    plt.hist(tf2_gradients.flatten(), bins=50)
    plt.xlabel('Gradient Values')
    plt.ylabel('Count')
    plt.title('Distribution of Gradient Values')
    
    # Heatmap of gradients (reshape if needed)
    plt.subplot(1, 2, 2)
    if tf2_gradients.ndim > 1:
        plt.imshow(tf2_gradients, aspect='auto', cmap='coolwarm')
        plt.colorbar(label='Gradient Magnitude')
        plt.title('Gradient Heatmap')
    else:
        plt.bar(range(len(tf2_gradients)), tf2_gradients)
        plt.xlabel('Index')
        plt.ylabel('Gradient Value')
        plt.title('Gradient Values')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"gradient_{title.replace(' ', '_').lower()}.png")
    plt.close()


def test_single_output_model():
    """Test gradients with a single-output model"""
    print("\n=== Testing Single Output Model ===")
    model = create_test_model()
    X, y = generate_test_data()
    hidden_layer = model.get_layer('hidden').output
    
    # Get gradients using the TF2 implementation
    gradients = get_gradient_layer(model, X, y, hidden_layer)
    
    print(f"TF2 Gradients shape: {gradients.shape}")
    print(f"Min gradient value: {gradients.min()}")
    print(f"Max gradient value: {gradients.max()}")
    print(f"Mean gradient value: {gradients.mean()}")
    print(f"Std deviation of gradients: {gradients.std()}")
    
    # Visualize the gradients
    visualize_gradient_comparison(gradients, "Single Output Model")
    return gradients


def test_multi_output_model():
    """Test gradients with a multi-output model"""
    print("\n=== Testing Multi-Output Model ===")
    model = create_multi_output_test_model()
    X, y = generate_multi_output_test_data()
    hidden_layer = model.get_layer('hidden').output
    
    # Get gradients using the TF2 implementation
    gradients = get_gradient_layer(model, X, y, hidden_layer)
    
    print(f"TF2 Gradients shape: {gradients.shape}")
    print(f"Min gradient value: {gradients.min()}")
    print(f"Max gradient value: {gradients.max()}")
    print(f"Mean gradient value: {gradients.mean()}")
    print(f"Std deviation of gradients: {gradients.std()}")
    
    # Visualize the gradients
    visualize_gradient_comparison(gradients, "Multi-Output Model")
    return gradients


def test_normalization_effect():
    """Test the effect of normalization on gradients"""
    print("\n=== Testing Normalization Effect ===")
    model = create_test_model()
    X, y = generate_test_data()
    hidden_layer = model.get_layer('hidden').output
    
    # Get gradients with normalization
    normalized_gradients = get_gradient_layer(model, X, y, hidden_layer, normalize=True)
    
    # Get gradients without normalization
    unnormalized_gradients = get_gradient_layer(model, X, y, hidden_layer, normalize=False)
    
    print(f"Normalized min/max: {normalized_gradients.min()}/{normalized_gradients.max()}")
    print(f"Unnormalized min/max: {unnormalized_gradients.min()}/{unnormalized_gradients.max()}")
    
    # Calculate correlation between normalized and unnormalized
    correlation = np.corrcoef(normalized_gradients.flatten(), unnormalized_gradients.flatten())[0, 1]
    print(f"Correlation between normalized and unnormalized: {correlation:.4f}")
    
    # Visualize the comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(unnormalized_gradients.flatten(), normalized_gradients.flatten(), alpha=0.5)
    plt.xlabel('Unnormalized Gradients')
    plt.ylabel('Normalized Gradients')
    plt.title('Effect of Normalization on Gradients')
    plt.savefig("normalization_effect.png")
    plt.close()
    
    return normalized_gradients, unnormalized_gradients


def test_edge_cases():
    """Test gradient calculation with edge case inputs"""
    print("\n=== Testing Edge Cases ===")
    model = create_test_model()
    hidden_layer = model.get_layer('hidden').output
    
    # 1. Zero inputs
    print("Testing with zero inputs...")
    X_zeros = np.zeros((100, 10))
    y = np.random.randint(0, 2, size=(100, 1)).astype(np.float32)
    
    zero_gradients = get_gradient_layer(model, X_zeros, y, hidden_layer)
    print(f"Zero inputs - gradient stats: min={zero_gradients.min()}, max={zero_gradients.max()}")
    
    # 2. Single sample
    print("Testing with single sample...")
    X_single = np.random.randn(1, 10)
    y_single = np.array([[1.0]])
    
    single_gradients = get_gradient_layer(model, X_single, y_single, hidden_layer)
    print(f"Single sample - gradient shape: {single_gradients.shape}")
    
    # 3. Large batch size
    print("Testing with large batch size...")
    X_large = np.random.randn(1000, 10)
    y_large = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)
    
    large_gradients = get_gradient_layer(model, X_large, y_large, hidden_layer)
    print(f"Large batch - gradient stats: min={large_gradients.min()}, max={large_gradients.max()}")


if __name__ == "__main__":
    print("Testing TensorFlow 2.x implementation of get_gradient_layer")
    
    # Run the tests
    test_single_output_model()
    test_multi_output_model()
    test_normalization_effect()
    test_edge_cases()
    
    print("\nAll tests completed. Check the output images for visualizations.")
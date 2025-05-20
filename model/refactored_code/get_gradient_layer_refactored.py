import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def get_gradient_layer_tf2(model, X, y, layer, normalize=True):
    """
    Refactored version of get_gradient_layer using TensorFlow 2.x's tf.GradientTape
    instead of the TF1.x K.function and K.gradients approach.
    
    Args:
        model: The Keras model
        X: Input data
        y: Target data (can be a single array or a list of arrays)
        layer: The layer tensor to compute gradients for
        normalize: Whether to normalize gradients (default: True)
    
    Returns:
        Gradients of the layer with respect to the model's loss
    """
    # Handle case where y is a list of outputs (for multi-output models)
    if isinstance(y, list):
        n = len(y)
    else:
        n = 1
        y = [y]  # Convert to list for consistent handling
    
    nb_sample = X.shape[0]
    
    # Create sample weights (ones by default)
    sample_weights = [np.ones(nb_sample) for _ in range(n)]
    
    # Reshape targets if needed
    y_reshaped = []
    for i in range(n):
        y_reshaped.append(y[i].reshape((nb_sample, 1)))
    
    # In TF2 with eager execution, we use tf.GradientTape to compute gradients
    @tf.function  # Use tf.function for better performance
    def get_gradients_fn(x, targets, weights):
        # Convert inputs to tensors
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Create a list of targets as tensors
        target_tensors = [tf.convert_to_tensor(t, dtype=tf.float32) for t in targets]
        
        # Create a list of sample weights as tensors
        weight_tensors = [tf.convert_to_tensor(w, dtype=tf.float32) for w in weights]
        
        with tf.GradientTape() as tape:
            tape.watch(layer)  # We need to watch the layer tensor explicitly
            
            # Get model predictions (forward pass)
            outputs = model(x, training=False)  # training=False for test mode (equivalent to K.learning_phase() = 0)
            
            # If the model has multiple outputs, handle as a list
            if isinstance(outputs, list):
                losses = []
                for i in range(n):
                    # Apply sample weights to loss
                    loss_i = model.loss_functions[i](target_tensors[i], outputs[i]) 
                    weighted_loss_i = loss_i * weight_tensors[i]
                    losses.append(tf.reduce_mean(weighted_loss_i))
                total_loss = tf.add_n(losses)
            else:
                # Single output model
                loss = model.loss(target_tensors[0], outputs)
                weighted_loss = loss * weight_tensors[0]
                total_loss = tf.reduce_mean(weighted_loss)
            
        # Compute gradients of the total loss with respect to the layer
        grads = tape.gradient(total_loss, layer)
        
        # Multiply gradients by the layer (as per original implementation)
        weighted_grads = layer * grads
        
        # Normalize if specified
        if normalize:
            weighted_grads = weighted_grads / (tf.sqrt(tf.reduce_mean(tf.square(weighted_grads))) + 1e-5)
        
        return weighted_grads
    
    # Call the gradient function with our inputs
    gradients = get_gradients_fn(X, y_reshaped, sample_weights)
    
    # Convert to numpy for consistency with original function
    gradients_np = gradients.numpy()
    
    # In the original function, [0] was applied to get the first element
    # We're returning the gradients with the same dimensionality as the original
    if gradients_np.ndim > 0:
        return gradients_np[0]
    else:
        return gradients_np
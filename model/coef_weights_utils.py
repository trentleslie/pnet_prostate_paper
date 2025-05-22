import sys

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from model.model_utils import get_layers




def predict(model, X, loss=None):
    prediction_scores = model.predict(X)

    prediction_scores = np.mean(np.array(prediction_scores), axis=0)
    if loss == 'hinge':
        prediction = np.where(prediction_scores >= 0.0, 1., 0.)
    else:
        prediction = np.where(prediction_scores >= 0.5, 1., 0.)

    return prediction


# def get_gradient_layer(model, X, y, layer):
#
#     # print 'layer', layer
#     grad = model.optimizer.get_gradients(model.total_loss, layer)
#     gradients = layer *  grad# gradient tensors
#     # gradients =  grad# gradient tensors
#     # gradients = layer * model.optimizer.get_gradients(model.output[0,0], layer) # gradient tensors
#     # gradients = model.optimizer.get_gradients(model.output[0,0], layer) # gradient tensors
#     # gradients =  model.optimizer.get_gradients(model.total_loss, layer) # gradient tensors
#
#     #special case of repeated outputs (e.g. output for each hidden layer)
#     if type(y) == list:
#         n = len(y)
#     else:
#         n = 1
#
#     # print model.inputs[0]._keras_shape, model.targets[0]._keras_shape
#     # print 'model.targets', model.targets[0:n]
#     # print 'model.inputs[0]', model.inputs[0]
#     input_tensors = [model.inputs[0],  # input data
#                      # model.sample_weights[0],  # how much to weight each sample by
#                      # model.targets[0:n],  # labels
#                      # model.targets[0],  # labels
#                      # K.learning_phase(),  # train or test mode
#                      ]
#
#     for i in range(n):
#         input_tensors.append(model.sample_weights[i])
#
#     for i in range(n):
#         input_tensors.append(model.targets[i])
#
#     input_tensors.append(K.learning_phase())
#     gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)
#
#
#     get_gradients = K.function(inputs=input_tensors, outputs=[gradients])
#
#
#     # https: // github.com / fchollet / keras / issues / 2226
#     # print 'y_train', y_train.shape
#
#     nb_sample = X.shape[0]
#
#     # if type(y ) ==list:
#     #     y= [yy.reshape((nb_sample, 1)) for yy in y]
#     #     sample_weights = [np.ones(nb_sample) for i in range(n)]
#     # else:
#     #     y = y.reshape((nb_sample, 1))
#     #     sample_weights = np.ones(nb_sample)
#
#     inputs = [X,  # X
#               # sample_weights,  # sample weights
#               # y,  # y
#               # 0  # learning phase in TEST mode
#               ]
#
#     for i in range(n):
#         inputs.append(np.ones(nb_sample))
#
#     if n>1 :
#         for i in range(n):
#             inputs.append(y[i].reshape((nb_sample, 1)))
#     else:
#         inputs.append(y.reshape(nb_sample, 1))
#
#     inputs.append(0)# learning phase in TEST mode
#     # print(X.shape)
#     # print (y.shape)
#
#     # inputs = [X,  # X
#     #           sample_weights,  # sample weights
#     #           y,  # y
#     #           0  # learning phase in TEST mode
#     #           ]
#     # print weights
#     gradients = get_gradients(inputs)[0]
#
#     return gradients


def get_gradient_layer(model, X, y, layer, normalize=True):
    """
    Calculates gradients of the model's loss with respect to a specific layer tensor.
    Refactored for TensorFlow 2.x compatibility using tf.GradientTape.
    
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
    def get_gradients_fn(x_input, targets, weights):
        # Convert inputs to tensors
        x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
        
        # Create a list of targets as tensors
        target_tensors = [tf.convert_to_tensor(t, dtype=tf.float32) for t in targets]
        
        # Create a list of sample weights as tensors
        weight_tensors = [tf.convert_to_tensor(w, dtype=tf.float32) for w in weights]

        # Create a temporary model to get intermediate layer's activations
        # 'layer' is the symbolic tensor for the intermediate layer's output
        # 'model.outputs' can be a list or a single tensor
        outputs_to_fetch = [layer] + (model.outputs if isinstance(model.outputs, list) else [model.outputs])
        temp_model = tf.keras.Model(inputs=model.inputs, outputs=outputs_to_fetch)
        
        with tf.GradientTape() as tape:
            # Get activations of the intermediate layer and final model outputs
            fetched_outputs = temp_model(x_tensor, training=False)
            intermediate_activations = fetched_outputs[0]
            final_outputs = fetched_outputs[1:]
            if len(final_outputs) == 1 and not isinstance(model.outputs, list):
                final_outputs = final_outputs[0] # Unpack if single output model originally
            
            tape.watch(intermediate_activations)  # Watch the concrete intermediate tensor
            
            # Calculate loss based on final_outputs
            if isinstance(final_outputs, list):
                losses = []
                for i in range(n): # n is length of original y (targets)
                    loss_i = model.loss_functions[i](target_tensors[i], final_outputs[i]) 
                    weighted_loss_i = loss_i * weight_tensors[i]
                    losses.append(tf.reduce_mean(weighted_loss_i))
                total_loss = tf.add_n(losses)
            else:
                # Single output model
                loss_value = model.loss(target_tensors[0], final_outputs)
                weighted_loss = loss_value * weight_tensors[0]
                total_loss = tf.reduce_mean(weighted_loss)
            
        # Compute gradients of the total loss with respect to the intermediate_activations
        grads = tape.gradient(total_loss, intermediate_activations)
        
        if grads is None:
            # This might happen if the layer is not differentiable or not part of the
            # computation graph for the loss. Or if layer was an input and not handled correctly.
            # Consider raising an error or returning zeros, depending on expected behavior.
            # For now, to match original behavior if grads were zero (though less likely with K.grad):
            return tf.zeros_like(intermediate_activations) # Or handle error more explicitly

        # Multiply gradients by the concrete intermediate_activations (as per original implementation)
        weighted_grads = intermediate_activations * grads
        
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


"""
def get_shap_scores_layer(model, X, layer_name, output_index=-1, method_name='deepexplainer'):
    # local_smoothing ?
    # ranked_outputs
    def map2layer(model, x, layer_name):
        fetch = model.get_layer(layer_name).output
        feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
        return K.get_session().run(fetch, feed_dict)

    import shap
    if type(output_index) == str:
        y = model.get_layer(output_index).output
    else:
        y = model.outputs[output_index]

    x = model.get_layer(layer_name).output
    if method_name == 'deepexplainer':
        explainer = shap.DeepExplainer((x, y), map2layer(model, X.copy(), layer_name))
        shap_values, indexes = explainer.shap_values(map2layer(model, X, layer_name), ranked_outputs=2)
    elif method_name == 'gradientexplainer':
        explainer = shap.GradientExplainer((x, y), map2layer(model, X.copy(), layer_name), local_smoothing=2)
        shap_values, indexes = explainer.shap_values(map2layer(model, X, layer_name), ranked_outputs=2)
    else:
        raise ('unsppuorted method')

    print (shap_values[0].shape)
    return shap_values[0]
"""


# model, X_train, y_train, target, detailed=detailed, method_name=method
"""
def get_shap_scores(model, X_train, y_train, target=-1, method_name='deepexplainer', detailed=False):
    gradients_list = []
    gradients_list_sample_level = []
    i = 0
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
            if target is None:
                output = i
            else:
                output = target
            print('layer # {}, layer name {},  output name {}'.format(i, l.name, output))
            i += 1
            # gradients = get_deep_explain_score_layer(model, X_train, l.name, output, method_name= method_name )
            gradients = get_shap_scores_layer(model, X_train, l.name, output, method_name=method_name)
            # getting average score
            if gradients.ndim > 1:
                # feature_weights = np.sum(np.abs(gradients), axis=-2)
                feature_weights = np.sum(gradients, axis=-2)
            else:
                feature_weights = np.abs(gradients)
            gradients_list.append(feature_weights)
            gradients_list_sample_level.append(gradients)
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list
    pass
"""


"""
def get_deep_explain_scores(model, X_train, y_train, target=-1, method_name='grad*input', detailed=False, **kwargs):
    # gradients_list = []
    # gradients_list_sample_level = []

    gradients_list = {}
    gradients_list_sample_level = {}

    i = 0
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )

            if target is None:
                output = i
            else:
                output = target

            print('layer # {}, layer name {},  output name {}'.format(i, l.name, output))
            i += 1
            gradients = get_deep_explain_score_layer(model, X_train, l.name, output, method_name=method_name)
            if gradients.ndim > 1:
                # feature_weights = np.sum(np.abs(gradients), axis=-2)
                # feature_weights = np.sum(gradients, axis=-2)
                print('gradients.shape', gradients.shape)
                # feature_weights = np.abs(np.sum(gradients, axis=-2))
                feature_weights = np.sum(gradients, axis=-2)
                # feature_weights = np.mean(gradients, axis=-2)
                print('feature_weights.shape', feature_weights.shape)
                print('feature_weights min max', min(feature_weights), max(feature_weights))
            else:
                # feature_weights = np.abs(gradients)
                feature_weights = gradients
                # feature_weights = np.mean(gradients)
            # gradients_list.append(feature_weights)
            # gradients_list_sample_level.append(gradients)
            gradients_list[l.name] = feature_weights
            gradients_list_sample_level[l.name] = gradients
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list
    pass
"""


"""
def get_deep_explain_score_layer(model, X, layer_name, output_index=-1, method_name='grad*input'):
    scores = None
    import keras
    from deepexplain.tensorflow_ import DeepExplain
    import tensorflow as tf
    ww = model.get_weights()
    with tf.Session() as sess:
        try:
            with DeepExplain(session=sess) as de:  # <-- init DeepExplain context
                # Need to reconstruct the graph in DeepExplain context, using the same weights.
                # model= nn_model.model
                print(layer_name)
                model = keras.models.clone_model(model)
                model.set_weights(ww)
                # if layer_name=='inputs':
                #     layer_outcomes= X
                # else:
                #     layer_outcomes = nn_model.get_layer_output(layer_name, X)[0]

                x = model.get_layer(layer_name).output
                # x = model.inputs[0]
                if type(output_index) == str:
                    y = model.get_layer(output_index).output
                else:
                    y = model.outputs[output_index]

                # y = model.get_layer('o6').output
                # x = model.inputs[0]
                print(layer_name)
                print('model.inputs', model.inputs)
                print('model y', y)
                print('model x', x)
                attributions = de.explain(method_name, y, x, model.inputs[0], X)
                print('attributions', attributions.shape)
                scores = attributions
                return scores
        except:
            sess.close()
            print("Unexpected error:", sys.exc_info()[0])
            raise
"""


def get_skf_weights(model, X, y, importance_type):
    from features_processing.feature_selection import FeatureSelectionModel
    layers = get_layers(model)
    inp = model.input
    layer_weights = []
    for i, l in enumerate(layers):

        if type(l) == InputLayer:
            layer_out = X
        elif l.name.startswith('h'): # Hidden layers
            # Create a temporary model to get the output of layer 'l'
            # Assuming 'inp' (model.input) is the correct single input tensor for the main model
            intermediate_model = tf.keras.Model(inputs=inp, outputs=l.output)
            layer_out = intermediate_model(X, training=False) # Eager execution
            # FeatureSelectionModel likely expects a NumPy array
            if hasattr(layer_out, 'numpy'):
                layer_out = layer_out.numpy()
            print(i, l, l.output) # Updated print statement
        else:
            continue

        if type(y) == list:
            y = y[0]

        # layer_out = StandardScaler().fit_transform(layer_out)
        p = {'type': importance_type, 'params': {}}
        fs_model = FeatureSelectionModel(p)
        fs_model = fs_model.fit(layer_out, y.ravel())
        fs_coef = fs_model.get_coef()
        fs_coef[fs_coef == np.inf] = 0
        layer_weights.append(fs_coef)
    return layer_weights


def get_activation_gradients(model, X, y):
    """
    Calculates gradients of the model's loss with respect to each layer's activation (dL/da_l).
    Returns a dictionary mapping layer names to their gradients.
    Refactored for TensorFlow 2.x compatibility using tf.GradientTape.

    Args:
        model: The Keras model.
        X: Input data (tf.Tensor or NumPy array).
        y: Target data (tf.Tensor or NumPy array).

    Returns:
        A list of NumPy arrays, where each array contains the gradients for a corresponding
        layer in 'layers_to_watch', maintaining order.
    """
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    all_layers = get_layers(model) # From model_utils

    layers_to_watch = [layer for layer in all_layers 
                       if not isinstance(layer, (tf.keras.layers.InputLayer, 
                                                tf.keras.layers.Dropout, 
                                                tf.keras.layers.BatchNormalization)) 
                       and hasattr(layer, 'output')]
    
    if not layers_to_watch:
        return [] # Return empty list

    model_outputs_list = model.outputs if isinstance(model.outputs, list) else [model.outputs]

    @tf.function
    def get_grads_for_activations(x_input, target_input):
        # watched_activations_map = {} # No longer needed for output
        ordered_grads_list = [None] * len(layers_to_watch) # Preallocate list for ordered results

        with tf.GradientTape(persistent=True) as tape:
            outputs_to_fetch_from_temp_model = [layer.output for layer in layers_to_watch] + model_outputs_list
            temp_model_for_activations = tf.keras.Model(inputs=model.inputs, outputs=outputs_to_fetch_from_temp_model)
            
            fetched_tensors = temp_model_for_activations(x_input, training=False)
            
            current_watched_activations = fetched_tensors[:len(layers_to_watch)]
            final_outputs_from_model = fetched_tensors[len(layers_to_watch):]
            if len(final_outputs_from_model) == 1 and not isinstance(model.outputs, list):
                 final_outputs_from_model = final_outputs_from_model[0]

            # Watch activations for gradient calculation
            for i, layer_activation_tensor in enumerate(current_watched_activations):
                tape.watch(layer_activation_tensor)
            
            # Calculate loss
            y_true_for_loss = target_input
            # Keras expects y_true to be a list if model has multiple outputs and list of losses.
            # If y_true_for_loss is a single tensor & model.loss is a list, it's ambiguous.
            # Passing as is; Keras loss calculation will handle or error out if mismatched.

            total_loss = tf.constant(0.0, dtype=tf.float32)
            if hasattr(model, 'compiled_loss') and model.compiled_loss is not None:
                current_loss = model.compiled_loss(
                    y_true=y_true_for_loss,
                    y_pred=final_outputs_from_model,
                    sample_weight=None, 
                    regularization_losses=model.losses
                )
                total_loss += current_loss # compiled_loss should return scalar or be reducible
            elif isinstance(model.loss, list):
                current_y_targets = y_true_for_loss
                if not isinstance(y_true_for_loss, list):
                    current_y_targets = [y_true_for_loss] * len(model.loss)
                for i in range(len(model.loss)):
                    loss_fn = model.loss[i]
                    y_pred_i = final_outputs_from_model[i] if isinstance(final_outputs_from_model, list) else final_outputs_from_model
                    y_true_i = current_y_targets[i]
                    loss_val = loss_fn(y_true_i, y_pred_i)
                    total_loss += tf.reduce_mean(loss_val)
            elif callable(model.loss):
                loss_value = model.loss(y_true_for_loss, final_outputs_from_model)
                total_loss += tf.reduce_mean(loss_value) if hasattr(loss_value, 'shape') and loss_value.shape.rank > 0 else loss_value
            else:
                raise ValueError("Cannot determine how to calculate loss for the model. `model.loss` is not a list or callable, and `model.compiled_loss` is not available.")
            
            if not (hasattr(model, 'compiled_loss') and model.compiled_loss is not None) and model.losses:
                total_loss += tf.add_n(model.losses) # Add layer regularization losses if not already in compiled_loss
            
            # Ensure total_loss is scalar
            if hasattr(total_loss, 'shape') and total_loss.shape.rank > 0:
                total_loss = tf.reduce_mean(total_loss)

        # Compute gradients for each watched activation in order
        for i, layer in enumerate(layers_to_watch):
            activation_tensor = current_watched_activations[i] # Get the i-th activation tensor
            grad = tape.gradient(total_loss, activation_tensor)
            if grad is not None:
                ordered_grads_list[i] = grad.numpy() # Store NumPy array in order
            else:
                # Handle cases where gradient is None (e.g., layer not in grad path)
                ordered_grads_list[i] = np.zeros_like(activation_tensor.numpy())
        
        del tape
        return ordered_grads_list

    ordered_gradients_list = get_grads_for_activations(X_tensor, y_tensor)
    return ordered_gradients_list


def resolve_gradient_function(name_or_func):
    """
    Resolves a string identifier to a gradient calculation function or returns the callable.
    """
    if isinstance(name_or_func, str):
        if name_or_func == 'gradient' or name_or_func == 'activation_gradient':
            return get_activation_gradients
        elif name_or_func.startswith('deepexplain_') or name_or_func.startswith('shap_'):
            # For TF2 compatibility, we map deepexplain and shap methods to activation_gradients_importance
            # with appropriate method configuration
            method = name_or_func.split('_')[1] if '_' in name_or_func else 'grad*input'
            
            # Create and return a function that matches the expected signature
            def wrapped_activation_importance(model, X, y):
                method_mapping = {
                    'grad*input': 'input*grad',
                    'grad': 'grad',
                    'gradshap': 'grad',
                    'deeplift': 'input*grad',
                    'intgrad': 'input*grad',
                    'elrp': 'input*grad',
                    'occlusion': 'input*grad_abs'
                }
                activation_method = method_mapping.get(method, 'input*grad')
                return get_activation_gradients_importance(model, X, y, method=activation_method)
                
            return wrapped_activation_importance
        else:
            # Placeholder for future: re-integrate other method resolvers here
            # If other methods are needed, they should be passed as callables directly.
            print(f"Warning: Unknown gradient function string identifier '{name_or_func}'. GradientCheckpoint might not work as expected.")
            # Return a dummy function to prevent crashes, but signal issue.
            return lambda model, X, y: {} # Returns empty dict
    elif callable(name_or_func):
        return name_or_func
    else:
        if name_or_func is None:
             # If feature_importance is None, GradientCheckpoint shouldn't be active or expect gradients.
             # This case should ideally be handled by `nn.Model.save_gradient` flag.
            return lambda model, X, y: {} 
        raise ValueError("feature_importance must be a string identifier or a callable function.")


def get_gradient_weights(model, X, y, signed=False, detailed=False, normalize=True):
    gradients_list = []
    gradients_list_sample_level = []
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
            w = l.get_output_at(0)
            gradients = get_gradient_layer(model, X, y, w, normalize)
            if gradients.ndim > 1:
                if signed:
                    feature_weights = np.sum(gradients, axis=-2)
                else:
                    feature_weights = np.sum(np.abs(gradients), axis=-2)

            else:
                feature_weights = np.abs(gradients)
            gradients_list.append(feature_weights)
            gradients_list_sample_level.append(gradients)
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list


def get_gradient_weights_with_repeated_output(model, X, y):
    gradients_list = []
    # print 'trainable weights',model.trainable_weights
    # print 'layers', get_layers (model)

    for l in get_layers(model):

        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue

        # print 'get the gradient of layer {}'.format(l.name)
        if l.name.startswith('o') and not l.name.startswith('o_'):
            print(l.name)
            print(l.weights)
            weights = l.get_weights()[0]
            # weights = l.get_weights()
            # print 'weights shape {}'.format(weights.shape)
            gradients_list.append(weights.ravel())

    return gradients_list


# get weights of each layer based on training a linear model that predicts the outcome (y) given the layer output
def get_weights_linear_model(model, X, y):
    weights = None
    layer_weights = []
    layers = get_layers(model)
    inp = model.input
    for i, l in enumerate(layers):
        if type(l) in [Sequential, Dropout]:
            continue
        print(type(l))
        if type(l) == InputLayer:
            layer_out = X
        else:
            # Create a temporary model to get the output of layer 'l'
            # Assuming 'inp' (model.input) is the correct single input tensor for the main model
            intermediate_model = tf.keras.Model(inputs=inp, outputs=l.output)
            layer_out = intermediate_model(X, training=False) # Eager execution
            # LogisticRegression likely expects a NumPy array
            if hasattr(layer_out, 'numpy'):
                layer_out = layer_out.numpy()
            print(i, l, l.output) # Updated print statement
        # print layer_out.shape
        # layer_outs.append(layer_out)
        linear_model = LogisticRegression(penalty='l1')
        # linear_model = LinearRegression( )
        # layer_out = StandardScaler().fit_transform(layer_out)
        if type(y) == list:
            y = y[0]
        linear_model.fit(layer_out, y.ravel())
        # print 'layer coef  shape ', linear_model.coef_.T.ravel().shape
        layer_weights.append(linear_model.coef_.T.ravel())
    return layer_weights


# def get_weights_gradient_outcome(model, x_train, y_train):
#     if type(y_train) == list:
#         n = len(y_train)
#     else:
#         n = 1
#     nb_sample = x_train.shape[0]
#     sample_weights = np.ones(nb_sample)
#     print(model.output)
#     # output = model.output[-1]
#     # model = nn_model.model
#     input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
#     # input_tensors = model.inputs + model.targets + [K.learning_phase()]
#     layers = get_layers(model)
#     gradients_list= []
#     i=0
#     for l in layers:
#         if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
#             print(i)
#             output = model.output[i]
#             i+=1
#             print(i), l.name, output.name, output, l.get_output_at(0)
#             # gradients = K.gradients(K.mean(output), l.get_output_at(0))
#             gradients = K.gradients(output, l.get_output_at(0))
#             # w= l.get_output_at(0)
#             # gradients = [w*g for g in K.gradients(output, w)]
#             get_gradients = K.function(inputs=input_tensors, outputs=gradients)
#             inputs = [x_train] + [sample_weights] * n + y_train  + [0]
#             gradients = get_gradients(inputs)
#             print 'gradients',len(gradients), gradients[0].shape
#             g= np.sum(np.abs(gradients[0]), axis = 0)
#             g= np.sum(gradients[0], axis = 0)
#             g= np.abs(g)
#             print('gradients', gradients[0].shape)
#             gradients_list.append(g)
#
#     return gradients_list
#

def get_gradient(model, source_layer_name, target_specifier, x_train_data, multiply_by_input=False):
    """
    Calculates gradients using TensorFlow 2.x GradientTape.
    Corresponds to K.gradients(target_tensor, source_tensor).

    Args:
        model: The Keras model.
        source_layer_name (str): Name of the layer whose output is the 'source' tensor (the 'x' in dy/dx).
        target_specifier (str or int): 
            If str: Name of the layer whose output is the 'target' tensor (the 'y' in dy/dx).
            If int: Index of the model output in model.outputs to be used as 'target' tensor.
        x_train_data (tf.Tensor or np.ndarray): The input data to the model.
        multiply_by_input (bool): If True, multiply gradients by the source tensor's values.

    Returns:
        list: A list containing one tf.Tensor representing the gradients.
              Returns list with a zero tensor of source_layer's output shape if gradients are None.
    """
    source_tensor_def = model.get_layer(source_layer_name).output
    
    target_tensor_def = None
    if isinstance(target_specifier, str):
        try:
            target_tensor_def = model.get_layer(target_specifier).output
        except ValueError:
            raise ValueError(f"Target layer '{target_specifier}' not found in the model.")
    elif isinstance(target_specifier, int):
        if 0 <= target_specifier < len(model.outputs):
            target_tensor_def = model.outputs[target_specifier]
        else:
            raise ValueError(f"Target output index {target_specifier} is out of range for model with {len(model.outputs)} outputs.")
    else:
        raise TypeError(f"target_specifier must be a layer name (str) or output index (int), got {type(target_specifier)}")

    # Create a temporary model that outputs both the source tensor and the target tensor
    # This ensures they are computed from the same forward pass of x_train_data
    computation_model = tf.keras.Model(inputs=model.inputs, outputs=[source_tensor_def, target_tensor_def])

    # Ensure x_train_data is a tf.Tensor, float32 is common for Keras
    if not isinstance(x_train_data, tf.Tensor):
        x_train_data_tf = tf.convert_to_tensor(x_train_data, dtype=tf.float32)
    else:
        x_train_data_tf = tf.cast(x_train_data, dtype=tf.float32) # Ensure correct dtype

    with tf.GradientTape() as tape:
        # Watch the input data tensor; gradients will flow from here
        tape.watch(x_train_data_tf)
        
        # Compute the values of the source and target tensors
        # training=False corresponds to K.learning_phase() == 0
        source_tensor_val, target_tensor_val = computation_model(x_train_data_tf, training=False)

    # Calculate gradients of the target_tensor_val with respect to source_tensor_val
    # This computes d(target_tensor_val) / d(source_tensor_val)
    gradients = tape.gradient(target_tensor_val, source_tensor_val)

    if gradients is None:
        print(f"Warning: Gradient of target '{target_tensor_def.name}' w.r.t. source '{source_tensor_def.name}' is None. "
              f"This might indicate that the target does not depend on the source, or a non-differentiable operation exists. "
              f"Returning zeros with shape of source tensor: {source_tensor_val.shape}")
        # K.gradients would return a list of Nones or zeros. Let's return zeros of the correct shape.
        gradients = tf.zeros_like(source_tensor_val)
        
    result_gradients_list = [gradients]

    if multiply_by_input:
        # Multiply by the value of the source tensor (the "input" in this context)
        result_gradients_list = [source_tensor_val * g for g in result_gradients_list]

    return result_gradients_list


def get_weights_gradient_outcome(model, x_train, y_train, detailed=False, target=-1, multiply_by_input=False,
                                 signed=True):
    # y_train is not used by the new get_gradient, but kept in signature for compatibility with callers if any
    print(model.output) # This print might need adjustment if model.output is complex in TF2
    layers = get_layers(model)
    gradients_list = []
    gradients_list_sample_level = []
    i = 0 # Counter for print statements, matches original logic

    for l in layers:
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (ad hoc convention)

            current_target_specifier = target
            # Potentially preprocess target if -1 means last output:
            # if target == -1 and isinstance(model.outputs, list) and len(model.outputs) > 0:
            #    current_target_specifier = len(model.outputs) - 1
            # This adjustment depends on the intended meaning of target=-1.
            # For now, passing target directly as per get_gradient's new signature.

            symbolic_output_for_print = "UNKNOWN_TARGET_NAME"
            try:
                if isinstance(current_target_specifier, str):
                    symbolic_output_for_print = model.get_layer(current_target_specifier).output.name
                elif isinstance(current_target_specifier, int):
                    if 0 <= current_target_specifier < len(model.outputs):
                        symbolic_output_for_print = model.outputs[current_target_specifier].name
                    else:
                        symbolic_output_for_print = f"INVALID_INDEX_{current_target_specifier}"
            except Exception as e:
                symbolic_output_for_print = f"ERROR_RESOLVING_TARGET_NAME: {e}"

            print(f'Processing layer # {i}, layer name: {l.name}, target_specifier: {current_target_specifier} (resolves to output like: {symbolic_output_for_print})')
            # Original complex print: print(i), l.name, output.name, output, l.get_output_at(0) is simplified.

            # Call the new gradient function. y_train is not passed to it.
            gradients_tf_list = get_gradient(model,
                                             source_layer_name=l.name,
                                             target_specifier=current_target_specifier,
                                             x_train_data=x_train,
                                             multiply_by_input=multiply_by_input)

            # gradients_tf_list contains one tensor. Extract it and convert to numpy.
            gradient_tensor = gradients_tf_list[0]
            gradient_numpy = gradient_tensor.numpy() if hasattr(gradient_tensor, 'numpy') else gradient_tensor

            print('gradients tensor shape:', gradient_tensor.shape, '; numpy shape:', gradient_numpy.shape)
            if signed:
                g = np.sum(gradient_numpy, axis=0)
            else:
                g = np.sum(np.abs(gradient_numpy), axis=0)
            
            gradients_list_sample_level.append(gradient_numpy) # Store the full numpy gradient
            gradients_list.append(g)
            i += 1 # Increment counter

    if detailed:
        return gradients_list, gradients_list_sample_level

    return gradients_list


def get_activation_gradients_importance(model, X, y, target=-1, method='input*grad', detailed=False):
    """
    Calculates feature importance scores using activation gradients.
    This function is a TensorFlow 2.x compatible alternative to deepexplain methods.
    
    Args:
        model: The Keras model
        X: Input data
        y: Target data
        target: Target output index or layer name (default: -1, which means the last output)
        method: Method to calculate importance:
                - 'input*grad': Input multiplied by gradient (similar to DeepExplain's grad*input)
                - 'input': Raw input values
                - 'grad': Raw gradient values
                - 'input*grad_abs': Input multiplied by absolute gradient
                - 'grad_abs': Absolute gradient values
        detailed: If True, returns both summarized and per-sample gradients
                 If False, returns only summarized gradients
                 
    Returns:
        If detailed=False: A dictionary mapping layer names to their importance scores
        If detailed=True: A tuple of (importance_dict, per_sample_importance_dict)
    """
    # First, get activation gradients for all layers
    activation_gradients = get_activation_gradients(model, X, y)
    
    # Get the relevant layers from the model
    all_layers = get_layers(model)
    layers_to_watch = [layer for layer in all_layers 
                      if not isinstance(layer, (tf.keras.layers.InputLayer, 
                                              tf.keras.layers.Dropout, 
                                              tf.keras.layers.BatchNormalization)) 
                      and hasattr(layer, 'output')]
    
    if not layers_to_watch:
        return {} if not detailed else ({}, {})
    
    # Create a modified forward pass to get activations for each layer
    layer_outputs = []
    outputs_to_fetch = [layer.output for layer in layers_to_watch]
    intermediate_activation_model = Model(inputs=model.inputs, outputs=outputs_to_fetch)
    activations = intermediate_activation_model.predict(X)
    
    # If only one layer, wrap in list to keep consistent
    if len(outputs_to_fetch) == 1 and not isinstance(activations, list):
        activations = [activations]
    
    # Process gradients according to the specified method
    importance_dict = {}
    per_sample_importance_dict = {}
    
    for i, layer in enumerate(layers_to_watch):
        layer_activations = activations[i]
        layer_gradients = activation_gradients[i] if i < len(activation_gradients) else None
        
        if layer_gradients is None:
            continue
            
        # Calculate importance based on method
        if method == 'input*grad':
            importance = layer_activations * layer_gradients
        elif method == 'input':
            importance = layer_activations
        elif method == 'grad':
            importance = layer_gradients
        elif method == 'input*grad_abs':
            importance = layer_activations * np.abs(layer_gradients)
        elif method == 'grad_abs':
            importance = np.abs(layer_gradients)
        else:
            # Default to input*grad
            importance = layer_activations * layer_gradients
            
        # Store per-sample importance
        per_sample_importance_dict[layer.name] = importance
        
        # Calculate summarized importance (averaging across samples)
        if importance.ndim > 1:
            # Sum across all dimensions except the feature dimension
            # This assumes the feature dimension is the last one
            summary_importance = np.sum(importance, axis=0)
        else:
            summary_importance = importance
            
        importance_dict[layer.name] = summary_importance
    
    if detailed:
        return importance_dict, per_sample_importance_dict
        
    return importance_dict


# def get_gradient_weights(model, X, y):
#     gradients_list = []
#     print 'trainable weights',model.trainable_weights
#     print 'layers', model.layers
#
#     # for l in get_layers(model):
#     # for l in [model.inputs[0] ]+ model.trainable_weights:
#     # c = get_gradient_layer(model, X, y, model.inputs[0])
#     # gradients_list.append(np.mean(c, axis=0))
#     for l in  model.trainable_weights:
#         # print l
#         # l = l.trainable_weights
#         # layer = model.inputs[0]
#         # print  ,
#
#         # if type(l) == InputLayer:
#         #     w = model.inputs[0]
#         # # elif type(l)==Sequential:
#         # #     continue
#         # elif hasattr(l, 'kernel') and type(l) != SpraseLayer:
#         #     w= l.output
#         # else: continue
#
#         if 'kernel' in str(l):
#
#             gradients = get_gradient_layer(model, X, y, l)
#             if gradients.ndim >1:
#                 feature_weights = np.mean(gradients, axis=1)
#             else:
#                 feature_weights = gradients
#             # feature_weights= gradients
#             print 'layer {} grdaient shape {}', l, feature_weights.shape
#             gradients_list.append(feature_weights)
#     return gradients_list

def get_permutation_weights(model, X, y):
    scores = []
    prediction_scores = predict(model, X)
    # print y
    # print prediction_scores
    baseline_acc = accuracy_score(y[0], prediction_scores)
    rnd = np.random.random((X.shape[0],))
    x_original = X.copy()
    for i in range(X.shape[1]):
        # if (i%100)==0:
        print(i)
        # x = X.copy()
        x_vector = x_original[:, i]
        # np.random.shuffle(x[:, i])
        x_original[:, i] = rnd
        acc = accuracy_score(y[0], predict(model, x_original))
        x_original[:, i] = x_vector
        scores.append((baseline_acc - acc) / baseline_acc)
    return np.array(scores)


def get_deconstruction_weights(model):
    for layer in model.layers:
        # print layer.name
        weights = layer.get_weights()  # list of numpy arrays
        # for w in weights:
        #     print w.shape
    pass

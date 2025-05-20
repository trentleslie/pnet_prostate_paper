"""
Refactored version of the build_pnet2 function for TensorFlow 2.x compatibility.
This is the refactored implementation of the function from prostate_models.py.
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Lambda, Concatenate
from tensorflow.keras.regularizers import l2

from data.data_access import Data
from model.builders.builders_utils import get_pnet  # This function also needs TF2.x refactoring
from model.layers_custom import f1  # Ensure this custom metric is TF2.x compatible


def build_pnet2(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0, dropout=0.5,
                use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None, n_hidden_layers=1,
                direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False,
                attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True, sparse_first_layer=True):
    """
    Builds a Pathway Network (P-NET) model using TensorFlow 2.x Keras.
    
    Args:
        optimizer: Keras optimizer for model compilation
        w_reg: Weight regularization factor
        w_reg_outcomes: Weight regularization factor for outcome layers
        add_unk_genes: Whether to add unknown genes node
        sparse: Whether to use sparse layers
        loss_weights: Weights for different outputs in loss calculation
        dropout: Dropout rate
        use_bias: Whether to use bias in layers
        activation: Activation function
        loss: Loss function
        data_params: Parameters for data access
        n_hidden_layers: Number of hidden layers
        direction: Direction of pathway connections (root_to_leaf or leaf_to_root)
        batch_normal: Whether to use batch normalization
        kernel_initializer: Initialization for kernel weights
        shuffle_genes: Whether to shuffle gene connections
        attention: Whether to use attention mechanism
        dropout_testing: Whether to apply dropout during testing
        non_neg: Whether to enforce non-negative constraints
        repeated_outcomes: Whether to use outcomes from all layers
        sparse_first_layer: Whether to use sparse layer for first layer
        
    Returns:
        model: Compiled Keras model
        feature_names: Dictionary mapping layer names to features
    """
    print(data_params)
    print(f'n_hidden_layers: {n_hidden_layers}')
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print(x.shape)
    print(y.shape)
    print(info.shape)
    print(cols.shape)
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info(f'x shape {x.shape}, y shape {y.shape}, info {info.shape}, genes {cols.shape}')

    n_features = x.shape[1]

    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')

    # Note: get_pnet function would also need to be refactored for TF2.x
    outcome, decision_outcomes, feature_n = get_pnet(ins,
                                                     features=features,
                                                     genes=genes,
                                                     n_hidden_layers=n_hidden_layers,
                                                     direction=direction,
                                                     activation=activation,
                                                     activation_decision=activation_decision,
                                                     w_reg=w_reg,
                                                     w_reg_outcomes=w_reg_outcomes,
                                                     dropout=dropout,
                                                     sparse=sparse,
                                                     add_unk_genes=add_unk_genes,
                                                     batch_normal=batch_normal,
                                                     sparse_first_layer=sparse_first_layer,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg
                                                     )

    feature_names = feature_n
    feature_names['inputs'] = cols

    print('Compiling...')

    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        outcome = decision_outcomes[-1]

    # Updated model creation syntax for TF2.x
    model = Model(inputs=ins, outputs=outcome)

    if isinstance(outcome, list):
        n_outputs = len(outcome)
    else:
        n_outputs = 1

    if isinstance(loss_weights, list):
        loss_weights = loss_weights
    else:
        loss_weights = [loss_weights] * n_outputs

    print(f'loss_weights: {loss_weights}')
    
    # For multi-output models, consider using dictionary format for loss and metrics
    # if n_outputs > 1 and isinstance(outcome, list):
    #     losses = {output.name: 'binary_crossentropy' for output in outcome}
    #     metrics = {output.name: f1 for output in outcome}
    #     model.compile(optimizer=optimizer, loss=losses, metrics=metrics, loss_weights=loss_weights)
    # else:
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, 
                  metrics=[f1], 
                  loss_weights=loss_weights)

    logging.info('done compiling')

    # Helper functions for model inspection
    from model.model_utils import print_model, get_layers
    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info(f'# of trainable params of the model is {model.count_params()}')
    
    return model, feature_names
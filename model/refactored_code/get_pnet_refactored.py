"""
Refactored version of the get_pnet function for TensorFlow 2.x compatibility.
This is the refactored implementation of the function from builders_utils.py.
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Multiply
from tensorflow.keras.regularizers import l2

from data.pathways.reactome import ReactomeNetwork
from model.layers_custom import Diagonal, SparseTF  # These custom layers would also need refactoring


def get_layer_maps(genes, n_levels, direction, add_unk_genes):
    """
    Get pathway maps for network construction.
    This function should be refactored separately as needed.
    """
    # Existing implementation, refactored for Python 3 print statements
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print(f'layer # {i}')
        # Rest of the implementation...
        # This would need print statement updates and any other Python 3 fixes
    return maps


def shuffle_genes_map(mapp):
    """
    Shuffle gene map for randomization.
    This function should be refactored separately as needed.
    """
    # Existing implementation, refactored for Python 3 print statements
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    logging.info(f'ones_ratio {ones_ratio}')
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info(f'random map ones_ratio {ones_ratio}')
    return mapp


def get_pnet(inputs, features, genes, n_hidden_layers, direction, activation, activation_decision, w_reg,
             w_reg_outcomes, dropout, sparse, add_unk_genes, batch_normal, kernel_initializer, use_bias=False,
             shuffle_genes=False, attention=False, dropout_testing=False, non_neg=False, sparse_first_layer=True):
    """
    Create a Pathway Network (P-NET) model using TensorFlow 2.x Keras.
    
    Args:
        inputs: Input tensor
        features: Features used in the model
        genes: Gene list
        n_hidden_layers: Number of hidden layers
        direction: Direction of pathway connections
        activation: Activation function
        activation_decision: Activation function for decision layers
        w_reg: Weight regularization factor
        w_reg_outcomes: Weight regularization for outcomes
        dropout: Dropout rate
        sparse: Whether to use sparse layers
        add_unk_genes: Whether to add unknown genes
        batch_normal: Whether to use batch normalization
        kernel_initializer: Initializer for weights
        use_bias: Whether to use bias
        shuffle_genes: Whether to shuffle gene connections
        attention: Whether to use attention mechanism
        dropout_testing: Whether to apply dropout during testing
        non_neg: Whether to enforce non-negative constraints
        sparse_first_layer: Whether to use sparse first layer
        
    Returns:
        outcome: Output tensor
        decision_outcomes: List of decision outputs
        feature_names: Dictionary of feature names
    """
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)

    if not isinstance(w_reg, list):
        w_reg = [w_reg] * 10

    if not isinstance(w_reg_outcomes, list):
        w_reg_outcomes = [w_reg_outcomes] * 10

    if not isinstance(dropout, list):
        dropout = [w_reg_outcomes] * 10

    w_reg0 = w_reg[0]
    w_reg_outcome0 = w_reg_outcomes[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    reg_l = l2
    constraints = {}
    if non_neg:
        from tensorflow.keras.constraints import non_neg  # Updated import
        constraints = {'kernel_constraint': non_neg()}  # Updated constraint name

    if sparse:
        if shuffle_genes == 'all':
            ones_ratio = float(n_features) / np.prod([n_genes, n_features])
            logging.info(f'ones_ratio random {ones_ratio}')
            mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
            layer1 = SparseTF(n_genes, mapp, activation=activation, kernel_regularizer=reg_l(w_reg0),  # Updated param name
                            name=f'h{0}', kernel_initializer=kernel_initializer, use_bias=use_bias,
                            **constraints)
        else:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, kernel_regularizer=l2(w_reg0),  # Updated param name
                            use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)
    else:
        if sparse_first_layer:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, kernel_regularizer=l2(w_reg0),  # Updated param name
                            use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)
        else:
            layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, kernel_regularizer=l2(w_reg0),  # Updated param name
                        use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)
    
    outcome = layer1(inputs)
    
    if attention:
        attention_probs = Diagonal(n_genes, input_shape=(n_features,), activation='sigmoid', kernel_regularizer=l2(w_reg0),  # Updated param name
                                name='attention0')(inputs)
        outcome = Multiply(name='attention_mul')([outcome, attention_probs])  # Updated merge operation

    decision_outcomes = []

    # Updated to use kernel_regularizer
    decision_outcome = Dense(1, activation='linear', name=f'o_linear{0}', kernel_regularizer=reg_l(w_reg_outcome0))(
        inputs)

    # Testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    # Updated to use kernel_regularizer
    decision_outcome = Dense(1, activation='linear', name=f'o_linear{1}',
                            kernel_regularizer=reg_l(w_reg_outcome1 / 2.))(outcome)

    # Using the training parameter in Dropout for compatibility with TF2.x
    drop2 = Dropout(dropout[0], name=f'dropout_{0}')
    outcome = drop2(outcome, training=dropout_testing)

    # Testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    decision_outcome = Activation(activation=activation_decision, name=f'o{1}')(decision_outcome)
    decision_outcomes.append(decision_outcome)

    if n_hidden_layers > 0:
        maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
        layer_inds = range(1, len(maps))
        
        print(f'original dropout {dropout}')
        print(f'dropout {layer_inds}, {dropout}, {w_reg}')
        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]
        
        for i, mapp in enumerate(maps[0:-1]):
            w_reg = w_regs[i]
            w_reg_outcome = w_reg_outcomes[i]
            dropout = dropouts[1]
            names = mapp.index
            mapp = mapp.values
            
            if shuffle_genes in ['all', 'pathways']:
                mapp = shuffle_genes_map(mapp)
            
            n_genes, n_pathways = mapp.shape
            logging.info(f'n_genes, n_pathways {n_genes} {n_pathways}')
            print(f'layer {i}, dropout {dropout} w_reg {w_reg}')
            layer_name = f'h{i + 1}'
            
            if sparse:
                hidden_layer = SparseTF(n_pathways, mapp, activation=activation, kernel_regularizer=reg_l(w_reg),  # Updated param name
                                    name=layer_name, kernel_initializer=kernel_initializer,
                                    use_bias=use_bias, **constraints)
            else:
                hidden_layer = Dense(n_pathways, activation=activation, kernel_regularizer=reg_l(w_reg),  # Updated param name
                                    name=layer_name, kernel_initializer=kernel_initializer, **constraints)

            outcome = hidden_layer(outcome)

            if attention:
                attention_probs = Dense(n_pathways, activation='sigmoid', name=f'attention{i + 1}',
                                    kernel_regularizer=l2(w_reg))(outcome)  # Updated param name
                outcome = Multiply(name=f'attention_mul{i + 1}')([outcome, attention_probs])  # Updated merge operation

            # Updated to use kernel_regularizer
            decision_outcome = Dense(1, activation='linear', name=f'o_linear{i + 2}',
                                    kernel_regularizer=reg_l(w_reg_outcome))(outcome)
                                    
            # Testing
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
                
            decision_outcome = Activation(activation=activation_decision, name=f'o{i + 2}')(decision_outcome)
            decision_outcomes.append(decision_outcome)
            
            drop2 = Dropout(dropout, name=f'dropout_{i + 1}')
            outcome = drop2(outcome, training=dropout_testing)

            feature_names[f'h{i}'] = names
            
        i = len(maps)
        feature_names[f'h{i - 1}'] = maps[-1].index
        
    return outcome, decision_outcomes, feature_names
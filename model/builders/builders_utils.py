# from data.pathways.pathway_loader import get_pathway_files
import itertools
import logging

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply
from tensorflow.keras.regularizers import l2

# from data.pathways.pathway_loader import get_pathway_files
from data.pathways.reactome import ReactomeNetwork
# Import TF2.x versions of custom layers
try:
    from model.layers_custom_tf2 import Diagonal, SparseTF
except ImportError:
    # Fall back to the original versions if TF2.x versions aren't available
    from model.layers_custom import Diagonal, SparseTF


def get_map_from_layer(layer_dict):
    pathways = list(layer_dict.keys()) # Ensure pathways is a list for pathways.index(p)
    print(f'pathways: {len(pathways)}')
    genes = list(itertools.chain.from_iterable(layer_dict.values()))
    genes = list(np.unique(genes))
    print(f'genes: {len(genes)}')

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs if g in genes] # Added check if g in genes
        p_ind = pathways.index(p)
        if g_inds: # Ensure g_inds is not empty before assignment
            mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    # for k, v in layer_dict.items():
    #     print k, v
    #     df.loc[k,v] = 1
    # df= df.fillna(0)
    return df.T


def get_layer_maps(genes, n_levels, direction, add_unk_genes):
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print(f'layer #: {i}') # Lint fix ID: dd257e51-cbee-4e92-a65c-2ae70dcd3831
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        print(f'filtered_map (filter_df) shape before merge: {filter_df.shape}')
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print(f'filtered_map shape after merge: {filtered_map.shape}')
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

        # UNK, add a node for genes without known reactome annotation
        if add_unk_genes:
            print("UNK ") # Standardized print
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, 'UNK'] = 1
        ####

        filtered_map = filtered_map.fillna(0)
        print(f'filtered_map shape after fillna: {filtered_map.shape}')
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum())) # This logging is already Py3
        maps.append(filtered_map)
    return maps


def shuffle_genes_map(mapp):
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    # logging.info('shuffling the map')
    # mapp = mapp.T
    # np.random.shuffle(mapp)
    # mapp= mapp.T
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    logging.info('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info('random map ones_ratio {}'.format(ones_ratio))
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
    # Import at function level for clarity
    from tensorflow.keras.constraints import NonNeg
    
    # Initialize data structures
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)
    
    # Ensure configuration parameters are in list format
    if not isinstance(w_reg, list):
        w_reg = [w_reg] * 10

    if not isinstance(w_reg_outcomes, list):
        w_reg_outcomes = [w_reg_outcomes] * 10

    if not isinstance(dropout, list):
        # Fixed bug from original: use dropout values instead of w_reg_outcomes
        dropout = [dropout] * 10 

    # Extract configuration for first layer
    w_reg0 = w_reg[0]
    w_reg_outcome0 = w_reg_outcomes[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    
    # Regularization and constraints
    reg_l = l2
    constraints = {}
    if non_neg:
        constraints = {'kernel_constraint': NonNeg()}

    # First layer definition - different options based on configuration
    if sparse:
        if shuffle_genes == 'all':
            # Create random connectivity matrix
            ones_ratio = float(n_features) / np.prod([n_genes, n_features])
            logging.info(f'ones_ratio random {ones_ratio}')
            mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
            
            # Create SparseTF layer with TF2.x parameter names
            layer1 = SparseTF(n_genes, mapp, activation=activation, 
                              kernel_regularizer=reg_l(w_reg0),  # Updated from W_regularizer
                              name='h0', 
                              kernel_initializer=kernel_initializer, 
                              use_bias=use_bias,
                              **constraints)
        else:
            # Create Diagonal layer with TF2.x parameter names
            layer1 = Diagonal(n_genes, 
                              input_shape=(n_features,), 
                              activation=activation, 
                              kernel_regularizer=l2(w_reg0),  # Updated from W_regularizer
                              use_bias=use_bias, 
                              name='h0', 
                              kernel_initializer=kernel_initializer, 
                              **constraints)
    else:
        if sparse_first_layer:
            # Create Diagonal layer with TF2.x parameter names
            layer1 = Diagonal(n_genes, 
                              input_shape=(n_features,), 
                              activation=activation, 
                              kernel_regularizer=l2(w_reg0),  # Updated from W_regularizer
                              use_bias=use_bias, 
                              name='h0', 
                              kernel_initializer=kernel_initializer, 
                              **constraints)
        else:
            # Create Dense layer with TF2.x parameter names
            layer1 = Dense(n_genes, 
                           input_shape=(n_features,), 
                           activation=activation, 
                           kernel_regularizer=l2(w_reg0),  # Updated from W_regularizer
                           use_bias=use_bias, 
                           name='h0', 
                           kernel_initializer=kernel_initializer)
    
    # Apply first layer
    outcome = layer1(inputs)
    
    # Add attention mechanism if requested
    if attention:
        attention_probs = Diagonal(n_genes, 
                                   input_shape=(n_features,), 
                                   activation='sigmoid', 
                                   kernel_regularizer=l2(w_reg0),  # Updated from W_regularizer
                                   name='attention0')(inputs)
        # Updated multiply operation with TF2.x API
        outcome = multiply([outcome, attention_probs], name='attention_mul')

    # Initialize decision outcomes list
    decision_outcomes = []
    
    # Store feature names for the initial gene layer (h0)
    feature_names['h0'] = genes
    
    # Create first decision output directly from inputs
    decision_outcome = Dense(1, 
                            activation='linear', 
                            name='o_linear0', 
                            kernel_regularizer=reg_l(w_reg_outcome0))(inputs)  # Updated from W_regularizer

    # Apply batch normalization if requested
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)
    
    # Create second decision output from first hidden layer
    decision_outcome = Dense(1, 
                            activation='linear', 
                            name='o_linear1',
                            kernel_regularizer=reg_l(w_reg_outcome1 / 2.))(outcome)  # Updated from W_regularizer
    
    # Apply dropout with TF2.x training parameter
    drop2 = Dropout(dropout[0], name='dropout_0')
    outcome = drop2(outcome, training=dropout_testing)

    # Apply batch normalization if requested
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    # Activate decision outcome and add to list
    decision_outcome = Activation(activation=activation_decision, name='o1')(decision_outcome)
    decision_outcomes.append(decision_outcome)

    # Handle additional layers if requested
    maps_for_iteration = [] 
    if n_hidden_layers > 0:
        # Get pathway maps from reactome
        maps_for_iteration = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
        
        # Debug information
        print(f'original dropout config: {dropout}')
        
        # Extract configurations for additional layers
        w_regs_iter = w_reg[1:] 
        w_reg_outcomes_list_iter = w_reg_outcomes[2:] 
        dropout_list_iter = dropout[1:] 

        # Iterate through maps and create layers
        for i in range(len(maps_for_iteration)):
            # Extract map and create names
            mapp = maps_for_iteration[i]
            layer_name = f'h{i + 1}'
            decision_name = f'o{i + 2}'
            dropout_name = f'dropout_{i + 1}'

            # Extract pathway dimensions
            n_pathways = mapp.shape[1]
            
            # Store feature names
            current_features_loop = maps_for_iteration[i].columns.tolist()
            feature_names[layer_name] = current_features_loop
            
            # Debug information
            print(f'Loop {i} - map shape: {maps_for_iteration[i].shape}, current_features count: {len(current_features_loop)}')
            print(f'Loop {i} - using dropout_val: {dropout_list_iter[i]}, w_reg_val: {w_regs_iter[i]}')

            # Create pathway layer with TF2.x parameter names
            hidden_layer = SparseTF(n_pathways, 
                                    mapp, 
                                    activation=activation, 
                                    kernel_regularizer=reg_l(w_regs_iter[i]),  # Updated from W_regularizer
                                    name=layer_name, 
                                    kernel_initializer=kernel_initializer, 
                                    use_bias=use_bias,
                                    **constraints)
            outcome = hidden_layer(outcome)

            # Add attention mechanism if requested
            if attention:
                # For attention, we need a pathway-to-pathway map (n_pathways x n_pathways)
                # Create an identity map where each pathway can attend to itself
                import numpy as np
                import pandas as pd
                attention_map = np.eye(n_pathways, dtype=np.float32)
                attention_map_df = pd.DataFrame(attention_map, 
                                               index=mapp.columns,  # pathway names
                                               columns=mapp.columns)  # same pathway names
                
                attention_probs = SparseTF(n_pathways, 
                                          attention_map_df, 
                                          activation='sigmoid', 
                                          kernel_regularizer=reg_l(w_regs_iter[i]),  # Updated from W_regularizer
                                          name=f'attention{i + 1}')(outcome) 
                # Updated multiply operation with TF2.x API
                outcome = multiply([outcome, attention_probs], name=f'attention_mul{i + 1}')

            # Create decision output for this layer
            decision_outcome = Dense(1, 
                                    activation='linear', 
                                    name=f'o_linear{i + 2}',
                                    kernel_regularizer=reg_l(w_reg_outcomes_list_iter[i] / 2.))(outcome)  # Updated from W_regularizer

            # Apply batch normalization if requested
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)

            # Apply dropout with TF2.x training parameter
            drop = Dropout(dropout_list_iter[i], name=dropout_name)
            outcome = drop(outcome, training=dropout_testing)

            # Activate decision outcome and add to list
            decision_outcome = Activation(activation=activation_decision, name=decision_name)(decision_outcome)
            decision_outcomes.append(decision_outcome)

        # Debug information
        print(f'After loop, final outcome tensor: {outcome}')

    # Final debug information
    print(f'Initial genes list: {genes}')
    print(f'Initial features list: {features}')
    
    maps_len_for_print = len(maps_for_iteration) if n_hidden_layers > 0 else 0
    print(f'Initial features_count: {len(features)}, processed maps_for_iteration count: {maps_len_for_print}')
    
    # Return model components
    return outcome, decision_outcomes, feature_names


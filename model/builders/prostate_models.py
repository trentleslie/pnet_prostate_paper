import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Lambda, Concatenate
from tensorflow.keras.regularizers import l2

from data.data_access import Data
from data.pathways.gmt_pathway import get_KEGG_map # This was in original, keep it
from model.builders.builders_utils import get_pnet # This function also needs TF2.x refactoring
from model.layers_custom import f1, Diagonal, SparseTF # Ensure these are TF2.x compatible
from model.model_utils import print_model, get_layers # Already present


# assumes the first node connected to the first n nodes and so on
def build_pnet(optimizer, w_reg, add_unk_genes=True, sparse=True, dropout=0.5, use_bias=False, activation='tanh',
               loss='binary_crossentropy', data_params=None, n_hidden_layers=1, direction='root_to_leaf',
               batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False, reg_outcomes=False,
               ignore_missing_histology=True):
    """
    Builds a Pathway Network (P-NET) model using TensorFlow 2.x Keras.

    This version is an earlier implementation compared to build_pnet2 and may have
    fewer configurable options or different default behaviors for some internal
    parameters passed to get_pnet.

    Args:
        optimizer: Keras optimizer for model compilation (e.g., 'Adam', tf.keras.optimizers.Adam()).
        w_reg (float): Weight regularization factor (L2) for pathway layers.
        add_unk_genes (bool, optional): Whether to add a node for unknown/unmapped genes.
            Defaults to True.
        sparse (bool, optional): Whether to use sparse connections/layers where applicable.
            Defaults to True.
        dropout (float, optional): Dropout rate to apply. Defaults to 0.5.
        use_bias (bool, optional): Whether to use bias terms in Dense layers. Defaults to False.
        activation (str, optional): Activation function for hidden layers (e.g., 'tanh', 'relu').
            Defaults to 'tanh'.
        loss (str, optional): Loss function for model compilation. Defaults to 'binary_crossentropy'.
        data_params (dict, optional): Parameters for data loading and preprocessing via the Data class.
            If None, default Data parameters are used. See `data.data_access.Data`.
        n_hidden_layers (int, optional): Number of hidden pathway layers in the P-NET.
            Defaults to 1.
        direction (str, optional): Direction of information flow in pathways ('root_to_leaf' or 'leaf_to_root').
            Defaults to 'root_to_leaf'.
        batch_normal (bool, optional): Whether to include batch normalization layers.
            Defaults to False.
        kernel_initializer (str, optional): Kernel initializer for Dense layers (e.g., 'glorot_uniform').
            Defaults to 'glorot_uniform'.
        shuffle_genes (bool, optional): Whether to shuffle gene order/connections (experimental).
            Defaults to False.
        reg_outcomes (bool, optional): Whether to apply regularization to outcome layers.
            Note: This parameter name differs from w_reg_outcomes in build_pnet2.
            Defaults to False.
        ignore_missing_histology (bool, optional): If True (default), the model is built using only
            genomic data, and any histology data is ignored. If False, the model will attempt
            to use histology data (currently, this path is not fully implemented and will
            default to genomic-only with a warning).

    Returns:
        tuple: A tuple containing:
            - model (tf.keras.Model): The compiled Keras P-NET model.
            - feature_names (list): A list of feature names corresponding to model layers/outputs.
    """
    print(data_params)
    print(f'n_hidden_layers: {n_hidden_layers}')
    print(f'ignore_missing_histology: {ignore_missing_histology}')
    
    # Handle histology data based on ignore_missing_histology flag
    if ignore_missing_histology:
        # Current implementation: Use only genomic data (mutations, CNAs)
        # Ensure data_params exclude any histology-related features
        if data_params is None:
            data_params = {}
        
        # Create a copy to avoid modifying original params
        genomic_data_params = data_params.copy()
        # Ensure 'params' key exists if it's expected by Data's __init__ for ProstateDataPaper
        if 'params' not in genomic_data_params:
            genomic_data_params['params'] = {}
        genomic_data_params['include_histology_features'] = False
        
        data = Data(**genomic_data_params)
        logging.info('Building P-NET model with genomic data only (histology ignored)')
    else:
        # Future implementation: Could integrate histology data pathway here
        # For now, behave identically to ignore_missing_histology=True
        logging.warning('ignore_missing_histology=False specified, but histology pathway not yet implemented. '
                       'Using genomic data only.')
        
        if data_params is None:
            data_params = {}
        
        # Create a copy to avoid modifying original params
        genomic_data_params = data_params.copy()
        # Ensure 'params' key exists if it's expected by Data's __init__ for ProstateDataPaper
        if 'params' not in genomic_data_params:
            genomic_data_params['params'] = {}
        genomic_data_params['include_histology_features'] = True
        
        data = Data(**genomic_data_params)
        logging.info('Building P-NET model with genomic data only (histology pathway not implemented)')
    
    x, y, info, cols = data.get_data()
    print(x.shape)
    print(y.shape)
    print(info.shape)
    print(cols.shape)
    # features = cols.tolist()
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    # n_genes = len(genes)
    # genes = list(genes)
    # layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg), use_bias=False, name='h0')
    # layer1 = SpraseLayer(n_genes, input_shape=(n_features,), activation=activation,  use_bias=False,name='h0')
    # layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, name='h0')
    ins = Input(shape=(n_features,), dtype='float32', name='inputs')

    outcome, decision_outcomes, feature_n = get_pnet(ins,
                                                     features,
                                                     genes,
                                                     n_hidden_layers,
                                                     direction,
                                                     activation,
                                                     activation_decision,
                                                     w_reg,
                                                     reg_outcomes,
                                                     dropout,
                                                     sparse,
                                                     add_unk_genes,
                                                     batch_normal,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes
                                                     # reg_outcomes=reg_outcomes,
                                                     # adaptive_reg =adaptive_reg,
                                                     # adaptive_dropout=adaptive_dropout
                                                     )
    # outcome= outcome[0:-2]
    # decision_outcomes= decision_outcomes[0:-2]
    # feature_n= feature_n[0:-2]

    feature_names.extend(feature_n)

    print('Compiling...')

    model = Model(inputs=ins, outputs=decision_outcomes)

    # n_outputs = n_hidden_layers + 2
    n_outputs = len(decision_outcomes)
    loss_weights = range(1, n_outputs + 1)
    # loss_weights = [l*l for l in loss_weights]
    loss_weights = [np.exp(l) for l in loss_weights]
    # loss_weights = [l*np.exp(l) for l in loss_weights]
    # loss_weights=1
    print(f'loss_weights: {loss_weights}')
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=loss_weights)
    # loss=['binary_crossentropy']*(n_hidden_layers +2))
    logging.info('done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info(f'# of trainable params of the model is {model.count_params()}')
    return model, feature_names


# assumes the first node connected to the first n nodes and so on
def build_pnet2(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0, dropout=0.5,
                use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None, n_hidden_layers=1,
                direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False,
                attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True, sparse_first_layer=True,
                ignore_missing_histology=True):
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
        ignore_missing_histology: Whether to ignore missing histology data and use only genomic features
        
    Returns:
        model: Compiled Keras model
        feature_names: Dictionary mapping layer names to features
    """
    print(data_params)
    print(f'n_hidden_layers: {n_hidden_layers}')
    print(f'ignore_missing_histology: {ignore_missing_histology}')
    
    # Handle histology data based on ignore_missing_histology flag
    if ignore_missing_histology:
        # Current implementation: Use only genomic data (mutations, CNAs)
        # Ensure data_params exclude any histology-related features
        if data_params is None:
            data_params = {}
        
        # Create a copy to avoid modifying original params
        genomic_data_params = data_params.copy()
        # Ensure 'params' key exists if it's expected by Data's __init__ for ProstateDataPaper
        if 'params' not in genomic_data_params:
            genomic_data_params['params'] = {}
        genomic_data_params['include_histology_features'] = False
        
        data = Data(**genomic_data_params)
        logging.info('Building P-NET2 model with genomic data only (histology ignored)')
    else:
        # Future implementation: Could integrate histology data pathway here
        # For now, behave identically to ignore_missing_histology=True
        logging.warning('ignore_missing_histology=False specified, but histology pathway not yet implemented. '
                       'Using genomic data only.')
        
        if data_params is None:
            data_params = {}
        
        # Create a copy to avoid modifying original params
        genomic_data_params = data_params.copy()
        # Ensure 'params' key exists if it's expected by Data's __init__ for ProstateDataPaper
        if 'params' not in genomic_data_params:
            genomic_data_params['params'] = {}
        genomic_data_params['include_histology_features'] = True
        
        data = Data(**genomic_data_params)
        logging.info('Building P-NET2 model with genomic data only (histology pathway not implemented)')
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

    # Create input layer with shape based on feature count
    ins = Input(shape=(n_features,), dtype='float32', name='inputs')

    # Generate model architecture using get_pnet function
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

    # Store feature names
    feature_names = feature_n
    feature_names['inputs'] = cols

    print('Compiling...')

    # Determine the model output based on repeated_outcomes flag
    if repeated_outcomes:
        model_output = decision_outcomes
    else:
        model_output = decision_outcomes[-1]

    # Create model with TF2.x API
    model = Model(inputs=ins, outputs=model_output)

    # Determine number of outputs for loss configuration
    if isinstance(model_output, list):
        n_outputs = len(model_output)
    else:
        n_outputs = 1

    # Handle loss weights
    if isinstance(loss_weights, list):
        # Use the provided list directly
        current_loss_weights = loss_weights
    else:
        # Create a list with the same value repeated
        current_loss_weights = [loss_weights] * n_outputs

    print(f'loss_weights: {current_loss_weights}')
    
    # Configure loss and metrics based on output structure
    if n_outputs > 1 and isinstance(model_output, list):
        # Option 1: Use dictionary format for named outputs
        # This is commented out but could be enabled for more flexible loss assignment
        # losses = {output.name: 'binary_crossentropy' for output in model_output}
        # metrics = {output.name: f1 for output in model_output}
        # model.compile(optimizer=optimizer, loss=losses, metrics=metrics, loss_weights=current_loss_weights)
        
        # Option 2: Use list format for all outputs (current implementation)
        model.compile(optimizer=optimizer,
                      loss=['binary_crossentropy'] * n_outputs, 
                      metrics=[f1], 
                      loss_weights=current_loss_weights)
    else:
        # Single output compilation
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', 
                      metrics=[f1])

    logging.info('done compiling')

    # Print model information for debugging
    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info(f'# of trainable params of the model is {model.count_params()}')
    
    return model, feature_names


def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output


def get_clinical_netowrk(ins, n_features, n_hids, activation):
    layers = []
    for i, n in enumerate(n_hids):
        if i == 0:
            layer = Dense(n, input_shape=(n_features,), activation=activation, kernel_regularizer=l2(0.001),
                          name='h_clinical' + str(i))
        else:
            layer = Dense(n, activation=activation, kernel_regularizer=l2(0.001), name='h_clinical' + str(i))

        layers.append(layer)
        drop = 0.5
        layers.append(Dropout(drop, name='droput_clinical_{}'.format(i)))

    merged = apply_models(layers, ins)
    output_layer = Dense(1, activation='sigmoid', name='clinical_out')
    outs = output_layer(merged)

    return outs


def build_pnet2_account_for(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0,
                            dropout=0.5,
                            use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None,
                            n_hidden_layers=1,
                            direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform',
                            shuffle_genes=False,
                            attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True,
                            sparse_first_layer=True):
    print(data_params)

    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    assert len(
        cols.levels) == 3, "expect to have pandas dataframe with 3 levels [{'clinicla, 'genomics'}, genes, features] "

    import pandas as pd
    x_df = pd.DataFrame(x, columns=cols, index=info)
    genomics_label = list(x_df.columns.levels[0]).index(u'genomics')
    genomics_ind = x_df.columns.labels[0] == genomics_label
    genomics = x_df['genomics']
    features_genomics = genomics.columns.remove_unused_levels()

    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    n_features = x_df.shape[1]
    n_features_genomics = len(features_genomics)

    if hasattr(features_genomics, 'levels'):
        genes = features_genomics.levels[0]
    else:
        genes = features_genomics

    print(f"n_features: {n_features}, n_features_genomics: {n_features_genomics}")
    print(f"genes: {len(genes)}, {genes}")

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')

    ins_genomics = Lambda(lambda x: x[:, 0:n_features_genomics])(ins)
    ins_clinical = Lambda(lambda x: x[:, n_features_genomics:n_features])(ins)

    clinical_outs = get_clinical_netowrk(ins_clinical, n_features, n_hids=[50, 1], activation=activation)

    outcome_from_pnet, decision_outcomes_from_pnet, feature_n = get_pnet(ins_genomics,
                                                     features=features_genomics,
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
    feature_names['inputs'] = x_df.columns

    print('Compiling...')

    pnet_outputs_for_concat = []
    if repeated_outcomes:
        pnet_outputs_for_concat = decision_outcomes_from_pnet if isinstance(decision_outcomes_from_pnet, list) else [decision_outcomes_from_pnet]
    else:
        # If not repeated, outcome_from_pnet is the single tensor output from get_pnet's last layer before combined layer
        pnet_outputs_for_concat = [outcome_from_pnet]

    # Construct the list of outputs that will be concatenated
    concatenation_list = pnet_outputs_for_concat + [clinical_outs]

    combined_tensor = Concatenate(axis=-1, name='combine')(concatenation_list)
    final_combined_output_tensor = Dense(1, activation='sigmoid', name='combined_outcome')(combined_tensor)
    
    # Define all model outputs: pnet individual (if repeated), clinical, and final combined
    final_model_outputs = pnet_outputs_for_concat + [clinical_outs, final_combined_output_tensor]

    model = Model(inputs=[ins], outputs=final_model_outputs)

    n_outputs = len(final_model_outputs)
    
    current_loss_weights = []
    if isinstance(loss_weights, list):
        if len(loss_weights) == n_outputs:
            current_loss_weights = loss_weights
        else:
            # Attempt to match provided loss_weights to the P-NET part, and add default for others
            num_pnet_related_outputs = len(pnet_outputs_for_concat)
            if len(loss_weights) >= num_pnet_related_outputs:
                current_loss_weights.extend(loss_weights[:num_pnet_related_outputs])
            else: # Not enough weights for pnet part
                current_loss_weights.extend([1.0] * num_pnet_related_outputs) # Default for pnet outputs
            
            # Add default weights for clinical_outs and final_combined_output_tensor
            # Assuming they should also be weighted, e.g. as 1.0 by default
            while len(current_loss_weights) < n_outputs:
                 current_loss_weights.append(1.0) # Default for additional outputs
            print(f"Warning: loss_weights list length mismatch. Adjusted to {n_outputs} outputs.")
    else: # Single float value, apply to all
        current_loss_weights = [loss_weights] * n_outputs

    print(f'loss_weights: {current_loss_weights}')
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=current_loss_weights)
    logging.info('done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info(f'# of trainable params of the model is {model.count_params()}')
    return model, feature_names


def build_dense(optimizer, n_weights, w_reg, activation='tanh', loss='binary_crossentropy', data_params=None, ignore_missing_histology=False):
    print(data_params)

    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print(x.shape)
    print(y.shape)
    print(info.shape)
    print(cols.shape)
    # features = cols.tolist()
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')
    n = np.ceil(float(n_weights) / float(n_features))
    print(n)
    layer1 = Dense(units=int(n), activation=activation, kernel_regularizer=l2(w_reg), name='h0')
    outcome = layer1(ins)
    outcome = Dense(1, activation=activation_decision, name='output')(outcome)
    model = Model(inputs=[ins], outputs=outcome)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=[f1])
    logging.info('done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info(f'# of trainable params of the model is {model.count_params()}')
    return model, feature_names


def build_pnet_KEGG(optimizer, w_reg, dropout=0.5, activation='tanh', use_bias=False,
                    kernel_initializer='glorot_uniform', data_params=None, arch=''):
    print(data_params)
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print(x.shape)
    print(y.shape)
    print(info.shape)
    print(cols.shape)

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = {}
    feature_names['inputs'] = cols
    # feature_names.append(cols)

    n_features = x.shape[1]
    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    feature_names['h0'] = genes
    # feature_names.append(genes)
    decision_outcomes = []
    n_genes = len(genes)
    genes = list(genes)

    layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, kernel_regularizer=l2(w_reg),
                      use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')
    layer1_output = layer1(ins)

    decision0 = Dense(1, activation='sigmoid', name='o0'.format(0))(ins)
    decision_outcomes.append(decision0)

    decision1 = Dense(1, activation='sigmoid', name='o{}'.format(1))(layer1_output)
    decision_outcomes.append(decision1)

    mapp, genes, pathways = get_KEGG_map(genes, arch)

    n_genes, n_pathways = mapp.shape
    logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))

    hidden_layer = SparseTF(n_pathways, mapp, activation=activation, kernel_regularizer=l2(w_reg),
                            name='h1', kernel_initializer=kernel_initializer,
                            use_bias=use_bias)

    # hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=L1L2_with_map(mapp, w_reg, w_reg),
    #                      kernel_constraint=ConnectionConstaints(mapp), use_bias=False,
    #                      name='h1')

    layer2_output = hidden_layer(layer1_output)
    decision2 = Dense(1, activation='sigmoid', name='o2')(layer2_output)
    decision_outcomes.append(decision2)

    feature_names['h1'] = pathways
    # feature_names.append(pathways)
    print('Compiling...')

    model = Model(inputs=[ins], outputs=decision_outcomes)

    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * 3, metrics=[f1])
    # loss=['binary_crossentropy']*(n_hidden_layers +2))
    logging.info('done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info(f'# of trainable params of the model is {model.count_params()}')
    return model, feature_names

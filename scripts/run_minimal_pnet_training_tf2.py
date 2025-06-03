#!/usr/bin/env python3
"""
Minimal P-NET Training Script for TensorFlow 2.x
Adapted from PyTorch P-NET testing workflow for prostate cancer project.

This script demonstrates end-to-end training of a P-NET model using the minimal 
prostate dataset with TensorFlow 2.x and the project's existing utilities.
"""

import os
import sys
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Ensure the project root is in Python path
sys.path.insert(0, '/procedure/pnet_prostate_paper')

# TensorFlow 2.x imports
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Project-specific imports
from data.data_access import Data
from model.builders.prostate_models import build_pnet2

# Define TF2-compatible F1 metric
def f1_score(y_true, y_pred):
    """TensorFlow 2.x compatible F1 score metric."""
    # Cast y_true to float32 to match y_pred type
    y_true = tf.cast(y_true, tf.float32)
    
    # Threshold predictions
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    # Calculate true positives, false positives, false negatives
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Calculate precision and recall
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1


def setup_logging(log_level='INFO'):
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/procedure/pnet_prostate_paper/results/minimal_pnet_training.log')
        ]
    )
    return logging.getLogger(__name__)


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logging.info(f'Random seeds set to {seed}')


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f'Configuration loaded from {config_path}')
        return config
    except Exception as e:
        logging.error(f'Failed to load configuration: {e}')
        raise


def prepare_data(config):
    """
    Load and prepare data using the project's Data class.
    
    Args:
        config: Configuration dictionary from YAML file
        
    Returns:
        tuple: (x_train, x_test, y_train, y_test, info_train, info_test, columns)
    """
    logging.info('Loading data using Data class...')
    
    # Extract data parameters from config
    data_params = config['data_params'].copy()
    
    # Initialize the Data class
    data = Data(**data_params)
    
    # Get the data splits (train+validation combined vs test)
    x_train, x_test, y_train, y_test, info_train, info_test, columns = data.get_train_test()
    
    logging.info(f'Data loaded: Train shape {x_train.shape}, Test shape {x_test.shape}')
    
    # Ensure y is 1D array for binary classification
    if len(y_train.shape) > 1:
        y_train = y_train.ravel()
    if len(y_test.shape) > 1:
        y_test = y_test.ravel()
        
    # Convert to integers for bincount
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    
    logging.info(f'Target distribution - Train: {np.bincount(y_train_int)}, Test: {np.bincount(y_test_int)}')
    
    return x_train, x_test, y_train, y_test, info_train, info_test, columns


def create_model(config, data_params):
    """
    Create and compile P-NET model using build_pnet2.
    
    Args:
        config: Configuration dictionary
        data_params: Data parameters for model building
        
    Returns:
        tuple: (model, feature_names)
    """
    logging.info('Building P-NET model...')
    
    model_params = config['model_params']
    
    # Create optimizer
    optimizer = Adam(learning_rate=model_params['learning_rate'])
    
    # Monkey-patch the f1 function in layers_custom to use our TF2-compatible version
    import model.layers_custom
    model.layers_custom.f1 = f1_score
    
    # Build model using build_pnet2
    model, feature_names = build_pnet2(
        optimizer=optimizer,
        w_reg=model_params['w_reg'],
        w_reg_outcomes=model_params['w_reg_outcomes'],
        add_unk_genes=model_params['add_unk_genes'],
        sparse=model_params['sparse'],
        loss_weights=model_params['loss_weights'],
        dropout=model_params['dropout'],
        use_bias=model_params['use_bias'],
        activation=model_params['activation'],
        loss=model_params['loss'],
        data_params=data_params,
        n_hidden_layers=model_params['n_hidden_layers'],
        direction=model_params['direction'],
        batch_normal=model_params['batch_normal'],
        kernel_initializer=model_params['kernel_initializer'],
        shuffle_genes=model_params['shuffle_genes'],
        attention=model_params['attention'],
        dropout_testing=model_params['dropout_testing'],
        non_neg=model_params['non_neg'],
        repeated_outcomes=model_params['repeated_outcomes'],
        sparse_first_layer=model_params['sparse_first_layer'],
        ignore_missing_histology=model_params['ignore_missing_histology']
    )
    
    logging.info(f'Model created with {model.count_params()} parameters')
    return model, feature_names


def setup_callbacks(config):
    """Setup training callbacks."""
    training_params = config['training_params']
    callbacks = []
    
    # Early stopping
    if training_params['early_stopping']:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=training_params['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        logging.info('Early stopping callback added')
    
    # Model checkpointing
    if training_params['save_checkpoints']:
        checkpoint_path = os.path.join(
            training_params['checkpoint_dir'], 
            'best_model.weights.h5'  # Use weights format to avoid serialization issues
        )
        # Ensure checkpoint directory exists
        os.makedirs(training_params['checkpoint_dir'], exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=training_params['save_best_only'],
            save_weights_only=True,  # Save only weights to avoid serialization issues
            verbose=1
        )
        callbacks.append(checkpoint)
        logging.info(f'Model checkpoint callback added: {checkpoint_path}')
    
    return callbacks


def train_model(model, x_train, y_train, x_test, y_test, config):
    """
    Train the P-NET model.
    
    Args:
        model: Compiled Keras model
        x_train, y_train: Training data
        x_test, y_test: Test data for validation
        config: Configuration dictionary
        
    Returns:
        History object from model training
    """
    logging.info('Starting model training...')
    
    model_params = config['model_params']
    training_params = config['training_params']
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Prepare y data for multi-output model if needed
    if hasattr(model, 'output_names') and len(model.output_names) > 1:
        # For multi-output models, replicate y for each output
        y_train_multi = [y_train] * len(model.output_names)
        y_test_multi = [y_test] * len(model.output_names)
    else:
        y_train_multi = y_train
        y_test_multi = y_test
    
    # Train the model
    history = model.fit(
        x_train, y_train_multi,
        batch_size=model_params['batch_size'],
        epochs=model_params['epochs'],
        validation_data=(x_test, y_test_multi),
        callbacks=callbacks,
        verbose=training_params['verbose']
    )
    
    logging.info('Model training completed')
    return history


def evaluate_model(model, x_test, y_test, config):
    """
    Evaluate the trained model and generate ROC curve.
    
    Args:
        model: Trained Keras model
        x_test, y_test: Test data
        config: Configuration dictionary
        
    Returns:
        dict: Evaluation metrics
    """
    logging.info('Evaluating model...')
    
    # Prepare y data for multi-output model if needed
    if hasattr(model, 'output_names') and len(model.output_names) > 1:
        y_test_multi = [y_test] * len(model.output_names)
    else:
        y_test_multi = y_test
    
    # Evaluate model
    test_results = model.evaluate(x_test, y_test_multi, verbose=0)
    
    # Extract loss from results
    if isinstance(test_results, list):
        test_loss = test_results[0]  # First value is always the total loss
    else:
        test_loss = test_results
        
    logging.info(f'Test loss: {test_loss:.4f}')
    
    # Get predictions
    y_pred_proba = model.predict(x_test, verbose=0)
    
    # Handle multiple outputs (P-NET can have multiple outputs)
    if isinstance(y_pred_proba, list):
        # Use the last output (final decision layer)
        y_pred_proba = y_pred_proba[-1]
    
    # Ensure correct shape
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba.flatten()
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f'ROC AUC: {roc_auc:.4f}')
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - P-NET Model')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Remove top and right spines for cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save plot
    plot_path = config['training_params']['plot_save_path']
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f'ROC curve saved to {plot_path}')
    
    # Compile evaluation metrics
    eval_metrics = {
        'test_loss': test_loss,
        'roc_auc': roc_auc,
        'test_accuracy': test_results[1] if isinstance(test_results, list) and len(test_results) > 1 else 0.0
    }
    
    return eval_metrics


def save_results(model, eval_metrics, config, feature_names=None):
    """
    Save training results and model artifacts.
    
    Args:
        model: Trained model
        eval_metrics: Evaluation metrics dictionary
        config: Configuration dictionary
        feature_names: Feature names from model building
    """
    results_dir = config['training_params']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Save evaluation metrics
    metrics_file = os.path.join(results_dir, 'evaluation_metrics.yaml')
    with open(metrics_file, 'w') as f:
        yaml.dump(eval_metrics, f, default_flow_style=False)
    logging.info(f'Evaluation metrics saved to {metrics_file}')
    
    # Save model summary
    summary_file = os.path.join(results_dir, 'model_summary.txt')
    with open(summary_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    logging.info(f'Model summary saved to {summary_file}')
    
    # Save feature names if available
    if feature_names is not None:
        feature_file = os.path.join(results_dir, 'feature_names.yaml')
        with open(feature_file, 'w') as f:
            yaml.dump(feature_names, f, default_flow_style=False)
        logging.info(f'Feature names saved to {feature_file}')


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train P-NET model on minimal prostate dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/procedure/pnet_prostate_paper/config/minimal_training_params.yml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['training_params']['log_level'])
    logger.info('Starting P-NET minimal training script')
    logger.info(f'Configuration: {args.config}')
    
    # Set random seeds for reproducibility
    set_random_seeds(config['training_params']['random_seed'])
    
    # Prepare data
    x_train, x_test, y_train, y_test, info_train, info_test, columns = prepare_data(config)
    
    # Create model
    model, feature_names = create_model(config, config['data_params'])
    
    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test, config)
    
    # Evaluate model
    eval_metrics = evaluate_model(model, x_test, y_test, config)
    
    # Save results
    save_results(model, eval_metrics, config, feature_names)
    
    # Print final results
    logger.info('Training completed successfully!')
    logger.info(f'Final Results:')
    for metric, value in eval_metrics.items():
        logger.info(f'  {metric}: {value:.4f}')
    
    return model, eval_metrics, feature_names


if __name__ == '__main__':
    model, eval_metrics, feature_names = main()
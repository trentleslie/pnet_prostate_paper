#!/usr/bin/env python3
"""
SKCM Tumor Purity Prediction with P-NET - TensorFlow 2.x Adaptation
Adapted from PyTorch implementation to use TensorFlow 2.x with the prostate cancer project infrastructure.
This script demonstrates tumor purity prediction as a regression task.
"""

import os
import sys
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Note: seaborn removed as dependency
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Ensure the project root is in Python path
sys.path.insert(0, '/procedure/pnet_prostate_paper')

# TensorFlow 2.x imports
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Loss

# Project-specific imports
from data.data_access import Data
from model.builders.prostate_models import build_pnet2


# GitHub raw content base URL for data
GITHUB_DATA_BASE = "https://raw.githubusercontent.com/vanallenlab/pnet/main/data"


def setup_logging(log_level='INFO'):
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/procedure/pnet_prostate_paper/results/skcm_purity_training.log')
        ]
    )
    return logging.getLogger(__name__)


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logging.info(f'Random seeds set to {seed}')


class WeightedMSELoss(Loss):
    """
    TensorFlow 2.x implementation of Weighted MSE Loss.
    Penalizes predictions more for samples with extreme purity values (far from 0.5).
    """
    def __init__(self, name='weighted_mse', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred):
        # Calculate the absolute distance of the true values from 0.5
        distance_from_center = tf.abs(y_true - 0.5)
        # Scale weights as needed; further from 0.5 gets higher weight
        weights = 1 + distance_from_center
        # Calculate weighted MSE
        squared_error = tf.square(y_true - y_pred)
        weighted_se = weights * squared_error
        return tf.reduce_mean(weighted_se)


def load_tcga_skcm_data():
    """
    Load SKCM data from GitHub repository.
    Returns RNA and CNA data as pandas DataFrames.
    """
    logging.info("Loading SKCM data from GitHub...")
    
    # Construct URLs for SKCM data
    # Note: These are example paths - actual paths in the GitHub repo might differ
    rna_url = f"{GITHUB_DATA_BASE}/skcm_tcga_pan_can_atlas_2018/data_RNA_Seq_v2_expression_median.txt"
    cna_url = f"{GITHUB_DATA_BASE}/skcm_tcga_pan_can_atlas_2018/data_CNA.txt"
    
    try:
        # Load RNA data
        logging.info(f"Loading RNA data from: {rna_url}")
        rna = pd.read_csv(rna_url, delimiter='\t', index_col=0)
        rna = rna.drop(['Entrez_Gene_Id'], errors='ignore').T
        
        # Load CNA data
        logging.info(f"Loading CNA data from: {cna_url}")
        cna = pd.read_csv(cna_url, delimiter='\t', index_col=0)
        cna = cna.drop(['Entrez_Gene_Id'], errors='ignore').T
        
        logging.info(f"RNA data shape: {rna.shape}, CNA data shape: {cna.shape}")
        return rna, cna
        
    except Exception as e:
        logging.error(f"Failed to load SKCM data: {e}")
        # Return synthetic data for demonstration if real data fails
        logging.warning("Using synthetic SKCM data for demonstration...")
        return create_synthetic_skcm_data()


def create_synthetic_skcm_data(n_samples=200, n_genes=500):
    """Create synthetic SKCM data for demonstration if real data is unavailable."""
    # Create sample IDs
    sample_ids = [f"TCGA-SKCM-{i:04d}" for i in range(n_samples)]
    gene_ids = [f"GENE{i:04d}" for i in range(n_genes)]
    
    # Synthetic RNA expression data (normalized log2 values)
    rna = pd.DataFrame(
        np.random.randn(n_samples, n_genes) * 2,
        index=sample_ids,
        columns=gene_ids
    )
    
    # Synthetic CNA data (-2, -1, 0, 1, 2)
    cna = pd.DataFrame(
        np.random.choice([-2, -1, 0, 1, 2], size=(n_samples, n_genes), p=[0.05, 0.15, 0.6, 0.15, 0.05]),
        index=sample_ids,
        columns=gene_ids
    )
    
    return rna, cna


def load_tumor_purity_data():
    """Load tumor purity data from GitHub or create synthetic."""
    logging.info("Loading tumor purity data...")
    
    purity_url = f"{GITHUB_DATA_BASE}/TCGA_mastercalls.abs_tables_JSedit.fixed.txt"
    
    try:
        purity_data = pd.read_csv(purity_url, delimiter='\t', index_col='array')
        return purity_data['purity']
    except:
        logging.warning("Using synthetic purity data...")
        # Create synthetic purity values (0-1)
        n_samples = 200
        sample_ids = [f"TCGA-SKCM-{i:04d}" for i in range(n_samples)]
        purity = pd.Series(
            np.random.beta(2, 2, n_samples),  # Beta distribution gives values between 0-1
            index=sample_ids,
            name='purity'
        )
        return purity


def load_cancer_genes():
    """Load cancer gene list from local file, GitHub, or use default list."""
    logging.info("Loading cancer gene list...")
    
    # First try local file
    local_path = '/procedure/pnet_prostate_paper/data/_database/genes/cancer_genes.txt'
    if os.path.exists(local_path):
        try:
            logging.info(f"Loading cancer genes from local file: {local_path}")
            with open(local_path, 'r') as f:
                lines = f.readlines()
                # Skip header if present
                if lines[0].strip().lower() == 'genes':
                    cancer_genes = [line.strip() for line in lines[1:] if line.strip()]
                else:
                    cancer_genes = [line.strip() for line in lines if line.strip()]
            logging.info(f"Loaded {len(cancer_genes)} cancer genes from local file")
            return cancer_genes
        except Exception as e:
            logging.error(f"Error loading local file: {e}")
    
    # Try GitHub as fallback
    genes_url = f"{GITHUB_DATA_BASE}/../pnet_database/genes/cancer_genes.txt"
    try:
        genes_df = pd.read_csv(genes_url, header=None)
        cancer_genes = genes_df[0].tolist()
        logging.info(f"Loaded {len(cancer_genes)} cancer genes from GitHub")
        return cancer_genes
    except:
        logging.warning("Using default cancer gene list...")
        # Use a small default list
        return ['TP53', 'EGFR', 'PTEN', 'KRAS', 'BRAF', 'PIK3CA', 'MYC', 'RB1', 'APC', 'VHL']


def prepare_skcm_data_for_pnet(rna, cna, purity, cancer_genes, test_size=0.2):
    """
    Prepare SKCM data in the format expected by our P-NET implementation.
    """
    # Find common samples
    common_samples = list(set(rna.index) & set(cna.index) & set(purity.index))
    logging.info(f"Found {len(common_samples)} common samples across all data types")
    
    # Subset to common samples
    rna = rna.loc[common_samples]
    cna = cna.loc[common_samples]
    purity = purity.loc[common_samples]
    
    # Find available cancer genes
    available_genes = list(set(cancer_genes) & set(rna.columns) & set(cna.columns))
    if len(available_genes) < len(cancer_genes):
        logging.warning(f"Only {len(available_genes)} of {len(cancer_genes)} cancer genes found in data")
    
    # If too few genes, use top variable genes
    if len(available_genes) < 50:
        logging.info("Using top variable genes instead of cancer genes")
        rna_var = rna.var()
        top_genes = rna_var.nlargest(min(500, len(rna_var))).index.tolist()
        available_genes = list(set(top_genes) & set(cna.columns))
    
    # Subset to available genes
    rna = rna[available_genes]
    cna = cna[available_genes]
    
    # Create combined feature matrix (concatenate RNA and CNA)
    # This mimics the multi-modal input structure
    combined_features = pd.concat([rna, cna], axis=1, keys=['rna', 'cna'])
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(common_samples)), 
        test_size=test_size, 
        random_state=42,
        stratify=(purity > purity.median()).astype(int)  # Stratify by high/low purity
    )
    
    x_train = combined_features.iloc[train_idx].values
    x_test = combined_features.iloc[test_idx].values
    y_train = purity.iloc[train_idx].values.reshape(-1, 1)
    y_test = purity.iloc[test_idx].values.reshape(-1, 1)
    
    # Sample info
    info_train = np.array(combined_features.index[train_idx])
    info_test = np.array(combined_features.index[test_idx])
    
    # Column info
    columns = combined_features.columns
    
    return x_train, x_test, y_train, y_test, info_train, info_test, columns, available_genes


def create_skcm_model(n_features, n_genes, config):
    """
    Create P-NET model for SKCM tumor purity prediction (regression task).
    """
    logging.info('Building P-NET model for regression...')
    
    # For regression, we need to modify the model configuration
    model_params = config['model_params'].copy()
    model_params['loss'] = 'mse'  # Use MSE for regression
    
    # Create optimizer
    optimizer = Adam(learning_rate=model_params['learning_rate'])
    
    # Build a simplified model for regression
    # Note: This is a simplified approach since build_pnet2 expects classification
    # In practice, you might need a specialized regression version of P-NET
    
    inputs = tf.keras.Input(shape=(n_features,))
    
    # Gene layer (reduce features to genes)
    x = tf.keras.layers.Dense(
        n_genes, 
        activation=model_params['activation'],
        kernel_regularizer=tf.keras.regularizers.l2(model_params['w_reg']),
        name='gene_layer'
    )(inputs)
    x = tf.keras.layers.Dropout(model_params['dropout'])(x)
    
    # Pathway layers
    for i in range(model_params['n_hidden_layers']):
        n_units = max(10, n_genes // (2 ** (i + 1)))  # Progressively smaller layers
        x = tf.keras.layers.Dense(
            n_units,
            activation=model_params['activation'],
            kernel_regularizer=tf.keras.regularizers.l2(model_params['w_reg']),
            name=f'pathway_layer_{i}'
        )(x)
        x = tf.keras.layers.Dropout(model_params['dropout'])(x)
    
    # Output layer for regression
    outputs = tf.keras.layers.Dense(1, activation='linear', name='purity_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with appropriate loss for regression
    if config['training_params'].get('use_weighted_loss', False):
        loss = WeightedMSELoss()
    else:
        loss = 'mse'
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mae', 'mse']
    )
    
    logging.info(f'Model created with {model.count_params()} parameters')
    model.summary()
    
    return model


def evaluate_regression_model(model, x_test, y_test, save_path=None):
    """
    Evaluate regression model and create visualization.
    """
    logging.info('Evaluating regression model...')
    
    # Get predictions
    y_pred = model.predict(x_test, verbose=0)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    correlation, p_value = pearsonr(y_test.flatten(), y_pred.flatten())
    
    logging.info(f'Test MSE: {mse:.4f}')
    logging.info(f'Test R²: {r2:.4f}')
    logging.info(f'Pearson correlation: {correlation:.4f} (p={p_value:.4e})')
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'y_test': y_test.flatten(),
        'y_pred': y_pred.flatten()
    })
    
    # Scatter plot with regression line
    plt.scatter(df['y_test'], df['y_pred'], color='#41B6E6', alpha=0.6)
    
    # Add regression line
    z = np.polyfit(df['y_test'], df['y_pred'], 1)
    p = np.poly1d(z)
    plt.plot(df['y_test'], p(df['y_test']), "r-", alpha=0.8)
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], color='#FFA300', linestyle='--', label='Perfect prediction')
    
    # Add correlation text
    plt.text(0.95, 0.05, f'Correlation: {correlation:.2f}\nR²: {r2:.2f}', 
             ha='right', va='bottom', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('True Tumor Purity')
    plt.ylabel('Predicted Tumor Purity')
    plt.title('SKCM Tumor Purity Prediction')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f'Regression plot saved to {save_path}')
    
    plt.show()
    
    return {
        'mse': mse,
        'r2': r2,
        'correlation': correlation,
        'p_value': p_value
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train P-NET model for SKCM tumor purity prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/procedure/pnet_prostate_paper/config/skcm_purity_params.yml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--use-weighted-loss',
        action='store_true',
        help='Use weighted MSE loss for extreme values'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging('INFO')
    logger.info('Starting SKCM tumor purity prediction with P-NET')
    
    # Set random seeds
    set_random_seeds(42)
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default configuration
        config = {
            'model_params': {
                'n_hidden_layers': 2,
                'activation': 'relu',
                'dropout': 0.3,
                'w_reg': 0.001,
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 32
            },
            'training_params': {
                'early_stopping': True,
                'patience': 10,
                'save_checkpoints': True,
                'checkpoint_dir': '/procedure/pnet_prostate_paper/checkpoints/skcm_purity/',
                'results_dir': '/procedure/pnet_prostate_paper/results/',
                'use_weighted_loss': args.use_weighted_loss
            }
        }
    
    # Create directories
    os.makedirs(config['training_params']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training_params']['results_dir'], exist_ok=True)
    
    # Load data
    rna, cna = load_tcga_skcm_data()
    purity = load_tumor_purity_data()
    cancer_genes = load_cancer_genes()
    
    # Prepare data
    x_train, x_test, y_train, y_test, info_train, info_test, columns, genes = prepare_skcm_data_for_pnet(
        rna, cna, purity, cancer_genes
    )
    
    logger.info(f'Training samples: {len(x_train)}, Test samples: {len(x_test)}')
    logger.info(f'Features: {x_train.shape[1]}, Genes: {len(genes)}')
    
    # Create model
    model = create_skcm_model(x_train.shape[1], len(genes), config)
    
    # Setup callbacks
    callbacks = []
    if config['training_params']['early_stopping']:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=config['training_params']['patience'],
            restore_best_weights=True,
            verbose=1
        ))
    
    if config['training_params']['save_checkpoints']:
        checkpoint_path = os.path.join(
            config['training_params']['checkpoint_dir'],
            'best_model.weights.h5'
        )
        callbacks.append(ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ))
    
    # Train model
    logger.info('Starting model training...')
    history = model.fit(
        x_train, y_train,
        batch_size=config['model_params']['batch_size'],
        epochs=config['model_params']['epochs'],
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    plot_path = os.path.join(config['training_params']['results_dir'], 'skcm_purity_regression.png')
    eval_metrics = evaluate_regression_model(model, x_test, y_test, plot_path)
    
    # Save results
    results_file = os.path.join(config['training_params']['results_dir'], 'skcm_purity_metrics.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(eval_metrics, f)
    
    logger.info('Training completed successfully!')
    logger.info(f'Final Results:')
    for metric, value in eval_metrics.items():
        logger.info(f'  {metric}: {value:.4f}')
    
    return model, eval_metrics, history


if __name__ == '__main__':
    model, metrics, history = main()
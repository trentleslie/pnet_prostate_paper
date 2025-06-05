# %% [markdown]
# # P-NET Example Notebook: SKCM Tumor Purity Prediction (TensorFlow 2.x)
# 
# This notebook demonstrates using P-NET for tumor purity prediction as a regression task.
# Adapted from the PyTorch implementation to use TensorFlow 2.x.

# %% [markdown]
# ## Imports

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Note: seaborn removed as dependency
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/procedure/pnet_prostate_paper')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Loss

# Set display options
pd.set_option('display.max_columns', None)

# Configure matplotlib for cleaner plots without seaborn
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# %% [markdown]
# ## Configuration

# %%
# GitHub data base URL
GITHUB_DATA_BASE = "https://raw.githubusercontent.com/vanallenlab/pnet/main/data"

# Model configuration
config = {
    'n_hidden_layers': 2,
    'activation': 'relu',
    'dropout': 0.3,
    'w_reg': 0.001,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'use_weighted_loss': False  # Set to True to use weighted MSE
}

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# %% [markdown]
# ## Custom Loss Function
# 
# We define a weighted MSE loss that penalizes bad predictions in extreme samples more.

# %%
class WeightedMSELoss(Loss):
    """
    Weighted MSE Loss that penalizes predictions more for samples 
    with extreme purity values (far from 0.5).
    """
    def __init__(self, name='weighted_mse', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred):
        # Calculate the absolute distance of the true values from 0.5
        distance_from_center = tf.abs(y_true - 0.5)
        # Scale weights: further from 0.5 gets higher weight
        weights = 1 + distance_from_center
        # Calculate weighted MSE
        squared_error = tf.square(y_true - y_pred)
        weighted_se = weights * squared_error
        return tf.reduce_mean(weighted_se)

# %% [markdown]
# ## Load Data
# 
# We'll attempt to load SKCM data from the GitHub repository. If that fails, we'll use synthetic data for demonstration.

# %%
def load_tcga_skcm_data():
    """Load SKCM RNA and CNA data."""
    print("Attempting to load SKCM data from GitHub...")
    
    # Try to load from GitHub
    try:
        # Note: Update these URLs based on actual file structure in the repository
        rna_url = f"{GITHUB_DATA_BASE}/skcm_tcga_pan_can_atlas_2018/data_RNA_Seq_v2_expression_median.txt"
        cna_url = f"{GITHUB_DATA_BASE}/skcm_tcga_pan_can_atlas_2018/data_CNA.txt"
        
        rna = pd.read_csv(rna_url, delimiter='\t', index_col=0)
        rna = rna.drop(['Entrez_Gene_Id'], errors='ignore').T
        
        cna = pd.read_csv(cna_url, delimiter='\t', index_col=0)
        cna = cna.drop(['Entrez_Gene_Id'], errors='ignore').T
        
        print(f"Successfully loaded data - RNA shape: {rna.shape}, CNA shape: {cna.shape}")
        return rna, cna
        
    except Exception as e:
        print(f"Failed to load from GitHub: {e}")
        print("Using synthetic data for demonstration...")
        
        # Create synthetic data
        n_samples = 200
        n_genes = 500
        sample_ids = [f"TCGA-SKCM-{i:04d}" for i in range(n_samples)]
        gene_ids = [f"GENE{i:04d}" for i in range(n_genes)]
        
        rna = pd.DataFrame(
            np.random.randn(n_samples, n_genes) * 2,
            index=sample_ids,
            columns=gene_ids
        )
        
        cna = pd.DataFrame(
            np.random.choice([-2, -1, 0, 1, 2], size=(n_samples, n_genes), 
                           p=[0.05, 0.15, 0.6, 0.15, 0.05]),
            index=sample_ids,
            columns=gene_ids
        )
        
        return rna, cna

# Load the data
rna, cna = load_tcga_skcm_data()
print(f"RNA shape: {rna.shape}")
print(f"CNA shape: {cna.shape}")

# %% [markdown]
# ## Load Tumor Purity Data

# %%
def load_tumor_purity_data(rna_samples):
    """Load or create tumor purity data."""
    try:
        purity_url = f"{GITHUB_DATA_BASE}/TCGA_mastercalls.abs_tables_JSedit.fixed.txt"
        purity_data = pd.read_csv(purity_url, delimiter='\t', index_col='array')
        
        # Get common samples
        common_samples = list(set(purity_data.index) & set(rna_samples))
        if len(common_samples) > 0:
            return purity_data.loc[common_samples, 'purity']
    except:
        pass
    
    print("Using synthetic purity data...")
    # Create synthetic purity values using beta distribution
    purity = pd.Series(
        np.random.beta(2, 2, len(rna_samples)),
        index=rna_samples,
        name='purity'
    )
    return purity

purity = load_tumor_purity_data(rna.index)
print(f"Purity data shape: {purity.shape}")
print(f"Purity range: [{purity.min():.3f}, {purity.max():.3f}]")

# Visualize purity distribution
plt.figure(figsize=(8, 4))
plt.hist(purity, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Tumor Purity')
plt.ylabel('Frequency')
plt.title('Distribution of Tumor Purity in SKCM Dataset')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Load Cancer Gene List

# %%
def load_cancer_genes():
    """Load cancer gene list."""
    # First try local file
    local_path = '/procedure/pnet_prostate_paper/data/_database/genes/cancer_genes.txt'
    if os.path.exists(local_path):
        try:
            print(f"Loading cancer genes from local file: {local_path}")
            with open(local_path, 'r') as f:
                lines = f.readlines()
                # Skip header if present
                if lines[0].strip().lower() == 'genes':
                    genes = [line.strip() for line in lines[1:] if line.strip()]
                else:
                    genes = [line.strip() for line in lines if line.strip()]
            print(f"Loaded {len(genes)} cancer genes from local file")
            return genes
        except Exception as e:
            print(f"Error loading local file: {e}")
    
    # Try GitHub as fallback
    try:
        genes_url = f"{GITHUB_DATA_BASE}/../pnet_database/genes/cancer_genes.txt"
        genes_df = pd.read_csv(genes_url, header=None)
        return genes_df[0].tolist()
    except:
        # Use a default list of well-known cancer genes
        print("Using default cancer gene list")
        return ['TP53', 'EGFR', 'PTEN', 'KRAS', 'BRAF', 'PIK3CA', 'MYC', 
                'RB1', 'APC', 'VHL', 'CDKN2A', 'NRAS', 'IDH1', 'BRCA1', 'BRCA2']

cancer_genes = load_cancer_genes()
print(f"Loaded {len(cancer_genes)} cancer genes")
print(f"First 5 genes: {cancer_genes[:5]}")

# %% [markdown]
# ## Prepare Data for P-NET

# %%
# Find common samples
common_samples = list(set(rna.index) & set(cna.index) & set(purity.index))
print(f"Common samples across all data types: {len(common_samples)}")

# Subset to common samples
rna = rna.loc[common_samples]
cna = cna.loc[common_samples]
purity = purity.loc[common_samples]

# Find available cancer genes in the data
available_genes = list(set(cancer_genes) & set(rna.columns) & set(cna.columns))
print(f"Available cancer genes in data: {len(available_genes)}")

# If too few cancer genes, use top variable genes
if len(available_genes) < 50:
    print("Using top variable genes instead...")
    rna_var = rna.var()
    top_genes = rna_var.nlargest(min(300, len(rna_var))).index.tolist()
    available_genes = list(set(top_genes) & set(cna.columns))
    print(f"Selected {len(available_genes)} genes based on variance")

# Subset to selected genes
rna_subset = rna[available_genes]
cna_subset = cna[available_genes]

# Create combined feature matrix
genetic_data = pd.concat([rna_subset, cna_subset], axis=1, keys=['rna', 'cna'])
print(f"Combined feature matrix shape: {genetic_data.shape}")

# %% [markdown]
# ## Split Data

# %%
# Create train/test split
train_idx, test_idx = train_test_split(
    range(len(common_samples)), 
    test_size=0.2, 
    random_state=42,
    stratify=(purity > purity.median()).astype(int)
)

# Prepare data arrays
x_train = genetic_data.iloc[train_idx].values.astype(np.float32)
x_test = genetic_data.iloc[test_idx].values.astype(np.float32)
y_train = purity.iloc[train_idx].values.reshape(-1, 1).astype(np.float32)
y_test = purity.iloc[test_idx].values.reshape(-1, 1).astype(np.float32)

print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")
print(f"Y train range: [{y_train.min():.3f}, {y_train.max():.3f}]")

# %% [markdown]
# ## Build P-NET Model

# %%
def build_pnet_regression_model(n_features, n_genes, config):
    """Build P-NET model for regression."""
    
    inputs = tf.keras.Input(shape=(n_features,), name='genetic_features')
    
    # Gene layer - reduce features to genes
    x = tf.keras.layers.Dense(
        n_genes, 
        activation=config['activation'],
        kernel_regularizer=tf.keras.regularizers.l2(config['w_reg']),
        name='gene_layer'
    )(inputs)
    x = tf.keras.layers.Dropout(config['dropout'])(x)
    
    # Pathway layers with progressive reduction
    layer_sizes = [n_genes]
    for i in range(config['n_hidden_layers']):
        n_units = max(10, layer_sizes[-1] // 2)
        layer_sizes.append(n_units)
        
        x = tf.keras.layers.Dense(
            n_units,
            activation=config['activation'],
            kernel_regularizer=tf.keras.regularizers.l2(config['w_reg']),
            name=f'pathway_layer_{i+1}'
        )(x)
        x = tf.keras.layers.Dropout(config['dropout'])(x)
    
    # Output layer for regression (no activation)
    outputs = tf.keras.layers.Dense(1, activation='linear', name='purity_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='PNET_Regression')
    
    return model

# Build model
model = build_pnet_regression_model(x_train.shape[1], len(available_genes), config)

# Compile model
optimizer = Adam(learning_rate=config['learning_rate'])
if config['use_weighted_loss']:
    loss = WeightedMSELoss()
else:
    loss = 'mse'

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['mae', 'mse']
)

model.summary()

# %% [markdown]
# ## Train Model

# %%
# Setup callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# Train model
print("Training model...")
history = model.fit(
    x_train, y_train,
    batch_size=config['batch_size'],
    epochs=config['epochs'],
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# %% [markdown]
# ## Plot Training History

# %%
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History - Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training History - MAE')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Evaluate Model on Test Set

# %%
# Get predictions
y_pred = model.predict(x_test, verbose=0)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)
correlation, p_value = pearsonr(y_test.flatten(), y_pred.flatten())

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Pearson correlation: {correlation:.4f} (p={p_value:.4e})")

# %% [markdown]
# ## Visualize Predictions

# %%
# Create visualization
plt.figure(figsize=(8, 6))

# Create DataFrame for plotting
df_results = pd.DataFrame({
    'True Purity': y_test.flatten(),
    'Predicted Purity': y_pred.flatten()
})

# Scatter plot with regression line
plt.scatter(df_results['True Purity'], df_results['Predicted Purity'], 
            color='#41B6E6', alpha=0.6)

# Add regression line
z = np.polyfit(df_results['True Purity'], df_results['Predicted Purity'], 1)
p = np.poly1d(z)
plt.plot(df_results['True Purity'], p(df_results['True Purity']), "r-", alpha=0.8)

# Add diagonal line
plt.plot([0, 1], [0, 1], color='#FFA300', linestyle='--', 
         label='Perfect prediction', linewidth=2)

# Add metrics text
plt.text(0.05, 0.95, 
         f'Correlation: {correlation:.3f}\nR²: {r2:.3f}\nMSE: {mse:.3f}', 
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel('True Tumor Purity')
plt.ylabel('Predicted Tumor Purity')
plt.title('SKCM Tumor Purity Prediction')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analyze Prediction Errors

# %%
# Calculate errors
errors = y_test.flatten() - y_pred.flatten()
abs_errors = np.abs(errors)

# Plot error distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error (True - Predicted)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.scatter(y_test, abs_errors, alpha=0.6)
plt.xlabel('True Tumor Purity')
plt.ylabel('Absolute Error')
plt.title('Absolute Error vs True Purity')
# Add trend line
z = np.polyfit(y_test.flatten(), abs_errors, 2)
p = np.poly1d(z)
x_trend = np.linspace(0, 1, 100)
plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Train with Weighted Loss
# 
# Let's retrain the model with weighted MSE loss to see if it improves predictions for extreme purity values.

# %%
# Rebuild model with weighted loss
config['use_weighted_loss'] = True

model_weighted = build_pnet_regression_model(x_train.shape[1], len(available_genes), config)

model_weighted.compile(
    optimizer=Adam(learning_rate=config['learning_rate']),
    loss=WeightedMSELoss(),
    metrics=['mae', 'mse']
)

print("Training model with weighted loss...")
history_weighted = model_weighted.fit(
    x_train, y_train,
    batch_size=config['batch_size'],
    epochs=config['epochs'],
    validation_data=(x_test, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=0
)

# Get predictions with weighted model
y_pred_weighted = model_weighted.predict(x_test, verbose=0)

# Calculate metrics
mse_weighted = mean_squared_error(y_test, y_pred_weighted)
r2_weighted = r2_score(y_test, y_pred_weighted)
corr_weighted, _ = pearsonr(y_test.flatten(), y_pred_weighted.flatten())

print(f"\nWeighted Loss Model Results:")
print(f"Test MSE: {mse_weighted:.4f} (vs {mse:.4f} standard)")
print(f"Test R²: {r2_weighted:.4f} (vs {r2:.4f} standard)")
print(f"Correlation: {corr_weighted:.4f} (vs {correlation:.4f} standard)")

# %% [markdown]
# ## Compare Standard vs Weighted Loss Models

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Standard MSE
ax1 = axes[0]
df1 = pd.DataFrame({
    'True': y_test.flatten(),
    'Predicted': y_pred.flatten()
})
ax1.scatter(df1['True'], df1['Predicted'], color='#41B6E6', alpha=0.6)
# Add regression line
z1 = np.polyfit(df1['True'], df1['Predicted'], 1)
p1 = np.poly1d(z1)
ax1.plot(df1['True'], p1(df1['True']), "b-", alpha=0.8)
ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)
ax1.set_title(f'Standard MSE Loss\nR² = {r2:.3f}')
ax1.set_xlabel('True Purity')
ax1.set_ylabel('Predicted Purity')
ax1.grid(True, alpha=0.3)

# Weighted MSE
ax2 = axes[1]
df2 = pd.DataFrame({
    'True': y_test.flatten(),
    'Predicted': y_pred_weighted.flatten()
})
ax2.scatter(df2['True'], df2['Predicted'], color='#FF6B6B', alpha=0.6)
# Add regression line
z2 = np.polyfit(df2['True'], df2['Predicted'], 1)
p2 = np.poly1d(z2)
ax2.plot(df2['True'], p2(df2['True']), "r-", alpha=0.8)
ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
ax2.set_title(f'Weighted MSE Loss\nR² = {r2_weighted:.3f}')
ax2.set_xlabel('True Purity')
ax2.set_ylabel('Predicted Purity')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Feature Importance Analysis (Simplified)
# 
# Since we built a custom model, we can examine the weights of the gene layer to identify important genes.

# %%
# Get gene layer weights
gene_layer = model.get_layer('gene_layer')
gene_weights = gene_layer.get_weights()[0]  # Shape: (n_features, n_genes)

# Calculate importance as mean absolute weight per gene
# Split weights for RNA and CNA
n_genes = len(available_genes)
rna_weights = gene_weights[:n_genes, :]
cna_weights = gene_weights[n_genes:, :]

# Calculate importance scores
rna_importance = np.mean(np.abs(rna_weights), axis=0)
cna_importance = np.mean(np.abs(cna_weights), axis=0)
total_importance = rna_importance + cna_importance

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Gene': available_genes,
    'RNA_Importance': rna_importance,
    'CNA_Importance': cna_importance,
    'Total_Importance': total_importance
}).sort_values('Total_Importance', ascending=False)

# Plot top important genes
top_n = 20
plt.figure(figsize=(10, 6))

top_genes = importance_df.head(top_n)
x = np.arange(top_n)
width = 0.35

plt.bar(x - width/2, top_genes['RNA_Importance'], width, label='RNA', alpha=0.8)
plt.bar(x + width/2, top_genes['CNA_Importance'], width, label='CNA', alpha=0.8)

plt.xlabel('Genes')
plt.ylabel('Importance Score')
plt.title(f'Top {top_n} Most Important Genes for Tumor Purity Prediction')
plt.xticks(x, top_genes['Gene'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

print("Top 10 most important genes:")
print(importance_df.head(10))

# %% [markdown]
# ## Conclusions
# 
# This notebook demonstrates:
# 1. Using P-NET architecture for regression tasks (tumor purity prediction)
# 2. Loading data from remote sources (GitHub)
# 3. Implementing custom loss functions (Weighted MSE)
# 4. Comparing different loss functions for improved performance
# 5. Basic feature importance analysis
# 
# The weighted MSE loss can help improve predictions for samples with extreme purity values by giving them more weight during training.
# SKCM Tumor Purity Prediction with P-NET (TensorFlow 2.x)

This directory contains the TensorFlow 2.x adaptation of the SKCM tumor purity prediction workflow, demonstrating P-NET for regression tasks.

## Files Overview

1. **`run_skcm_purity_tf2.py`** - Main Python script for training
2. **`notebooks/SKCM_purity_tf2.ipynb`** - Interactive Jupyter notebook version
3. **`notebooks/run_skcm_purity_tf2.ipynb`** - Script-based notebook version
4. **`config/skcm_purity_params.yml`** - Configuration file with parameters

## Key Features

### Regression Task
Unlike the prostate cancer classification example, this demonstrates:
- **Continuous target prediction**: Tumor purity values (0-1)
- **MSE loss**: Appropriate for regression
- **Custom weighted loss**: Penalizes errors on extreme purity values
- **Regression metrics**: MSE, MAE, R², Pearson correlation

### Data Sources
The script attempts to load data from GitHub:
- SKCM RNA expression data
- SKCM copy number alteration (CNA) data
- TCGA tumor purity estimates
- Cancer gene lists

If data loading fails, synthetic data is generated for demonstration.

## How to Run

### Option 1: Command Line Script

```bash
cd /procedure/pnet_prostate_paper

# Run with standard MSE loss
python scripts/run_skcm_purity_tf2.py

# Run with weighted MSE loss
python scripts/run_skcm_purity_tf2.py --use-weighted-loss

# Run with custom config
python scripts/run_skcm_purity_tf2.py --config path/to/config.yml
```

### Option 2: Jupyter Notebook

```bash
jupyter lab notebooks/SKCM_purity_tf2.ipynb
```

The notebook version includes:
- Interactive data exploration
- Visualization of purity distribution
- Comparison of standard vs weighted loss
- Feature importance analysis

## Configuration Parameters

Edit `config/skcm_purity_params.yml` to modify:

```yaml
model_params:
  n_hidden_layers: 2      # Pathway layers
  activation: 'relu'      # Activation function
  dropout: 0.3           # Dropout rate
  learning_rate: 0.001   # Learning rate
  epochs: 100            # Training epochs

training_params:
  use_weighted_loss: true  # Enable weighted MSE
```

## Weighted MSE Loss

The custom loss function gives more weight to samples with extreme purity values:

```python
weights = 1 + |y_true - 0.5|
loss = mean(weights * (y_true - y_pred)²)
```

This helps the model better predict samples with very high or very low tumor purity.

## Expected Output

1. **Regression plot**: Scatter plot of true vs predicted purity
2. **Metrics**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R² score
   - Pearson correlation coefficient
3. **Feature importance**: Top genes contributing to predictions

## Output Files

- `/checkpoints/skcm_purity/best_model.weights.h5` - Best model weights
- `/results/skcm_purity_regression.png` - Regression visualization
- `/results/skcm_purity_metrics.yaml` - Evaluation metrics
- `/results/skcm_purity_training.log` - Training log

## Differences from PyTorch Version

| PyTorch | TensorFlow 2.x |
|---------|----------------|
| `Pnet.run()` | Custom regression model |
| `torch.nn.MSELoss` | `tf.keras.losses.mse` |
| `WeightedMSELoss` class | TF2 `Loss` subclass |
| PyTorch DataLoader | Direct numpy arrays |
| `model.interpret()` | Weight analysis |

## Troubleshooting

1. **Data loading fails**: The script will automatically use synthetic data
2. **Memory issues**: Reduce `batch_size` or number of genes
3. **Poor predictions**: Try adjusting:
   - Learning rate
   - Number of hidden layers
   - Dropout rate
   - Use weighted loss for extreme values

## Dependencies

```bash
pip install tensorflow scikit-learn matplotlib seaborn pandas scipy pyyaml
```

## Notes

- This is a regression task (continuous output) vs classification in the prostate example
- The model architecture is simplified compared to full P-NET due to lack of pathway annotations
- External validation on Liu 2019 cohort requires additional data download
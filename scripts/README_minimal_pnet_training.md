# Minimal P-NET Training for TensorFlow 2.x

This directory contains the adapted P-NET training workflow for the prostate cancer project, converted from PyTorch to TensorFlow 2.x.

## Files Overview

1. **`run_minimal_pnet_training_tf2.py`** - Main Python script for training
2. **`notebooks/run_minimal_pnet_training_tf2.ipynb`** - Jupyter notebook version of the script
3. **`config/minimal_training_params.yml`** - Configuration file with all parameters

## How to Run

### Option 1: Python Script (Command Line)

```bash
# From the project root directory
cd /procedure/pnet_prostate_paper

# Run with default configuration
python scripts/run_minimal_pnet_training_tf2.py

# Run with custom configuration file
python scripts/run_minimal_pnet_training_tf2.py --config /path/to/your/config.yml
```

### Option 2: Jupyter Notebook

1. Start Jupyter Lab or Jupyter Notebook:
   ```bash
   cd /procedure/pnet_prostate_paper
   jupyter lab notebooks/run_minimal_pnet_training_tf2.ipynb
   ```

2. Run all cells sequentially or modify the notebook for interactive exploration.

## Configuration

The training parameters are defined in `config/minimal_training_params.yml`. Key sections include:

- **`data_params`**: Data loading and preprocessing parameters
- **`model_params`**: P-NET model architecture and training parameters  
- **`training_params`**: Training loop configuration, callbacks, and output settings

### Key Parameters to Modify

- `model_params.epochs`: Number of training epochs (default: 50)
- `model_params.batch_size`: Training batch size (default: 32)
- `model_params.learning_rate`: Learning rate for Adam optimizer (default: 0.001)
- `model_params.n_hidden_layers`: Number of hidden pathway layers (default: 1)
- `training_params.random_seed`: Random seed for reproducibility (default: 42)

## Output Files

The script will generate:

1. **Results directory** (`/procedure/pnet_prostate_paper/results/`):
   - `minimal_pnet_roc_curve.png` - ROC curve plot
   - `evaluation_metrics.yaml` - Final model performance metrics
   - `model_summary.txt` - Model architecture summary
   - `feature_names.yaml` - Feature names used by the model
   - `minimal_pnet_training.log` - Training log file

2. **Checkpoints directory** (`/procedure/pnet_prostate_paper/checkpoints/minimal_pnet/`):
   - `best_model.h5` - Best model weights during training

## Expected Output

The script demonstrates:
- Loading minimal prostate dataset using the project's `Data` class
- Building P-NET model with pathway-informed architecture
- Training with TensorFlow 2.x and Keras
- Evaluation with ROC-AUC metrics
- ROC curve visualization

## Dependencies

The following packages are required and have been installed:
- TensorFlow 2.x (`pip install tensorflow`)
- scikit-learn (`pip install scikit-learn`)
- matplotlib (`pip install matplotlib`)
- PyYAML (`pip install pyyaml`)
- numpy (installed with TensorFlow)
- pandas (`pip install pandas`)
- jupytext (for script-to-notebook conversion: `pip install jupytext`)

If running in a new environment, install all dependencies with:
```bash
pip install tensorflow scikit-learn matplotlib pandas pyyaml jupytext
```

## Script-to-Notebook Conversion

To recreate the notebook from the Python script:

```bash
# Install jupytext if not already available
pip install jupytext

# Convert script to notebook
cd /procedure/pnet_prostate_paper
jupytext --to ipynb scripts/run_minimal_pnet_training_tf2.py -o notebooks/run_minimal_pnet_training_tf2.ipynb
```

## Troubleshooting

1. **Data loading errors**: Ensure the minimal dataset exists at the expected paths in `test_data/minimal_prostate_set/`
2. **Memory issues**: Reduce `batch_size` in the configuration file
3. **Training convergence**: Adjust `learning_rate`, `epochs`, or `patience` parameters
4. **Path errors**: Ensure all paths in the configuration file are absolute and correct

## Adaptation Notes

This script was adapted from the original PyTorch P-NET implementation with the following key changes:

1. **Framework**: PyTorch → TensorFlow 2.x/Keras
2. **Data loading**: Custom PyTorch loader → Project's `Data` class
3. **Model building**: PyTorch model → `build_pnet2` function
4. **Training loop**: PyTorch training → Keras `model.fit()`
5. **Evaluation**: PyTorch metrics → scikit-learn + TensorFlow metrics

The core workflow and intent remain the same: demonstrate end-to-end P-NET training for binary classification on genomic data.
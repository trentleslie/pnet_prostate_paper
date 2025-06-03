# Implementation Notes: P-NET Full Training Pipeline Integration Testing

## Date: 2025-06-03

### Progress:

- Feature transitioned from planning to in-progress stage
- Analyzed existing codebase for data sources and model builders
- Identified key implementation decisions based on existing infrastructure

### Decisions Made:

#### 1. Data Strategy: Simplified Real Data Approach

After investigating the paper.txt file and existing data infrastructure, the recommended approach is to use **simplified real data** for integration testing:

- **Primary Data Source**: The project already has a minimal dataset creator and existing minimal test data in `/test_data/minimal_prostate_set/` with 12 training, 4 validation, and 4 test samples
- **Data Access Options**:
  1. Use the existing minimal dataset as a starting point (quickest option)
  2. Run `scripts/create_minimal_dataset.py` to generate a fresh minimal dataset
  3. Download full data from Zenodo (https://doi.org/10.5281/zenodo.5163213) and subsample
  4. Use existing download scripts (`data/prostate_paper/download_data.py`) to fetch and prepare data

- **Recommended Approach**: The initial implementation will use the existing minimal dataset (12 training samples) located in `/test_data/minimal_prostate_set/`. This allows for rapid initial testing of the pipeline. Creation of a larger subset (50-100 samples) can be considered as a follow-up task after the basic end-to-end flow is confirmed.

- **Data Format**: The data consists of:
  - Somatic mutations
  - Copy number alterations (CNAs)
  - Gene fusions (optional)
  - Binary labels (Primary vs Metastatic)

#### 2. Test Framework Location

- **New Directory Structure**: `/procedure/pnet_prostate_paper/integration_tests/`
  - This keeps integration tests separate from unit tests
  - Allows for dedicated configuration and logging
  - Makes it easy to exclude from production deployments

- **Subdirectory Organization**:
  ```
  integration_tests/
  ├── __init__.py
  ├── configs/           # YAML test configurations
  ├── data/             # Test datasets
  ├── scripts/          # Test execution scripts
  ├── logs/             # Training logs
  ├── results/          # Test results and reports
  └── utils/            # Helper functions
  ```
- **Test File Naming**: Integration test scripts will follow the naming convention `test_integration_*.py`.

#### 3. Model Variant Selection

Based on analysis of `model/builders/prostate_models.py`, the initial implementation should focus on:

- **Primary Model**: `build_pnet` with `n_hidden_layers=1`
  - This creates a P-NET with a single sparse pathway layer
  - Simpler architecture reduces potential failure points
  - Still representative of the P-NET approach

- **Configuration Parameters**:
  ```python
  model_config = {
      'n_hidden_layers': 1,
      'dropout': 0.5,
      'sparse': True,
      'activation': 'tanh',
      'w_reg': 0.001,
      'use_bias': False,
      'add_unk_genes': True,
      'direction': 'root_to_leaf'
  }
  ```

- **Future Extensions**: Once basic model works, test with:
  - `n_hidden_layers=2` for deeper architecture
  - `build_pnet2` for newer implementation with additional features
  - Dense baseline models for comparison

#### 4. Resource Usage & Performance Considerations

- **Training Parameters**:
  - Initial epochs: 5 (sufficient to verify convergence)
  - Batch size: 4-8 (small due to limited samples)
  - Learning rate: 0.001 (standard Adam default)
  
- **Performance Monitoring**:
  - Log training time per epoch using `time.time()`
  - Track peak memory usage with `tracemalloc` or `psutil`
  - Include these metrics in test reports but don't enforce strict limits initially

- **Hardware Assumptions**:
  - Should run on standard development machines (8GB RAM minimum)
  - GPU optional but beneficial
  - Total test runtime target: <10 minutes for full suite

#### 5. CI/CD Integration

- **Current Scope**: Not required for initial implementation
- **Future Considerations**:
  - Tests should be designed to be CI-friendly (deterministic, resource-conscious)
  - Consider creating a separate lightweight test profile for CI
  - Document how to integrate with existing test infrastructure

#### 6. Logging Strategy

- **Framework**: Python's standard `logging` module will be used for logging within the integration tests.
- **Configuration**: Basic logging configuration (e.g., logging to console and/or a file in the `integration_tests/logs/` directory) will be set up.
- **Verbosity**: Log levels should be configurable to allow for detailed debugging when needed.

#### 7. Model Checkpointing

- **Initial Implementation**: Basic model checkpointing (e.g., saving model weights at the end of training or if validation loss improves) will be included in the initial implementation.
- **Purpose**: This will aid in debugging and allow for inspection of trained models, even with the small dataset.
- **Location**: Checkpoints can be saved to a subdirectory within `integration_tests/results/` or a dedicated `integration_tests/checkpoints/` directory.

### Challenges Encountered:

- The codebase has ongoing TensorFlow 1.x to 2.x migration, need to ensure integration tests use TF2-compatible code
- Multiple model builder variants exist (`build_pnet` vs `build_pnet2`), requiring careful selection
- Data pipeline has multiple entry points, need to identify the most appropriate for testing

### Next Steps:

1. Create the integration test directory structure
2. Implement a basic data loader using the existing minimal dataset
3. Create a minimal training script that instantiates a simple P-NET model
4. Add gradient and loss validation checks
5. Extend to more comprehensive test scenarios

### Technical Notes:

#### Data Loading Strategy

The existing `Data` class in `data/data_access.py` provides a high-level interface. For integration testing:

```python
from data.data_access import Data

# Use existing minimal dataset
data_params = {
    'data_type': 'prostate_paper',
    'data_path': '/test_data/minimal_prostate_set/',
    'fold': 0,
    'seed': 42
}

data = Data(**data_params)
x_train, y_train, info_train, cols = data.get_train_data()
x_val, y_val, info_val, _ = data.get_validation_data()
```

#### Model Instantiation Pattern

Based on existing code patterns:

```python
from model.builders.prostate_models import build_pnet
import tensorflow as tf

# Model configuration
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model, feature_names = build_pnet(
    optimizer=optimizer,
    w_reg=0.001,
    data_params=data_params,
    n_hidden_layers=1,
    sparse=True,
    activation='tanh'
)
```

#### Gradient Validation Implementation

Key checks to implement:

1. **Non-zero gradients**: Ensure gradients are computed for all trainable variables
2. **Finite values**: Check for NaN or Inf in gradients
3. **Reasonable magnitudes**: Gradients should neither vanish nor explode

```python
@tf.function
def train_step(model, x, y, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Validation checks
    for grad, var in zip(gradients, model.trainable_variables):
        if grad is None:
            raise ValueError(f"No gradient for {var.name}")
        if tf.reduce_any(tf.math.is_nan(grad)):
            raise ValueError(f"NaN gradient for {var.name}")
        if tf.reduce_any(tf.math.is_inf(grad)):
            raise ValueError(f"Inf gradient for {var.name}")
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Integration Test Architecture

The integration test framework will follow a modular architecture:

1. **Configuration Layer**: YAML-based test scenarios
2. **Data Layer**: Simplified data loading with validation
3. **Model Layer**: Model instantiation with architecture validation
4. **Training Layer**: Training loop with health checks
5. **Validation Layer**: Comprehensive checks and metrics
6. **Reporting Layer**: Structured output generation

This architecture ensures extensibility and maintainability while providing comprehensive validation of the P-NET training pipeline.
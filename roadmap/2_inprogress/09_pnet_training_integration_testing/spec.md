# Feature Specification: P-NET Full Training Pipeline Integration Testing

## 1. Functional Scope

The integration testing framework will provide comprehensive validation of P-NET models in realistic training scenarios. The system will:

### Core Functionality
- **Training Script Development**: Create a modular training script that can:
  - Load training, validation, and test datasets (either mock or simplified real data)
  - Instantiate P-NET models using the debugged builder functions
  - Execute standard training loops with configurable parameters
  - Log metrics and training progress
  - Save checkpoints and final models

- **Data Pipeline**: Implement robust data handling that:
  - Generates or loads mock genomic datasets (mutations, CNAs)
  - Provides pathway map information compatible with P-NET architecture
  - Ensures proper data formatting and batching
  - Supports both TensorFlow Dataset API and NumPy array inputs

- **Model Configuration**: Enable flexible model instantiation:
  - Support all P-NET variants (pnet, pnet2)
  - Allow easy parameter configuration (architecture, regularization, etc.)
  - Provide sensible defaults for quick testing
  - Support both genomic-only and multi-modal configurations

- **Validation Metrics**: Track essential training indicators:
  - Loss values (training and validation)
  - Gradient flow health checks
  - Memory usage and training speed
  - Basic performance metrics (if applicable to test data)

## 2. Technical Scope

### Dependencies and Constraints
- **TensorFlow Compatibility**: Must work with TensorFlow 2.x migration efforts
- **Existing Codebase Integration**: Leverage debugged components from:
  - `/model/builders/builders_utils.py` (get_pnet function)
  - `/model/layers_custom_tf2.py` (custom layers)
  - Existing data loading utilities where applicable

### Performance Requirements
- Training should complete within reasonable time for CI/CD integration
- Memory usage should stay within typical development machine limits
- Support for both CPU and GPU execution

### Testing Infrastructure
- Integration with existing test framework
- Automated validation of successful training completion
- Clear error reporting for debugging failures

## 3. Implementation Approaches

### Option A: Minimal Mock Data Approach

Create a lightweight integration test using purely synthetic data:

```python
# Pseudo-code structure
def test_pnet_training_minimal():
    # Generate small mock dataset
    mock_data = generate_mock_genomic_data(n_samples=100, n_features=1000)
    mock_pathways = generate_mock_pathway_map(n_pathways=50)
    
    # Build model
    model = get_pnet(input_shapes, pathway_map=mock_pathways, ...)
    
    # Train for few epochs
    history = train_model(model, mock_data, epochs=3)
    
    # Validate results
    assert history.loss[-1] < history.loss[0]
    assert no_nan_in_gradients(model)
```

**Pros:**
- Fast execution, suitable for CI/CD
- Complete control over data characteristics
- Easy to debug and reproduce issues
- No external data dependencies

**Cons:**
- May not catch real-world data edge cases
- Limited biological relevance
- Might miss data formatting issues

### Option B: Simplified Real Data Approach

Use a subset of actual genomic data with simplified preprocessing:

```python
# Pseudo-code structure
def test_pnet_training_real_subset():
    # Load subset of real data
    data = load_simplified_prostate_data(n_samples=500)
    pathways = load_reactome_pathways(filter='cancer-relevant')
    
    # Build model with realistic configuration
    model = get_pnet(data.shapes, pathway_map=pathways, 
                     architecture='standard', ...)
    
    # Train with realistic parameters
    history = train_model(model, data, epochs=10, 
                         validation_split=0.2)
    
    # Comprehensive validation
    validate_training_metrics(history)
    validate_model_predictions(model, data.test)
```

**Pros:**
- Tests realistic data scenarios
- Validates biological pathway integration
- More confidence in real-world applicability
- Can serve as baseline for performance benchmarks

**Cons:**
- Slower execution
- Requires data preprocessing pipeline
- More complex debugging
- Data availability dependencies
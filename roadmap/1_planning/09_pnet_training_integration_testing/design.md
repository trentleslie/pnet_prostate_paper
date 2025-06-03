# Design Document: P-NET Full Training Pipeline Integration Testing

## 1. Architectural Considerations

### Overall System Architecture

The integration testing framework will be structured as a modular system with clear separation of concerns:

```
Integration Test Framework
├── Data Generation Module
│   ├── Mock Data Generator
│   ├── Real Data Loader
│   └── Data Validation
├── Model Building Module
│   ├── P-NET Configuration
│   ├── Model Instantiation
│   └── Architecture Validation
├── Training Pipeline Module
│   ├── Training Loop
│   ├── Metric Collection
│   └── Checkpoint Management
└── Validation Module
    ├── Gradient Health Checks
    ├── Loss Validation
    └── Performance Metrics
```

### Key Design Principles

1. **Modularity**: Each component should be independently testable and reusable
2. **Configurability**: Support different test scenarios through configuration files
3. **Extensibility**: Easy to add new test cases or model variants
4. **Observability**: Comprehensive logging and metric tracking

### Data Model

The integration test will work with the following data structures:

```python
# Core data structures
class IntegrationTestData:
    train_x: np.ndarray  # Shape: (n_samples, n_features)
    train_y: np.ndarray  # Shape: (n_samples, n_classes)
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    pathway_map: Dict[str, List[int]]  # Pathway to gene indices
    feature_names: List[str]
    
class TrainingConfig:
    model_type: str  # 'pnet' or 'pnet2'
    architecture_params: Dict
    training_params: Dict
    data_params: Dict
```

## 2. Component Interactions

### Data Flow Diagram

```
[Test Configuration] 
    |
    v
[Data Generator/Loader]
    |
    ├──> [Mock Data Creation]
    |    └──> Synthetic genomic features
    |         └──> Synthetic pathway maps
    |
    └──> [Real Data Loading]
         └──> Preprocessed subset
              └──> Actual pathway maps
    |
    v
[Model Builder]
    |
    ├──> get_pnet() ──> P-NET Model Instance
    |
    v
[Training Pipeline]
    |
    ├──> Forward Pass
    ├──> Loss Calculation
    ├──> Backward Pass
    └──> Optimizer Step
    |
    v
[Validation & Metrics]
    |
    ├──> Gradient Health
    ├──> Loss Tracking
    └──> Performance Metrics
    |
    v
[Test Results & Reports]
```

### Integration Points with Existing Code

1. **Model Building Integration**:
   - Uses `/model/builders/builders_utils.py::get_pnet()`
   - Leverages custom layers from `/model/layers_custom_tf2.py`
   - Applies constraints from `/model/constraints_custom.py`

2. **Data Pipeline Integration**:
   - Can utilize existing data readers from `/data/`
   - Pathway information from `/data/pathways/`
   - Preprocessing utilities from `/preprocessing/`

3. **Training Infrastructure**:
   - Adapts patterns from `/train/` scripts
   - Uses callbacks from `/model/callbacks_custom.py`
   - Leverages utilities from `/utils/`

## 3. Visual Sketches / Mockups

### Test Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  Integration Test Runner                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Initialize Test Environment                               │
│     ├─ Load configuration                                     │
│     ├─ Set random seeds                                       │
│     └─ Configure logging                                      │
│                                                               │
│  2. Prepare Data                                              │
│     ├─ Generate/Load dataset                                  │
│     ├─ Create train/val/test splits                          │
│     └─ Validate data integrity                               │
│                                                               │
│  3. Build Model                                               │
│     ├─ Configure architecture                                 │
│     ├─ Instantiate P-NET model                               │
│     └─ Compile with optimizer                                │
│                                                               │
│  4. Execute Training                                          │
│     ├─ Run training loop                                      │
│     ├─ Collect metrics                                        │
│     └─ Monitor gradient health                               │
│                                                               │
│  5. Validate Results                                          │
│     ├─ Check loss convergence                                │
│     ├─ Verify gradient flow                                  │
│     └─ Generate test report                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Configuration File Structure

```yaml
# integration_test_config.yaml
test_name: "pnet_basic_training"
model:
  type: "pnet"
  architecture:
    n_hidden_layers: 2
    hidden_layer_sizes: [512, 256]
    activation: "tanh"
    use_attention: false
    
data:
  type: "mock"  # or "real_subset"
  n_samples: 1000
  n_features: 2000
  n_pathways: 100
  
training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  
validation:
  check_gradients: true
  loss_threshold: null  # or specific value
  save_checkpoints: false
```

### Expected Output Format

```
=== P-NET Integration Test Results ===
Test Name: pnet_basic_training
Model Type: pnet
Data Type: mock (1000 samples, 2000 features)

Training Progress:
Epoch 1/5 - Loss: 0.693 - Val Loss: 0.689 - Time: 2.3s
Epoch 2/5 - Loss: 0.621 - Val Loss: 0.635 - Time: 2.1s
Epoch 3/5 - Loss: 0.584 - Val Loss: 0.612 - Time: 2.2s
Epoch 4/5 - Loss: 0.561 - Val Loss: 0.598 - Time: 2.1s
Epoch 5/5 - Loss: 0.543 - Val Loss: 0.589 - Time: 2.2s

Validation Results:
✓ Model trained without errors
✓ Loss decreased over epochs
✓ All gradients are valid (no NaN/Inf)
✓ Memory usage within limits (Peak: 1.2 GB)

Test Status: PASSED
```
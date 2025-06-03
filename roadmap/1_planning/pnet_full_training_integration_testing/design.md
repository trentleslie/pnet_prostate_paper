# Design Document: P-NET Full Training Pipeline Integration Testing

## 1. Architectural Considerations

The integration testing framework will be designed as a modular system that can validate the P-NET training pipeline at multiple levels:

### Test Architecture Layers:
1. **Data Layer Testing**: Validate data loading, preprocessing, and pathway integration
2. **Model Layer Testing**: Test model construction, layer interactions, and parameter initialization
3. **Training Layer Testing**: Verify forward/backward passes, optimization, and convergence
4. **Metrics Layer Testing**: Confirm logging, evaluation metrics, and result persistence

### Key Design Principles:
- **Isolation**: Each test component should be independently executable
- **Reproducibility**: Tests must produce consistent results across runs
- **Scalability**: Framework should handle both minimal and full-scale datasets
- **Observability**: Comprehensive logging and metrics collection for debugging
- **Compatibility**: Primarily target TensorFlow 2.x (refactored codebase). Original TensorFlow 1.x behavior (as per the Elmarakeby et al., Nature 2021 paper) serves as the baseline for validation.
- **Alignment with Paper Methodology**: Test design and execution must facilitate direct comparison with Elmarakeby et al., Nature 2021, including datasets, data splits, evaluation metrics, and key analyses.

### Data Models:
- Test configuration objects defining datasets (including specific paper cohorts like Armenia et al., external validation sets, and their prescribed train/validation/test splits from `/procedure/pnet_prostate_paper/data/_database/prostate/splits/`), model parameters (replicating or adapting original `_params.yml` configurations for the modernized codebase), and expected outcomes (benchmarked against paper-reported results).
- Result objects capturing training metrics, performance data, and validation results
- Error tracking objects for debugging failed test scenarios

## 2. Component Interactions

### Integration Test Flow:
```
Test Configuration → Data Loader → Model Builder → Training Pipeline → Metrics Collector → Validation
      ↓                  ↓             ↓               ↓                 ↓              ↓
Test Config File → data/loaders → model/builders → pipeline/train → utils/evaluate → test/results
```

### Component Dependencies:
- **Data Loading**: Utilizes existing data access patterns from `/data/data_access.py` and specifically loads the Armenia et al. cohort and external validation datasets from `/procedure/pnet_prostate_paper/data/_database/` using the data splits defined in the paper (located in `/procedure/pnet_prostate_paper/data/_database/prostate/splits/`). Will need to integrate with replicated/adapted `_params.yml` files for specific configurations.
- **Model Construction**: Leverages debugged builders from `/model/builders/prostate_models.py`
- **Training Execution**: Integrates with pipeline components from `/pipeline/train_validate.py`
- **Metrics Collection**: Uses evaluation utilities from `/utils/evaluate.py`

### Error Handling:
- Graceful degradation when components fail
- Detailed error reporting with component-specific context
- Automatic cleanup of resources on test failure
- Rollback mechanisms for persistent state changes

## 3. Implementation Strategy

### Phase 1: Foundational Integration & Minimal Data Testing
Establish the core integration testing framework. Validate the basic end-to-end pipeline (data loading, model construction using refactored TF2.x components, training loop, basic metrics calculation) using minimal, synthetic, or heavily subsetted data. Focus on ensuring the modernized components integrate correctly before proceeding to full-scale paper replication.
- Tasks: Mock data generators, simplified model configurations (not necessarily paper-aligned yet), basic training loop validation, core metrics verification.

### Phase 2: Core Paper Replication - Prediction & Validation
Focus on replicating the primary predictive performance reported in Elmarakeby et al., Nature 2021. This involves using the full Armenia et al. dataset with the specified train/validation/test splits, and the external validation datasets, all processed through the modernized P-NET (TF2.x) codebase.
- Tasks:
    - Utilize actual prostate cancer datasets (Armenia et al. cohort) with paper-specific splits from `/procedure/pnet_prostate_paper/data/_database/prostate/splits/`.
    - Implement training using configurations that replicate or adapt original `_params.yml` settings for the modernized codebase.
    - Test full P-NET model architecture (refactored to TF2.x).
    - Benchmark core prediction performance (AUC, AUPRC, Accuracy, F1) against paper-reported values.
    - Perform external validation using datasets from `/procedure/pnet_prostate_paper/data/_database/prostate/external_validation/` and compare against paper results.

### Phase 3: Advanced Paper Analyses & Extended Testing
Implement and validate more advanced analyses from the paper and conduct further robustness testing on the modernized P-NET.
- Tasks:
    - Attempt to replicate Biochemical Recurrence (BCR) prediction analysis (Fig. 2d in paper), contingent on clinical data availability and linkage.
    - Integrate and test interpretability methods (e.g., TF2.x compatible DeepLIFT/SHAP) and compare key feature/gene/pathway importance with paper findings.
    - Conduct memory profiling and performance stress testing with full datasets.
    - Test compatibility across different hardware configurations (if relevant to paper's context or reproducibility).
    - Validate long-running training stability.

### Test Organization Structure:
```
/model/testing/integration/
├── test_foundational_pipeline.py         # Phase 1: Basic pipeline validation with minimal/mock data
├── test_paper_replication_core.py        # Phase 2: Core prediction & external validation (vs. paper)
├── test_paper_replication_advanced.py    # Phase 3: BCR, interpretability (vs. paper)
├── test_performance_stability.py       # Phase 3: Stress tests, long runs, hardware configs
├── configs/                        # Test configuration files
├── fixtures/                       # Test data and mock objects
└── utils/                          # Test utility functions
```
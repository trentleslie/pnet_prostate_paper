# Feature Specification: P-NET Full Training Pipeline Integration Testing

## 1. Functional Scope

This feature will create comprehensive integration tests that validate the entire P-NET training pipeline from data loading to model training completion, **with a primary goal of replicating key analyses and performance benchmarks from Elmarakeby et al., Nature 2021, using the modernized (Python 3.11, TensorFlow 2.x) codebase.** The tests will:

- Load realistic prostate cancer datasets using existing data loading mechanisms
- Initialize P-NET models using the debugged model building functions
- Execute forward passes through the complete model architecture
- Perform backward passes with gradient computation and weight updates
- Calculate and log training metrics (loss, accuracy, AUC, etc.)
- Validate convergence behavior over multiple training epochs
- Test with different model configurations and hyperparameters
- Ensure memory usage and performance are within acceptable bounds
- **Replicate core prediction performance (primary vs. metastatic CRPC) using the main dataset and splits from the paper.**
- **Perform external validation using independent cohorts as described in the paper.**
- **Conduct Biochemical Recurrence (BCR) prediction analysis if linked clinical data is available.**
- **Implement interpretability analysis (e.g., using TF2.x compatible DeepLIFT, SHAP, or Integrated Gradients) to compare identified key features/genes/pathways with paper findings.**
- **Benchmark model performance (AUC, AUPRC, Accuracy, F1-score) against results reported in the paper.**

## 2. Technical Scope

The implementation will leverage existing codebase components and specific data assets:
- **Primary dataset (Armenia et al. cohort) and external validation datasets as referenced in Elmarakeby et al., Nature 2021, located in `/procedure/pnet_prostate_paper/data/_database/`.**
- **Specific training/validation/testing data splits from `/procedure/pnet_prostate_paper/data/_database/prostate/splits/` to ensure methodological alignment with the paper.**
- Data loading utilities from `/data/` directory
- Model builders from `/model/builders/`
- Training pipelines from `/pipeline/`
- Utility functions from `/utils/`

Technical constraints and considerations:
- **Primarily target TensorFlow 2.x (refactored codebase). Comparisons with TensorFlow 1.x behavior (as per the original paper) will be for baseline validation.**
- **Acknowledge and address the dependency on model configuration files (e.g., `_params.yml`, roadmap item `FP002`) for replicating specific experimental setups from the paper.**
- Should handle both GPU and CPU execution environments
- Need to work with existing pathway data structures and genomic data formats
- Must integrate with current logging and metrics collection systems
- Should be compatible with existing hyperparameter configurations

## 3. Testing Approaches / Implementation Options

### Option A: Minimal Dataset Integration Test

Create a lightweight test using minimal synthetic or subset datasets that focuses on pipeline functionality rather than model performance.

- Pros: Fast execution, easy to maintain, good for CI/CD
- Cons: May not catch issues with real-world data complexity

### Option B: Full Dataset Integration Test (Replicating Paper Methodology)

Use complete prostate cancer datasets in integration tests to validate real-world scenarios.

- Pros: High confidence in validating the modernized codebase against the paper's findings, directly assesses replication of results, catches data-specific issues
- Cons: Slower execution, requires larger test data storage, may be unstable due to data dependencies

### Option C: Hybrid Approach

Implement both minimal and full dataset tests, with minimal tests for regular CI and full tests for release validation.

- Pros: Balances speed and coverage, provides multiple validation levels
- Cons: More complex to maintain, requires careful test organization
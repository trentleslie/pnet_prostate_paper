# Feature: P-NET Full Training Pipeline Integration Testing

## Goal

Validate the modernized P-NET models (Python 3.11, TensorFlow 2.x, built using debugged model building functions) in an end-to-end training scenario. **The primary objective is to replicate key methodologies, analyses, and performance benchmarks from the Elmarakeby et al., Nature 2021 paper, ensuring all components work correctly when integrated within the refactored codebase.**

## Key Requirements

- Test data loading mechanisms with P-NET model architecture
- Validate model forward pass functionality in training context
- Verify model backward pass (gradient computation and weight updates)
- Confirm metrics calculation and logging work correctly
- Ensure compatibility of all components when integrated in full training pipeline
- Test with realistic dataset configurations used in prostate cancer research
- Validate training convergence and performance metrics
- **Utilize the primary (Armenia et al. cohort) and external validation datasets as specified in Elmarakeby et al., 2021, using the exact data splits from `/procedure/pnet_prostate_paper/data/_database/prostate/splits/`.**
- **Replicate core prediction performance (primary vs. metastatic CRPC) and external validation results as reported in the paper.**
- **Attempt to replicate Biochemical Recurrence (BCR) prediction analysis, contingent on availability and linkage of clinical data.**
- **Implement and validate interpretability analysis (e.g., using TF2.x compatible DeepLIFT, SHAP, or Integrated Gradients), comparing identified key features/genes/pathways with paper findings.**
- **Benchmark model performance (AUC, AUPRC, Accuracy, F1-score) against those reported in the paper.**
- **Ensure tests are compatible with the refactored TensorFlow 2.x architecture and address dependencies such as replicating or adapting original `_params.yml` configurations.**

## Target Audience

- Machine learning engineers working on P-NET model development
- Researchers using P-NET for prostate cancer analysis
- Quality assurance team validating model pipeline functionality
- Project maintainers ensuring system reliability

## Open Questions

- **How will the specific `_params.yml` configurations used in the original paper be precisely replicated or appropriately adapted for the modernized codebase to ensure fair comparison (addresses roadmap item `FP002`)?**
- **What level of deviation from the paper's reported performance metrics will be considered acceptable for the refactored model to be deemed a successful replication?**
- **While the paper provides baselines, should the integration tests *also* re-implement and run these baselines (e.g., Logistic Regression, SVM) within the new framework for direct comparison, or is comparison against reported numbers sufficient?**
- How should we handle different hardware configurations (GPU/CPU) and ensure reproducibility of results across them, especially if the original paper had specific hardware notes?
- What specific logging level and detail in metrics collection will be needed for effectively debugging discrepancies from paper results?
- How do we ensure that batch sizes, learning rates, and other critical training configurations precisely match or are appropriately adapted from the original paper's setup?
- **What is the strategy for the Biochemical Recurrence (BCR) analysis if the required linked clinical outcome data is incomplete or unavailable for the datasets at hand?**
- **Are there specific versions of external libraries (beyond core Python/TF) used in the original paper that need to be considered for faithful replication of the environment, if not already handled by `environment.yml`?**